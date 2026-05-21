"""
TRC3500 Project 3 – Breath Rate Test Suite
===========================================
Runs 6 sequential breath-rate tests at: 30, 25, 20, 15, 10, 5 BPM.
In each test the tester breathes 10 times following an audible metronome,
and the conductive-rubber sensor's detected breath count is recorded.

Output: bpm_test_results.csv with the following columns
  test_number, target_bpm, expected_breaths, detected_breaths,
  error, error_percent, test_duration_s

Usage
-----
1. Connect STM32 and set SERIAL_PORT below (same value used in listen4.py).
2. Run: python bpm_test_suite.py
3. Press ENTER before each test, then breathe in time with the beeps.
4. After the 6th test, results are printed and saved to CSV automatically.
"""

import csv
import os
import sys
import time
import threading

import serial

# Optional Windows beeper for the metronome
try:
    import winsound
    HAS_BEEP = True
except ImportError:
    HAS_BEEP = False


# ─────────────────────────────── Configuration ────────────────────────────────
SERIAL_PORT       = "COM6"           # must match STM32 port (same as listen4.py)
BAUD_RATE         = 115200
SAMPLE_RATE       = 100              # Hz – informational only
BREATH_THRESHOLD  = 200              # ADC counts (same as listen4.py)

TEST_BPMS         = [60, 50, 40, 30, 20, 10]
BREATHS_PER_TEST  = 10
REST_BETWEEN_S    = 15               # seconds rest between tests

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "bpm_test_results.csv")


# ─────────────────────────────── Metronome ────────────────────────────────────

def metronome_worker(interval_s, num_beats, stop_event):
    """Beep + print on a separate thread so serial reading is not blocked."""
    start = time.time()
    for i in range(num_beats):
        target = start + i * interval_s
        delay = target - time.time()
        if delay > 0:
            if stop_event.wait(delay):
                return
        if stop_event.is_set():
            return
        if HAS_BEEP:
            try:
                winsound.Beep(880, 80)         # short A5 beep, 80 ms
            except RuntimeError:
                print("\a", end="", flush=True)
        else:
            print("\a", end="", flush=True)
        print(f"  ♪ Beat {i + 1}/{num_beats}", flush=True)


# ─────────────────────────────── Breath Detection ─────────────────────────────

class BreathCounter:
    """
    State-machine threshold breath detector (same approach as listen4.py).
    Counts full inhale–exhale cycles using two half-cycle crossings.
    """

    def __init__(self, threshold, debounce_s):
        self.threshold      = threshold
        self.debounce_s     = debounce_s
        self.extreme_val    = None
        self.direction      = None        # "up" or "down"
        self.half_cycles    = 0
        self.total_breaths  = 0
        self.last_peak_t    = -999.0

    def update(self, sample, t):
        if self.extreme_val is None:
            self.extreme_val = sample
            return False

        crossed = False
        if self.direction == "down":
            if sample < self.extreme_val:
                self.extreme_val = sample
            if sample - self.extreme_val >= self.threshold:
                self.direction   = "up"
                self.extreme_val = sample
                self.half_cycles += 1
                crossed = True
        else:
            if sample > self.extreme_val:
                self.extreme_val = sample
            if self.extreme_val - sample >= self.threshold:
                self.direction   = "down"
                self.extreme_val = sample
                self.half_cycles += 1
                crossed = True

        if crossed and self.half_cycles % 2 == 0:
            if t - self.last_peak_t >= self.debounce_s:
                self.total_breaths += 1
                self.last_peak_t = t
                return True
        return False


# ─────────────────────────────── One Test Case ────────────────────────────────

def run_test(ser, target_bpm):
    interval_s     = 60.0 / target_bpm
    total_duration = interval_s * BREATHS_PER_TEST
    # Debounce must be shorter than the breath interval so we do not miss
    # legitimate breaths at high BPM.
    debounce_s     = max(0.5, 0.4 * interval_s)

    counter = BreathCounter(BREATH_THRESHOLD, debounce_s)

    print()
    print(f"┌──────────────────────────────────────────────┐")
    print(f"│  TEST: {target_bpm:>2} BPM  →  {BREATHS_PER_TEST} breaths in "
          f"{total_duration:5.1f}s   │")
    print(f"│  Breath interval: {interval_s:5.2f}s   "
          f"Debounce: {debounce_s:4.2f}s   │")
    print(f"└──────────────────────────────────────────────┘")
    input("  Press ENTER when ready to start …")

    # 3-second countdown
    for n in (3, 2, 1):
        print(f"  Starting in {n} …", flush=True)
        time.sleep(1.0)
    print("  GO!  Inhale on each beep.\n", flush=True)

    # Flush any stale serial data captured during the prompt / countdown
    ser.reset_input_buffer()

    stop_event = threading.Event()
    metro = threading.Thread(
        target=metronome_worker,
        args=(interval_s, BREATHS_PER_TEST, stop_event),
        daemon=True,
    )
    metro.start()

    start = time.time()

    try:
        while True:
            now = time.time() - start
            if now >= total_duration:
                break

            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue

            try:
                parts = raw_line.split(",")
                if len(parts) != 2:
                    continue
                rubber_val = float(parts[0])
            except ValueError:
                continue

            counter.update(rubber_val, now)
    finally:
        stop_event.set()
        metro.join(timeout=1.0)

    detected = counter.total_breaths
    print(f"\n  → Detected {detected} breaths "
          f"(expected {BREATHS_PER_TEST})\n")
    return detected, total_duration


# ─────────────────────────────────── Main ─────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  TRC3500 P3 – Breath Rate Test Suite         ║")
    print("║  6 tests:  30, 25, 20, 15, 10, 5  BPM        ║")
    print("║  10 breaths per test                         ║")
    print("╚══════════════════════════════════════════════╝\n")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} @ {BAUD_RATE} baud.")
    except serial.SerialException as e:
        sys.exit(f"Serial error: {e}\nCheck SERIAL_PORT in the script.")

    # Discard a few startup lines
    for _ in range(5):
        ser.readline()

    results = []   # list of (test_no, target_bpm, expected, detected, duration_s)

    try:
        for i, bpm in enumerate(TEST_BPMS, start=1):
            print(f"\n══════════════════════════════════════════════")
            print(f"  Test {i} of {len(TEST_BPMS)}")
            print(f"══════════════════════════════════════════════")

            detected, duration = run_test(ser, bpm)
            results.append((i, bpm, BREATHS_PER_TEST, detected, duration))

            if i < len(TEST_BPMS):
                print(f"  Rest for {REST_BETWEEN_S}s before next test "
                      f"(breathe normally) …")
                for s in range(REST_BETWEEN_S, 0, -1):
                    print(f"    {s:>2}s remaining …", end="\r", flush=True)
                    time.sleep(1)
                print(" " * 40, end="\r")

    except KeyboardInterrupt:
        print("\n\n  Aborted by user.")

    finally:
        if ser.is_open:
            ser.close()

        # ── Summary table ─────────────────────────────────────────────────
        if results:
            print("\n╔════════════════════════════════════════════════════╗")
            print("║                  RESULTS SUMMARY                   ║")
            print("╠════════╤═════╤═════╤═════╤═════╤═════════╤═════════╣")
            print("║ Test # │ BPM │ Exp │ Det │ Err │  Err %  │  Dur(s) ║")
            print("╟────────┼─────┼─────┼─────┼─────┼─────────┼─────────╢")
            for (n, b, e, d, t) in results:
                err = d - e
                pct = 100.0 * err / e if e else 0.0
                print(f"║   {n:<3}  │ {b:>3} │ {e:>3} │ {d:>3} │"
                      f" {err:>+3} │ {pct:>+6.1f}% │  {t:>5.1f}  ║")
            print("╚════════╧═════╧═════╧═════╧═════╧═════════╧═════════╝\n")

            # ── CSV output ────────────────────────────────────────────────
            with open(OUTPUT_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "test_number",
                    "target_bpm",
                    "expected_breaths",
                    "detected_breaths",
                    "error",
                    "error_percent",
                    "test_duration_s",
                ])
                for (n, b, e, d, t) in results:
                    err = d - e
                    pct = 100.0 * err / e if e else 0.0
                    w.writerow([n, b, e, d, err, f"{pct:.2f}", f"{t:.2f}"])

            print(f"  Results saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()