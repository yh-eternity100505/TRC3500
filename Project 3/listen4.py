"""
TRC3500 Project 3 – Breath Rate Monitor
========================================
Sensors  : Conductive Rubber (strain) + TMP61 Thermistor (temperature)
DSP      : Applied to rubber only  – 2nd-order Butterworth bandpass (0.1–1.0 Hz)
           followed by polynomial detrending
Fusion   : Independent BPM estimates from each sensor are combined via a
           confidence-weighted average (rubber 60 %, thermistor 40 %)
Live plot: Single panel – cumulative breath-cycle counter, updates every 0.25 s
MSE      : Accumulated error vs optional metronome ground-truth; exported to
           CSV on exit for report histograms

Usage
-----
1. Connect STM32 and set SERIAL_PORT below.
2. Run: python listen4.py
3. Optionally enter metronome BPM at the prompt for MSE logging.
4. Press Ctrl-C to stop; CSV of errors is saved automatically.
"""

import csv
import os
import sys
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import serial
from scipy.signal import butter, filtfilt, detrend, find_peaks

# ─────────────────────────────── Configuration ────────────────────────────────
SERIAL_PORT     = "COM6"        # Windows: "COM3"; Linux/Mac: "/dev/ttyACM0"
BAUD_RATE       = 115200
SAMPLE_RATE     = 100           # Hz – must match STM32 TIM2 period
WINDOW_SIZE_SEC = 15            # Sliding analysis window (seconds)
SLIDE_SEC       = 0.25          # Window slides by this much each update
BUFFER_SIZE     = SAMPLE_RATE * WINDOW_SIZE_SEC
SLIDE_SAMPLES   = int(SAMPLE_RATE * SLIDE_SEC)

# Fusion weights (must sum to 1.0)
W_RUBBER = 0.5
W_THERM  = 0.5

# Threshold for breath detection on raw rubber ADC (units)
# A swing of ≥200 ADC counts in either direction = inhale or exhale
BREATH_THRESHOLD = 200

# Output CSV for MSE data (saved next to this script)
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "mse_log.csv")

# ─────────────────────────── Keyboard Listener ────────────────────────────────

stop_flag = threading.Event()

def _keyboard_listener():
    """
    Background thread: sets stop_flag when the user presses C (or c).
    Works on Windows (msvcrt) and Linux/Mac (tty/termios).
    """
    try:
        import msvcrt                          # Windows
        print("Press C to stop and print results.\n")
        while not stop_flag.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch.lower() == "c":
                    stop_flag.set()
            time.sleep(0.05)
    except ImportError:
        import tty, termios                    
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        print("Press C to stop and print results.\n")
        try:
            tty.setraw(fd)
            while not stop_flag.is_set():
                ch = sys.stdin.read(1)
                if ch.lower() == "c":
                    stop_flag.set()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)



def dsp_rubber(raw: np.ndarray, fs: int) -> np.ndarray:
    """
    Full DSP pipeline applied exclusively to the conductive rubber sensor.

    Steps
    -----
    1. 2nd-order Butterworth bandpass 0.1 – 1.0 Hz
       • 0.1 Hz low-cut removes slow baseline drift without detrend alone
       • 1.0 Hz high-cut rejects motion artefacts & high-freq noise
       • Normal breathing: 12-20 bpm → 0.20-0.33 Hz (well within passband)
    2. Linear detrend to remove any residual slope/offset after filtering
    """
    nyq  = 0.5 * fs
    low  = 0.10 / nyq
    high = 1.00 / nyq
    b, a = butter(2, [low, high], btype="band")
    filtered = filtfilt(b, a, raw)       # zero-phase, no lag
    return detrend(filtered)


def basic_filter_thermistor(raw: np.ndarray, fs: int) -> np.ndarray:
    """
    Lightweight 1st-order low-pass + detrend for the thermistor signal.
    This is NOT the main DSP stage; it is used only to derive an
    independent BPM estimate for the data-fusion step.
    """
    nyq = 0.5 * fs
    b, a = butter(1, 0.80 / nyq, btype="low")
    return detrend(filtfilt(b, a, raw))


def detect_peaks(signal: np.ndarray, fs: int):
    """
    Peak detection with physiology-aware constraints.

    distance : peaks ≥ 1.5 s apart  →  max detectable rate ≈ 40 bpm
    prominence: ≥ 0.5 × std         →  ignores noise bumps
    """
    peaks, _ = find_peaks(
        signal,
        distance=fs * 1.5,
        prominence=np.std(signal) * 0.5,
    )
    bpm = 0.0
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fs        # seconds between peaks
        bpm = 60.0 / np.mean(intervals)
    return peaks, bpm


# ─────────────────────────── Data Fusion ─────────────────────────────────────

def fuse_bpm(bpm_r: float, bpm_t: float) -> float:
    """
    Confidence-weighted average fusion.
    If one sensor fails to detect (BPM = 0), the other's value is used alone.
    """
    valid_r = bpm_r > 0
    valid_t = bpm_t > 0
    if valid_r and valid_t:
        return W_RUBBER * bpm_r + W_THERM * bpm_t
    elif valid_r:
        return bpm_r
    elif valid_t:
        return bpm_t
    return 0.0


# ─────────────────────────── Plot Initialisation ─────────────────────────────

def init_figure(ground_truth: float | None):
    """Create a single-panel interactive figure: cumulative breath cycle counter."""
    plt.ion()
    fig, ax_cyc = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    ax_cyc.set_facecolor("#2a2a3e")
    ax_cyc.tick_params(colors="white")
    for spine in ax_cyc.spines.values():
        spine.set_edgecolor("#555")

    ln_cyc, = ax_cyc.plot([], [], color="#80cbc4", linewidth=2)
    sc_cyc  = ax_cyc.scatter([], [], color="#a5d6a7", zorder=5, s=60,
                              label="Breath event")
    ax_cyc.set_title("Cumulative Breath Cycles (Live)", color="white", fontsize=12)
    ax_cyc.set_xlabel("Time (s)", color="white", fontsize=10)
    ax_cyc.set_ylabel("Total breaths counted", color="white", fontsize=10)
    ax_cyc.legend(fontsize=9, facecolor="#333", labelcolor="white")

    # ── Status bar ─────────────────────────────────────────────────────────
    status_txt = fig.text(
        0.5, 0.01,
        "Waiting for full buffer …",
        ha="center", fontsize=11, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#37474f", alpha=0.9),
    )

    fig.suptitle(
        "TRC3500 Project 3 – Breath Rate Monitor  ", 
        color="white", fontsize=11,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    artists = dict(ln_cyc=ln_cyc, sc_cyc=sc_cyc, status_txt=status_txt)
    return fig, ax_cyc, artists


# ─────────────────────────────────── Main ─────────────────────────────────────

def main():
    # ── Ground truth prompt ────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════╗")
    print("║  TRC3500 Project 3 – Breath Rate Monitor     ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print("Enter metronome BPM for MSE logging (press Enter to skip): ", end="", flush=True)
    gt_raw = input().strip()
    ground_truth: float | None = None
    if gt_raw:
        try:
            ground_truth = float(gt_raw)
            print(f"  → Ground truth set to {ground_truth:.1f} bpm")
        except ValueError:
            print("  → Invalid – skipping ground truth")

    # Use a lower threshold when no metronome is set (free breathing tends
    # to produce smaller ADC swings than paced breathing)
    active_threshold = BREATH_THRESHOLD if ground_truth else 200
    print(f"  → Breath detection threshold: {active_threshold} ADC counts")

    print()

    # ── Data buffers ───────────────────────────────────────────────────────
    times       = deque(maxlen=BUFFER_SIZE)
    rubber_buf  = deque(maxlen=BUFFER_SIZE)
    therm_buf   = deque(maxlen=BUFFER_SIZE)

    # Breath cycle tracking
    last_peak_time   = -999.0
    total_breaths    = 0
    cycle_times: list[float] = []
    cycle_counts: list[int]  = []

    # Threshold-based breath detector state (runs per raw sample)
    # Uses a running-extreme algorithm:
    #   - During upswing: track the running peak; trigger exhale when value
    #     falls ≥ threshold BELOW that peak
    #   - During downswing: track the running trough; trigger inhale when value
    #     rises ≥ threshold ABOVE that trough
    extreme_val      = None   # running peak (upswing) or trough (downswing)
    breath_direction = None   # 'up' (inhaling) or 'down' (exhaling)
    half_cycles      = 0      # inhale + exhale = 2 half-cycles = 1 full breath

    # MSE accumulation
    mse_records: list[tuple[float, float, float, float]] = []
    # columns: time, bpm_rubber, bpm_therm, bpm_fused

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax_cyc, art = init_figure(ground_truth)

    # ── Serial ─────────────────────────────────────────────────────────────
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} @ {BAUD_RATE} baud.")
        print("Collecting data … (Ctrl-C to stop)\n")
    except serial.SerialException as e:
        sys.exit(f"Serial error: {e}\nCheck SERIAL_PORT in the script.")

    # Flush any partial lines
    for _ in range(5):
        ser.readline()

    # Start keyboard listener
    t = threading.Thread(target=_keyboard_listener, daemon=True)
    t.start()

    start_time      = time.time()
    last_print_time = 0.0       # tracks when we last printed the per-second line
    now             = 0.0       # safe default if loop exits before first sample

    try:
        while not stop_flag.is_set():
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue

            try:
                parts = raw_line.split(",")
                if len(parts) != 2:
                    continue
                therm_val  = float(parts[0])
                rubber_val = float(parts[1])
            except ValueError:
                continue

            now = time.time() - start_time
            times.append(now)
            therm_buf.append(therm_val)
            rubber_buf.append(rubber_val)

            # ── Threshold breath detection (runs on every raw sample) ─────
            # Initialise on very first sample
            if extreme_val is None:
                extreme_val = rubber_val

            if breath_direction == "down":
                # Downswing: keep tracking the running trough
                if rubber_val < extreme_val:
                    extreme_val = rubber_val
                # Inhale detected when value rises ≥ threshold above the trough
                if rubber_val - extreme_val >= active_threshold:
                    half_cycles += 1
                    breath_direction = "up"
                    extreme_val = rubber_val        # start tracking new peak
                    if half_cycles % 2 == 0:        # every 2 half-cycles = 1 breath
                        if now - last_peak_time >= 3.0:
                            total_breaths += 1
                            last_peak_time = now
                            cycle_times.append(now)
                            cycle_counts.append(total_breaths)

            else:
                # Upswing (or initial): keep tracking the running peak
                if rubber_val > extreme_val:
                    extreme_val = rubber_val
                # Exhale detected when value falls ≥ threshold below the peak
                if extreme_val - rubber_val >= active_threshold:
                    half_cycles += 1
                    breath_direction = "down"
                    extreme_val = rubber_val        # start tracking new trough
                    if half_cycles % 2 == 0:
                        if now - last_peak_time >= 3.0:
                            total_breaths += 1
                            last_peak_time = now
                            cycle_times.append(now)
                            cycle_counts.append(total_breaths)

            # ── Only process once the window is full ──────────────────────
            if len(rubber_buf) < BUFFER_SIZE:
                if now - last_print_time >= 1.0:
                    pct = len(rubber_buf) / BUFFER_SIZE * 100
                    print(f"[{now:5.1f}s] Buffering … {pct:.0f}% "
                          f"({len(rubber_buf)}/{BUFFER_SIZE} samples) | "
                          f"Raw rubber: {rubber_val:6.0f} | Raw therm: {therm_val:6.0f}")
                    last_print_time = now
                continue

            t_arr = np.array(times)
            raw_r = np.array(rubber_buf)
            raw_t = np.array(therm_buf)

            # ── DSP: rubber only ──────────────────────────────────────────
            proc_r = dsp_rubber(raw_r, SAMPLE_RATE)
            peaks_r, bpm_r = detect_peaks(proc_r, SAMPLE_RATE)

            # ── Basic filter: thermistor (for fusion only) ─────────────────
            proc_t = basic_filter_thermistor(raw_t, SAMPLE_RATE)
            _, bpm_t = detect_peaks(proc_t, SAMPLE_RATE)

            # ── Data fusion ───────────────────────────────────────────────
            bpm_f = fuse_bpm(bpm_r, bpm_t)

            # ── Breath cycle accumulation (now handled per-sample above) ──

            # ── MSE logging ───────────────────────────────────────────────
            mse_records.append((now, bpm_r, bpm_t, bpm_f))
            if ground_truth and bpm_f > 0:
                err = bpm_f - ground_truth
                mse_sq = err ** 2

            # ── Update plot ───────────────────────────────────────────────
            # Breathing cycle live graph
            if cycle_times:
                art["ln_cyc"].set_data(cycle_times, cycle_counts)
                art["sc_cyc"].set_offsets(np.c_[cycle_times, cycle_counts])
                ax_cyc.set_xlim(0, max(cycle_times) + 5)
                ax_cyc.set_ylim(0, total_breaths + 3)

            # Status bar
            status = (
                f"Fused: {bpm_f:.1f} bpm  |  "
                f"Rubber: {bpm_r:.1f}  |  Thermistor: {bpm_t:.1f}  |  "
                f"Total breaths: {total_breaths}"
            )
            if ground_truth and bpm_f > 0:
                status += f"  |  Error vs GT: {bpm_f - ground_truth:+.1f} bpm"
            art["status_txt"].set_text(status)

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Console log — print once per second
            if now - last_print_time >= 1.0:
                print(
                    f"[{now:6.1f}s] "
                    f"Rubber: {bpm_r:5.1f} bpm | "
                    f"Therm: {bpm_t:5.1f} bpm | "
                    f"Fused: {bpm_f:5.1f} bpm | "
                    f"Breaths: {total_breaths:4d} | "
                    f"Raw rubber: {rubber_buf[-1]:6.0f} | "
                    f"Raw therm: {therm_buf[-1]:6.0f}",
                    end="",
                )
                if ground_truth and bpm_f > 0:
                    print(f" | Δ={bpm_f - ground_truth:+.1f} bpm", end="")
                print()
                last_print_time = now

            # ── Slide window ──────────────────────────────────────────────
            for _ in range(SLIDE_SAMPLES):
                times.popleft()
                rubber_buf.popleft()
                therm_buf.popleft()

    except Exception as e:
        print(f"\nUnexpected error: {e}")

    finally:
        stop_flag.set()
        if "ser" in dir() and ser.is_open:
            ser.close()

        # ── Print processed data summary ─────────────────────────────────
        print("\n" + "═" * 54)
        print("  RESULTS SUMMARY")
        print("═" * 54)
        print(f"  Total breaths counted : {total_breaths}")
        print(f"  Session duration      : {now:.1f} s" if mse_records else "  No data collected.")

        if mse_records:
            bpm_r_vals = [r[1] for r in mse_records if r[1] > 0]
            bpm_t_vals = [r[2] for r in mse_records if r[2] > 0]
            bpm_f_vals = [r[3] for r in mse_records if r[3] > 0]

            if bpm_r_vals:
                print(f"  Rubber BPM   — mean: {np.mean(bpm_r_vals):.1f}  "
                      f"std: {np.std(bpm_r_vals):.2f}  "
                      f"min: {np.min(bpm_r_vals):.1f}  "
                      f"max: {np.max(bpm_r_vals):.1f}")
            if bpm_t_vals:
                print(f"  Therm BPM    — mean: {np.mean(bpm_t_vals):.1f}  "
                      f"std: {np.std(bpm_t_vals):.2f}  "
                      f"min: {np.min(bpm_t_vals):.1f}  "
                      f"max: {np.max(bpm_t_vals):.1f}")
            if bpm_f_vals:
                print(f"  Fused BPM    — mean: {np.mean(bpm_f_vals):.1f}  "
                      f"std: {np.std(bpm_f_vals):.2f}  "
                      f"min: {np.min(bpm_f_vals):.1f}  "
                      f"max: {np.max(bpm_f_vals):.1f}")

            if ground_truth and bpm_f_vals:
                errors   = [bf - ground_truth for bf in bpm_f_vals]
                mse_val  = np.mean([e**2 for e in errors])
                rmse_val = np.sqrt(mse_val)
                print(f"\n  Ground truth          : {ground_truth:.1f} bpm")
                print(f"  Mean error (fused)    : {np.mean(errors):+.2f} bpm")
                print(f"  MSE  (fused)          : {mse_val:.3f}")
                print(f"  RMSE (fused)          : {rmse_val:.3f} bpm")

        print("═" * 54 + "\n")

        # ── Save MSE CSV ────────────────────────────────────────────────
        if mse_records:
            with open(OUTPUT_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "bpm_rubber", "bpm_thermistor", "bpm_fused",
                            "error_rubber", "error_thermistor", "error_fused",
                            "sq_error_rubber", "sq_error_thermistor", "sq_error_fused"])
                for (t, br, bt, bf) in mse_records:
                    if ground_truth and bf > 0:
                        er = br - ground_truth if br > 0 else float("nan")
                        et = bt - ground_truth if bt > 0 else float("nan")
                        ef = bf - ground_truth
                        w.writerow([f"{t:.3f}", f"{br:.2f}", f"{bt:.2f}", f"{bf:.2f}",
                                    f"{er:.3f}", f"{et:.3f}", f"{ef:.3f}",
                                    f"{er**2:.3f}" if br > 0 else "nan",
                                    f"{et**2:.3f}" if bt > 0 else "nan",
                                    f"{ef**2:.3f}"])
                    else:
                        w.writerow([f"{t:.3f}", f"{br:.2f}", f"{bt:.2f}", f"{bf:.2f}",
                                    *["na"]*6])

            print(f"MSE data saved → {OUTPUT_CSV}")

            # Quick MSE summary
            if ground_truth:
                valid = [(bf - ground_truth)**2
                         for (_, _, _, bf) in mse_records if bf > 0]
                if valid:
                    print(f"MSE (fused, vs {ground_truth:.1f} bpm): {np.mean(valid):.3f}")
                    print(f"RMSE: {np.sqrt(np.mean(valid)):.3f} bpm")

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()