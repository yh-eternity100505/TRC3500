"""
TRC3500 Project 3 – Breath Rate Monitor
========================================
Sensors  : Conductive Rubber (strain) + TMP61 Thermistor (temperature)
DSP      : Applied to rubber only  – 2nd-order Butterworth bandpass (0.1–1.0 Hz)
           followed by polynomial detrending
Fusion   : Independent BPM estimates from each sensor are combined via a
           confidence-weighted average (50% rubber, 50% thermistor)
Live plot: Single panel – cumulative breath-cycle counter, updates every 0.25 s
MSE      : Accumulated error vs optional metronome ground-truth; exported to
           CSV on exit. MSE histograms are plotted automatically on exit.

Usage
-----
1. Connect STM32 and set SERIAL_PORT below.
2. Run: python listen4.py
3. Optionally enter metronome BPM at the prompt for MSE logging.
   You can also update the ground truth mid-session by pressing G.
4. Press C to stop; CSV of errors and MSE histograms are saved automatically.
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
W_RUBBER = 0.6
W_THERM  = 0.4

# Threshold for breath detection on raw rubber ADC (units)
BREATH_THRESHOLD = 200

# Output CSV for MSE data (saved next to this script)
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "mse_log.csv")

# ─────────────────────────── Keyboard Listener ────────────────────────────────

stop_flag        = threading.Event()
update_gt_flag   = threading.Event()   # signals that user wants to update GT

def _keyboard_listener():
    """
    Background thread:
      C / c  → stop recording and print results
      G / g  → prompt for a new ground truth BPM mid-session
    Works on Windows (msvcrt) and Linux/Mac (tty/termios).
    """
    try:
        import msvcrt                          # Windows
        print("Press C to stop | Press G to update ground truth BPM mid-session.\n")
        while not stop_flag.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch.lower() == "c":
                    stop_flag.set()
                elif ch.lower() == "g":
                    update_gt_flag.set()
            time.sleep(0.05)
    except ImportError:
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        print("Press C to stop | Press G to update ground truth BPM mid-session.\n")
        try:
            tty.setraw(fd)
            while not stop_flag.is_set():
                ch = sys.stdin.read(1)
                if ch.lower() == "c":
                    stop_flag.set()
                elif ch.lower() == "g":
                    update_gt_flag.set()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ─────────────────────────────── DSP ──────────────────────────────────────────

def dsp_rubber(raw: np.ndarray, fs: int) -> np.ndarray:
    """
    Full DSP pipeline applied exclusively to the conductive rubber sensor.

    Steps
    -----
    1. 2nd-order Butterworth bandpass 0.1 – 1.0 Hz
       • 0.1 Hz low-cut removes slow baseline drift
       • 1.0 Hz high-cut rejects motion artefacts & high-freq noise
       • Normal breathing: 12-20 bpm → 0.20-0.33 Hz (well within passband)
    2. Linear detrend to remove any residual slope/offset after filtering
    """
    nyq  = 0.5 * fs
    low  = 0.10 / nyq
    high = 1.00 / nyq
    b, a = butter(2, [low, high], btype="band")
    filtered = filtfilt(b, a, raw)
    return detrend(filtered)


def basic_filter_thermistor(raw: np.ndarray, fs: int) -> np.ndarray:
    """
    Lightweight 1st-order low-pass + detrend for the thermistor signal.
    Used only to derive an independent BPM estimate for data fusion.
    """
    nyq = 0.5 * fs
    b, a = butter(1, 0.80 / nyq, btype="low")
    return detrend(filtfilt(b, a, raw))


def detect_peaks(signal: np.ndarray, fs: int):
    """
    Peak detection with physiology-aware constraints.
      distance  : peaks ≥ 1.5 s apart  →  max detectable rate ≈ 40 bpm
      prominence: ≥ 0.5 × std          →  ignores noise bumps
    """
    peaks, _ = find_peaks(
        signal,
        distance=fs * 1.5,
        prominence=np.std(signal) * 0.5,
    )
    bpm = 0.0
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fs
        bpm = 60.0 / np.mean(intervals)
    return peaks, bpm


# ─────────────────────────── Data Fusion ─────────────────────────────────────

def fuse_bpm(bpm_r: float, bpm_t: float) -> float:
    """
    Confidence-weighted average fusion.
    Falls back to the valid sensor if one returns 0.
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


# ─────────────────────────── MSE Histogram Plot ───────────────────────────────

def plot_mse_histograms(mse_records, ground_truth, output_dir):
    """
    Plot error histograms for rubber, thermistor, and fused BPM estimates.
    Each histogram shows the distribution of (estimated - ground_truth) errors.
    MSE and RMSE are annotated on each panel.
    Saves the figure as mse_histograms.png next to the script.
    """
    if ground_truth is None:
        print("  No ground truth set — skipping MSE histogram plot.")
        return

    # Extract valid errors for each sensor
    err_r = [r[1] - ground_truth for r in mse_records if r[1] > 0]
    err_t = [r[2] - ground_truth for r in mse_records if r[2] > 0]
    err_f = [r[3] - ground_truth if r[3] > 0 else None for r in mse_records]
    err_f = [e for e in err_f if e is not None]

    if not err_f:
        print("  No valid fused BPM data — skipping MSE histogram plot.")
        return

    datasets = [
        (err_r, "Rubber Sensor",   "#64b5f6"),
        (err_t, "Thermistor",      "#ef9a9a"),
        (err_f, "Fused (Combined)","#a5d6a7"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"BPM Error Histograms  (Ground Truth = {ground_truth:.1f} BPM)",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )

    # Shared x range across all panels for fair visual comparison
    all_errors = err_r + err_t + err_f
    x_lim = max(abs(min(all_errors)), abs(max(all_errors))) * 1.3

    for ax, (errors, label, colour) in zip(axes, datasets):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        if len(errors) == 0:
            ax.set_title(f"{label}\n(no data)", color="white")
            continue

        mse  = np.mean(np.array(errors) ** 2)
        rmse = np.sqrt(mse)
        bias = np.mean(errors)

        n_bins = min(20, max(5, len(errors) // 3))
        ax.hist(errors, bins=n_bins, color=colour, edgecolor="#0f1117",
                alpha=0.85, zorder=3)

        # Zero-error reference line
        ax.axvline(0, color="white", linestyle="--", linewidth=1.2,
                   label="Zero error", zorder=4)

        # Mean error (bias) line
        ax.axvline(bias, color="#ffcc02", linestyle="-", linewidth=1.5,
                   label=f"Mean error = {bias:+.2f}", zorder=5)

        ax.set_title(label, color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Error (estimated − ground truth) BPM",
                      color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Count", color="#aaaaaa", fontsize=9)
        ax.set_xlim(-x_lim, x_lim)
        ax.grid(True, color="#2a2d3a", linewidth=0.5, zorder=0)
        ax.legend(fontsize=8, facecolor="#2a2d3a", labelcolor="white",
                  framealpha=0.8)

        # Annotate MSE / RMSE in corner
        ax.text(0.97, 0.97,
                f"MSE  = {mse:.3f}\nRMSE = {rmse:.3f} BPM\nn = {len(errors)}",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2d3a", alpha=0.9))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "mse_histograms.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  MSE histograms saved → {save_path}")
    plt.show()


# ─────────────────────────── Plot Initialisation ─────────────────────────────

def init_figure(ground_truth):
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

    status_txt = fig.text(
        0.5, 0.01,
        "Waiting for full buffer …",
        ha="center", fontsize=11, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#37474f", alpha=0.9),
    )

    fig.suptitle(
        "TRC3500 Project 3 – Breath Rate Monitor",
        color="white", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    artists = dict(ln_cyc=ln_cyc, sc_cyc=sc_cyc, status_txt=status_txt)
    return fig, ax_cyc, artists


# ─────────────────────────────────── Main ─────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  TRC3500 Project 3 – Breath Rate Monitor     ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print("Enter metronome BPM for MSE logging (press Enter to skip): ", end="", flush=True)
    gt_raw = input().strip()

    # ground_truth is stored in a mutable list so the keyboard thread can
    # update it mid-session when the user presses G
    ground_truth = [None]
    if gt_raw:
        try:
            ground_truth[0] = float(gt_raw)
            print(f"  → Ground truth set to {ground_truth[0]:.1f} bpm")
        except ValueError:
            print("  → Invalid – skipping ground truth")

    active_threshold = BREATH_THRESHOLD
    print(f"  → Breath detection threshold: {active_threshold} ADC counts\n")

    # ── Data buffers ───────────────────────────────────────────────────────
    times      = deque(maxlen=BUFFER_SIZE)
    rubber_buf = deque(maxlen=BUFFER_SIZE)
    therm_buf  = deque(maxlen=BUFFER_SIZE)

    last_peak_time   = -999.0
    total_breaths    = 0
    cycle_times      = []
    cycle_counts     = []

    extreme_val      = None
    breath_direction = None
    half_cycles      = 0

    mse_records = []   # list of (time, bpm_rubber, bpm_therm, bpm_fused)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax_cyc, art = init_figure(ground_truth[0])

    # ── Serial ─────────────────────────────────────────────────────────────
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} @ {BAUD_RATE} baud.")
        print("Collecting data … (C to stop | G to update ground truth)\n")
    except serial.SerialException as e:
        sys.exit(f"Serial error: {e}\nCheck SERIAL_PORT in the script.")

    for _ in range(5):
        ser.readline()

    kb_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    kb_thread.start()

    start_time      = time.time()
    last_print_time = 0.0
    now             = 0.0

    try:
        while not stop_flag.is_set():

            # ── Mid-session ground truth update (G key) ───────────────────
            if update_gt_flag.is_set():
                update_gt_flag.clear()
                print("\n  Enter new ground truth BPM: ", end="", flush=True)
                try:
                    new_gt = float(input().strip())
                    ground_truth[0] = new_gt
                    print(f"  → Ground truth updated to {new_gt:.1f} bpm\n")
                except ValueError:
                    print("  → Invalid input, ground truth unchanged.\n")

            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue

            try:
                parts = raw_line.split(",")
                if len(parts) != 2:
                    continue
                therm_val  = float(parts[1])
                rubber_val = float(parts[0])
            except ValueError:
                continue

            now = time.time() - start_time
            times.append(now)
            therm_buf.append(therm_val)
            rubber_buf.append(rubber_val)

            # ── Threshold breath detection ────────────────────────────────
            if extreme_val is None:
                extreme_val = rubber_val

            if breath_direction == "down":
                if rubber_val < extreme_val:
                    extreme_val = rubber_val
                if rubber_val - extreme_val >= active_threshold:
                    half_cycles += 1
                    breath_direction = "up"
                    extreme_val = rubber_val
                    if half_cycles % 2 == 0:
                        if now - last_peak_time >= 3.0:
                            total_breaths += 1
                            last_peak_time = now
                            cycle_times.append(now)
                            cycle_counts.append(total_breaths)
            else:
                if rubber_val > extreme_val:
                    extreme_val = rubber_val
                if extreme_val - rubber_val >= active_threshold:
                    half_cycles += 1
                    breath_direction = "down"
                    extreme_val = rubber_val
                    if half_cycles % 2 == 0:
                        if now - last_peak_time >= 3.0:
                            total_breaths += 1
                            last_peak_time = now
                            cycle_times.append(now)
                            cycle_counts.append(total_breaths)

            # ── Wait for full buffer ──────────────────────────────────────
            if len(rubber_buf) < BUFFER_SIZE:
                if now - last_print_time >= 1.0:
                    pct = len(rubber_buf) / BUFFER_SIZE * 100
                    print(f"[{now:5.1f}s] Buffering … {pct:.0f}% "
                          f"({len(rubber_buf)}/{BUFFER_SIZE} samples) | "
                          f"Raw rubber: {rubber_val:6.0f} | Raw therm: {therm_val:6.0f}")
                    last_print_time = now
                continue

            raw_r = np.array(rubber_buf)
            raw_t = np.array(therm_buf)

            proc_r = dsp_rubber(raw_r, SAMPLE_RATE)
            peaks_r, bpm_r = detect_peaks(proc_r, SAMPLE_RATE)

            proc_t = basic_filter_thermistor(raw_t, SAMPLE_RATE)
            _, bpm_t = detect_peaks(proc_t, SAMPLE_RATE)

            bpm_f = fuse_bpm(bpm_r, bpm_t)

            mse_records.append((now, bpm_r, bpm_t, bpm_f))

            # ── Update live plot ──────────────────────────────────────────
            if cycle_times:
                art["ln_cyc"].set_data(cycle_times, cycle_counts)
                art["sc_cyc"].set_offsets(np.c_[cycle_times, cycle_counts])
                ax_cyc.set_xlim(0, max(cycle_times) + 5)
                ax_cyc.set_ylim(0, total_breaths + 3)

            gt_str = f"{ground_truth[0]:.1f}" if ground_truth[0] else "not set"
            status = (
                f"Fused: {bpm_f:.1f} bpm  |  "
                f"Rubber: {bpm_r:.1f}  |  Thermistor: {bpm_t:.1f}  |  "
                f"Total breaths: {total_breaths}  |  GT: {gt_str} bpm"
            )
            if ground_truth[0] and bpm_f > 0:
                status += f"  |  Error: {bpm_f - ground_truth[0]:+.1f} bpm"
            art["status_txt"].set_text(status)

            fig.canvas.draw()
            fig.canvas.flush_events()

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
                if ground_truth[0] and bpm_f > 0:
                    print(f" | Δ={bpm_f - ground_truth[0]:+.1f} bpm", end="")
                print()
                last_print_time = now

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

        # ── Results summary ───────────────────────────────────────────────
        print("\n" + "═" * 54)
        print("  RESULTS SUMMARY")
        print("═" * 54)
        print(f"  Total breaths counted : {total_breaths}")
        if mse_records:
            print(f"  Session duration      : {now:.1f} s")

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

            if ground_truth[0] and bpm_f_vals:
                errors   = [bf - ground_truth[0] for bf in bpm_f_vals]
                mse_val  = np.mean([e**2 for e in errors])
                rmse_val = np.sqrt(mse_val)
                print(f"\n  Ground truth          : {ground_truth[0]:.1f} bpm")
                print(f"  Mean error (fused)    : {np.mean(errors):+.2f} bpm")
                print(f"  MSE  (fused)          : {mse_val:.3f}")
                print(f"  RMSE (fused)          : {rmse_val:.3f} bpm")

        print("═" * 54 + "\n")

        # ── Save CSV ──────────────────────────────────────────────────────
        if mse_records:
            with open(OUTPUT_CSV, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "bpm_rubber", "bpm_thermistor", "bpm_fused",
                            "error_rubber", "error_thermistor", "error_fused",
                            "sq_error_rubber", "sq_error_thermistor", "sq_error_fused"])
                for (t, br, bt, bf) in mse_records:
                    if ground_truth[0] and bf > 0:
                        er = br - ground_truth[0] if br > 0 else float("nan")
                        et = bt - ground_truth[0] if bt > 0 else float("nan")
                        ef = bf - ground_truth[0]
                        w.writerow([f"{t:.3f}", f"{br:.2f}", f"{bt:.2f}", f"{bf:.2f}",
                                    f"{er:.3f}", f"{et:.3f}", f"{ef:.3f}",
                                    f"{er**2:.3f}" if br > 0 else "nan",
                                    f"{et**2:.3f}" if bt > 0 else "nan",
                                    f"{ef**2:.3f}"])
                    else:
                        w.writerow([f"{t:.3f}", f"{br:.2f}", f"{bt:.2f}", f"{bf:.2f}",
                                    *["na"] * 6])
            print(f"  MSE data saved → {OUTPUT_CSV}")

        # ── Plot MSE histograms ───────────────────────────────────────────
        if mse_records and ground_truth[0]:
            print("\n  Plotting MSE histograms...")
            plot_mse_histograms(mse_records, ground_truth[0],
                                os.path.dirname(OUTPUT_CSV))
        else:
            print("\n  No ground truth set — MSE histograms skipped.")
            print("  Tip: next time press G during recording to set ground truth.")

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()