import serial
import time
import numpy as np
import joblib

MODEL_FILE = "coin_drop_model_features.pkl"

PORT = "COM6"
BAUDRATE = 115200
NUM_SAMPLES = 10000
TRIGGER_THRESHOLD = 50
FS = 2000   # change this to your actual sampling rate


def decode_label(label):
    label_map = {
        0: ("10 cm", "10 cm"),
        1: ("10 cm", "30 cm"),
        2: ("30 cm", "10 cm"),
        3: ("30 cm", "30 cm"),
    }
    return label_map.get(int(label), ("Unknown", "Unknown"))


def extract_features(signal, fs=FS):
    signal = np.array(signal, dtype=float)

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    ptp_val = np.ptp(signal)
    rms_val = np.sqrt(np.mean(signal ** 2))
    energy_val = np.sum(signal ** 2)

    peak_index = np.argmax(np.abs(signal - mean_val))
    peak_value = signal[peak_index]

    centered = signal - mean_val
    fft_vals = np.fft.rfft(centered)
    fft_freqs = np.fft.rfftfreq(len(centered), d=1 / fs)
    fft_magnitude = np.abs(fft_vals)

    if len(fft_magnitude) > 1:
        dominant_freq = fft_freqs[np.argmax(fft_magnitude[1:]) + 1]
        spectral_energy = np.sum(fft_magnitude ** 2)
    else:
        dominant_freq = 0.0
        spectral_energy = 0.0

    return np.array([
        mean_val,
        std_val,
        max_val,
        min_val,
        ptp_val,
        rms_val,
        energy_val,
        peak_index,
        peak_value,
        dominant_freq,
        spectral_energy
    ], dtype=float)


def read_one_event(ser, num_samples=NUM_SAMPLES, trigger_threshold=TRIGGER_THRESHOLD):
    print("Estimating baseline... do not drop yet.")

    baseline_values = []
    while len(baseline_values) < 50:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        try:
            value = int(line)
            baseline_values.append(value)
        except ValueError:
            continue

    baseline = np.mean(baseline_values)
    print(f"Baseline: {baseline:.2f}")
    print("Now drop the coin.")

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        try:
            value = int(line)
        except ValueError:
            continue

        if abs(value - baseline) > trigger_threshold:
            samples = [value]
            break

    while len(samples) < num_samples:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        try:
            value = int(line)
            samples.append(value)
        except ValueError:
            continue

    return np.array(samples, dtype=float)


def main():
    model = joblib.load(MODEL_FILE)

    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

    try:
        signal = read_one_event(ser)

        features = extract_features(signal)
        X_live = features.reshape(1, -1)

        print("Live feature length:", len(features))
        print("Model expects:", model.n_features_in_)

        pred = model.predict(X_live)[0]
        distance, height = decode_label(pred)

        print(f"Final result: distance = {distance}, height = {height}")

    finally:
        ser.close()


if __name__ == "__main__":
    main()