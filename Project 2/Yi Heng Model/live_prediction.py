import serial
import time
import numpy as np
import scipy.fft as fft
import joblib
import sys

# --- Configuration ---
SERIAL_PORT = 'COM6'  # Update to match your STM32's port
BAUD_RATE = 115200
MODEL_FILE = 'pindrop_rf_model.pkl'

# Hardware Settings (Must match your data collection exactly)
FS = 2000
SAMPLES_PER_DROP = 10000
BASELINE = 2048
THRESHOLD_OFFSET = 100

# Human-readable labels for the In-Lab Demo
CLASS_NAMES = {
    0: "10cm Distance / 10cm Height (Class 0)",
    1: "10cm Distance / 30cm Height (Class 1)",
    2: "30cm Distance / 10cm Height (Class 2)",
    3: "30cm Distance / 30cm Height (Class 3)"
}

def extract_live_features(raw_samples):
    """Applies the exact same math used in the training script."""
    # Convert to numpy array and clip to 12-bit ADC limits (0-4095)
    samples = np.clip(np.array(raw_samples), 0, 4095)
    
    # Feature 1: Peak-to-Peak Amplitude
    ptp_amp = np.max(samples) - np.min(samples)
    
    # Feature 2: Signal Energy
    mean_val = np.mean(samples)
    centered_samples = samples - mean_val
    energy = np.sum(centered_samples ** 2)
    
    # Feature 3: Peak Frequency
    yf = np.abs(fft.rfft(centered_samples))
    xf = fft.rfftfreq(len(centered_samples), 1/FS)
    peak_freq = xf[np.argmax(yf)]
    
    return [ptp_amp, energy, peak_freq]

def main():
    # 1. Load the trained Machine Learning Model
    print(f"Loading model '{MODEL_FILE}'...")
    try:
        clf = joblib.load(MODEL_FILE)
        print("Model loaded successfully!\n")
    except FileNotFoundError:
        print(f"ERROR: Could not find {MODEL_FILE}.")
        print("Please run the training script first to generate the model.")
        sys.exit(1)

    # 2. Connect to STM32
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=1)
        ser.reset_input_buffer()
        time.sleep(1)
        print("Connected! Ready for the live demo.")
        print("-" * 50)
    except serial.SerialException as e:
        print(f"Failed to connect to {SERIAL_PORT}: {e}")
        sys.exit(1)

    is_recording = False
    adc_buffer = []

    print("\nWaiting for a coin drop... (Press Ctrl+C to exit)\n")

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Filter to only grab clean 4-digit numbers as discussed previously
                raw_value = "".join(filter(str.isdigit, line))
                
                if len(raw_value) == 4:
                    value = int(raw_value)

                    if len(adc_buffer) % 500 == 0:
                        print(f"[{len(adc_buffer)}/{SAMPLES_PER_DROP}] Samples collected...")

                    # Trigger Logic: Wait for a spike above/below the baseline
                    if not is_recording:
                        if abs(value - BASELINE) > THRESHOLD_OFFSET:
                            is_recording = True
                            adc_buffer = [value]
                            print("Drop Detected! Capturing 10,000 samples...", end="", flush=True)
                    
                    # Recording Logic
                    else:
                        adc_buffer.append(value)
                        
                        # Once we have a full window of data, process it!
                        if len(adc_buffer) >= SAMPLES_PER_DROP:
                            print(" Done.")
                            
                            # Extract the 3 features
                            features = extract_live_features(adc_buffer)
                            
                            # Ask the model to predict based on those features
                            # We wrap `features` in a list because sklearn expects a 2D array
                            prediction = clf.predict([features])[0]
                            
                            # Print the results nicely for the demonstrators
                            print("=" * 50)
                            print(f" PREDICTION: {CLASS_NAMES[prediction]}")
                            print("-" * 50)
                            print(f" Features -> PtP: {features[0]:.1f} | Energy: {features[1]:.1e} | Freq: {features[2]:.1f} Hz")
                            print("=" * 50 + "\n")
                            
                            # Reset variables to wait for the next drop
                            is_recording = False
                            adc_buffer = []
                            ser.reset_input_buffer() # Clear old buffer data so it doesn't instantly re-trigger
                            print("Waiting for the next coin drop...\n")

                            break

    except KeyboardInterrupt:
        print("\nDemo session ended by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()