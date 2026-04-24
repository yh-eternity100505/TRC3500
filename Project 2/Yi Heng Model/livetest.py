import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import tensorflow as tf
import os

# --- Configuration ---
MODEL_FILENAME = "pindrop_cnn_model.h5"
# Make sure this port matches what works on your Mac/PC (e.g., '/dev/cu.usbserial-XXXX' or 'COM6')
PORT = 'COM6' 
BAUD = 115200
SAMPLES_PER_BATCH = 10000
FS = 2000

CLASS_NAMES = {
    0: "10cm Height / 10cm Distance",
    1: "10cm Height / 30cm Distance",
    2: "30cm Height / 10cm Distance",
    3: "30cm Height / 30cm Distance"
}

# 1. Load the Trained Model
print(f"Loading CNN model from {MODEL_FILENAME}...")
try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    os._exit(1)

# 2. Setup Serial Connection
print(f"Connecting to {PORT}...")
try:
    ser = serial.Serial(port=PORT, baudrate=BAUD, timeout=1)
    ser.reset_input_buffer()
    time.sleep(1)
    print("Serial connected!")
except serial.SerialException as e:
    print(f"Failed to connect to {PORT}: {e}")
    os._exit(1)

adc_values = []
inference_count = 0

print("\n--- Live Inference Started ---")
print("Waiting for pin drop...")

try:
    # Turn on interactive mode for live plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            try:
                # Extract numeric values
                raw_value = "".join(filter(str.isdigit, line))
                
                # Strict length check to ensure data integrity
                if len(raw_value) == 4: 
                    val = float(raw_value)
                    adc_values.append(val)
                    
                    # Print progress so you know it's reading
                    if len(adc_values) % 500 == 0:
                        print(f"[{len(adc_values)}/{SAMPLES_PER_BATCH}] Samples collected...")
                        
                    # When a full batch is collected
                    if len(adc_values) >= SAMPLES_PER_BATCH:
                        
                        # Grab EXACTLY the first 10,000 samples
                        data = np.array(adc_values[:SAMPLES_PER_BATCH])
                        
                        # --- 1. PREPROCESSING ---
                        f, t, Sxx = spectrogram(data, fs=FS, nperseg=256, noverlap=128)
                        Sxx_log = np.log(Sxx + 1e-10)
                        
                        # Reshape for CNN (Batch, Height, Width, Channels)
                        X_input = np.expand_dims(Sxx_log, axis=0)
                        X_input = np.expand_dims(X_input, axis=-1)
                        
                        # --- 2. INFERENCE ---
                        predictions = model.predict(X_input, verbose=0)
                        predicted_class = np.argmax(predictions[0])
                        confidence = np.max(predictions[0]) * 100
                        
                        inference_count += 1
                        predicted_label = CLASS_NAMES.get(predicted_class, "Unknown")
                        
                        print("\n" + "="*50)
                        print(f"INFERENCE #{inference_count}")
                        print(f"PREDICTED: {predicted_label} (Class {predicted_class})")
                        print(f"CONFIDENCE: {confidence:.2f}%")
                        print("="*50 + "\n")
                        
                        # --- 3. VISUALIZATION ---
                        ax1.clear()
                        ax1.plot(data, color='tab:blue')
                        ax1.set_title(f"Live Signal | Prediction: {predicted_label} ({confidence:.1f}%)")
                        ax1.set_xlabel("Sample Index")
                        ax1.set_ylabel("ADC Value")
                        
                        ax2.clear()
                        ax2.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
                        ax2.set_title("2D Spectrogram (CNN Input)")
                        ax2.set_xlabel("Time (sec)")
                        ax2.set_ylabel("Frequency (Hz)")
                        
                        plt.tight_layout()
                        plt.pause(0.05)
                        
                        # --- 4. RESET FOR NEXT INFERENCE ---
                        print("Waiting for next pin drop...")
                        
                        # Clear the array so we start counting from 0 again
                        adc_values = [] 
                        
                        # Flush the serial buffer to drop any noise that arrived 
                        # during the split-second the CNN and graphs were updating
                        ser.reset_input_buffer() 
                        
            except ValueError:
                pass # Ignore parsing errors of non-numeric lines

except KeyboardInterrupt:
    print(f"\nManual exit triggered. Total inferences: {inference_count}")
finally:
    # Safely close everything down
    if 'ser' in locals() and ser.is_open:
        ser.close()
    plt.ioff()
    plt.show()