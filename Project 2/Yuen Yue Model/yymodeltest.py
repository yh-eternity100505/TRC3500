import serial
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import spectrogram
from tensorflow.keras.models import load_model

# --- 1. Load the Trained TensorFlow Model ---
print("Loading Spectrogram CNN Model...")
try:
    model = load_model('pindrop_model.keras') # Now matches train.py format
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    exit() # Exit cleanly so we don't crash later if the model is missing

class_names = {
    0: "10cm Distance / 10cm Height (Class 0)",
    1: "10cm Distance / 30cm Height (Class 1)",
    2: "30cm Distance / 10cm Height (Class 2)",
    3: "30cm Distance / 30cm Height (Class 3)"
}

# --- 2. Serial Configuration ---
# Update to your actual COM port
ser = serial.Serial(port='COM6', baudrate=115200, timeout=1)
time.sleep(2) 

# IMPORTANT: Ensure this matches the print statement from train.py!
samples_per_batch = 10000  
adc_values = []

print("--- Live Spectrogram CNN Classification Started ---")
print("Waiting for coin drop...")

try:
    plt.ion()
    fig, (ax_raw, ax_spec) = plt.subplots(2, 1, figsize=(10, 8))

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            try:
                # Safer numeric extraction: grabs integers or decimals, handles negatives
                raw_values = "".join(filter(str.isdigit, line))
                
                if len(raw_values) == 4:  # Expecting 4 numeric values per line
                    value = int(raw_values)
                    adc_values.append(value)
                    
                    if len(adc_values) % 1000 == 0:
                        print(f"[{len(adc_values)}/{samples_per_batch}] Samples collected...")

                # When a full drop is collected
                if len(adc_values) >= samples_per_batch:
                    data = np.array(adc_values[:samples_per_batch])
                    
                    # --- 3. Process Data into Spectrogram (IMAGE) ---
                    f, t, Sxx = spectrogram(data, fs=1000, nperseg=256, noverlap=128)
                    Sxx_log = np.log(Sxx + 1e-10)
                    
                    # Add Batch and Channel dimensions (1, Height, Width, 1)
                    input_image = Sxx_log[np.newaxis, ..., np.newaxis]
                    
                    # --- 4. Neural Network Inference ---
                    predictions = model.predict(input_image, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class] * 100
                    
                    category_name = class_names.get(predicted_class, "Unknown")
                    
                    print("\n" + "="*50)
                    print(f">>> PREDICTION: {category_name}")
                    print(f">>> CONFIDENCE: {confidence:.1f}%")
                    print("="*50 + "\n")

                    # --- 5. Visualization ---
                    ax_raw.clear()
                    ax_raw.plot(data)
                    ax_raw.set_title(f"Raw Signal - Prediction: {category_name}")
                    
                    ax_spec.clear()
                    ax_spec.pcolormesh(t, f, Sxx_log, shading='gouraud')
                    ax_spec.set_ylabel('Frequency [Hz]')
                    ax_spec.set_xlabel('Time [sec]')
                    ax_spec.set_title("Input Spectrogram (Image)")
                    
                    plt.pause(0.01)
                    
                    # Reset for next capture
                    adc_values = []
                    print("Waiting for next drop...")
                    break
                    
                        
            except Exception as e:
                print(f"Error processing data chunk: {e}")

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    ser.close()
    plt.ioff()
    plt.show()