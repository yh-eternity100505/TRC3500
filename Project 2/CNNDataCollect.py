import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# --- Machine Learning Dataset Configuration ---
# IMPORTANT: Change this label before running the script for each test category!
# 0 = 10cm Distance / 10cm Height
# 1 = 10cm Distance / 30cm Height
# 2 = 30cm Distance / 10cm Height
# 3 = 30cm Distance / 30cm Height
CURRENT_CLASS_LABEL = 3 
CSV_FILENAME = "pindrop_dataset5.csv"

# Serial configuration
ser = serial.Serial(port='COM6', baudrate=115200, timeout=1)
# Flushes any old data out of the buffer so you start fresh
ser.reset_input_buffer() 
time.sleep(1) 

# --- Parameters ---
samples_per_batch = 10000
fs = 2000 # Your hardware timer Hz
adc_values = []
drop_count = 0

# 1. Initialize the CSV file and write column headers if it is a new file
if not os.path.isfile(CSV_FILENAME):
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Headers: label, sample_0 ... sample_N, energy
        headers = ['label'] + [f'sample_{i}' for i in range(samples_per_batch)] + ['energy']
        writer.writerow(headers)

print("--- Immediate Data Collection Started ---")
print(f"Recording data for CLASS: {CURRENT_CLASS_LABEL}")
print(f"Saving to: {CSV_FILENAME}")
print("Recording immediately... Drop the coin!")

try:
    plt.ion() # Interactive mode on
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            try:
                # Extracts numeric value regardless of label format
                raw_value = "".join(filter(str.isdigit, line))
                if len(raw_value) == 4:

                    try:
                        val = float(raw_value)
                        adc_values.append(val)
                    except ValueError:
                        pass

                    
                    # --- RECORDING IMMEDIATELY ---

                    # Print progress every 500 samples
                    if len(adc_values) % 500 == 0:
                        print(f"[{len(adc_values)}/{samples_per_batch}] Samples collected...")

                    # When a full batch of samples is collected
                    if len(adc_values) == samples_per_batch:
                        data = np.array(adc_values)
                        
                        # 1. FFT & Energy Calculations
                        data_centered = data - np.mean(data)
                        fft_result = np.fft.fft(data_centered)
                        fft_magnitude = np.abs(fft_result)[:samples_per_batch // 2]
                        freqs = np.fft.fftfreq(samples_per_batch, 1/fs)[:samples_per_batch // 2]
                        
                        fft_energy = np.sum(np.square(fft_magnitude)) / samples_per_batch
                        
                        # 2. Save the batch AND energy to the CSV
                        with open(CSV_FILENAME, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            # Combine label + samples + energy
                            row_data = [CURRENT_CLASS_LABEL] + data.tolist() + [fft_energy]
                            writer.writerow(row_data)
                        
                        drop_count += 1
                        print(f"*** Data saved successfully! (Class {CURRENT_CLASS_LABEL}) ***")

                        # 3. Visualization
                        ax1.clear()
                        ax1.plot(data)
                        ax1.set_title(f"Class {CURRENT_CLASS_LABEL} Time Domain - Max ADC: {np.max(data)}")
                        ax1.set_xlabel("Sample Index")
                        ax1.set_ylabel("ADC Raw Value")
                        
                        ax2.clear()
                        ax2.plot(freqs, fft_magnitude, color='red')
                        ax2.set_title(f"FFT | Total Energy: {fft_energy:.0f}")
                        ax2.set_xlabel("Frequency (Hz)")
                        ax2.set_ylabel("Magnitude")
                        
                        plt.tight_layout()
                        plt.pause(0.05)
                        
                        # Stop the loop after 1 batch
                        print(f"\nTarget of {samples_per_batch} samples reached. Stopping script...") 
                        # Remove the 'break' and reset the list instead:
                        adc_values = [] 
                        plt.show(block=True) # Change to False so it doesn't halt the code
                        break
                    
            except Exception as e:
                print(f"Error processing: {e}")

except KeyboardInterrupt:
    print(f"\nExiting... Total drops collected this session: {drop_count}")
finally:
    ser.close()