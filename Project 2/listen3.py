import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# --- Configuration ---
CSV_FILENAME = "pindrop_binary_dataset.csv"

# Serial configuration
ser = serial.Serial(port='COM6', baudrate=115200, timeout=1)
time.sleep(2) 

# --- Parameters ---
samples_per_batch = 10000
fs = 2000 # Hardware timer Hz
drop_count = 0

# 1. Initialize the CSV file and write column headers if it is a new file
if not os.path.isfile(CSV_FILENAME):
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Headers: label, sample_0 ... sample_N, energy
        headers = ['label'] + [f'sample_{i}' for i in range(samples_per_batch)] + ['energy']
        writer.writerow(headers)

print("--- YES/NO Dataset Collection Mode Started ---")
print(f"Saving to: {CSV_FILENAME}")

try:
    plt.ion() # Interactive mode on
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.show(block=False) # Non-blocking plot window

    # Create a time axis for the ADC plot based on the sampling frequency
    time_axis = np.arange(samples_per_batch) / fs

    while True:
        # --- MENU SYSTEM ---
        print("\n" + "="*40)
        print("Select what you are about to record:")
        print("  [1] - YES (A coin is being dropped)")
        print("  [0] - NO  (Silence / Background Noise)")
        print("  [q] - Quit and Exit")
        print("="*40)
        
        user_choice = input("Enter choice (1/0/q): ").strip().lower()
        
        if user_choice == 'q':
            print("Stopping data collection...")
            break
            
        if user_choice not in ['0', '1']:
            print("Invalid input! Please enter 1, 0, or q.")
            continue
            
        # Dynamically set the class label based on your input
        CURRENT_CLASS_LABEL = int(user_choice)
        
        print(f"\n--> Recording LABEL {CURRENT_CLASS_LABEL}...")
        if CURRENT_CLASS_LABEL == 1:
            print("Drop the coin NOW!")
        else:
            print("Keep quiet, recording background noise...")
        
        # Flush buffer to ensure we only get fresh data AFTER you select the menu option
        ser.reset_input_buffer() 
        
        adc_values = []
        
        # --- INNER LOOP: Collect exactly 10,000 samples ---
        while len(adc_values) < samples_per_batch:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                try:
                    # Extracts numeric value regardless of label format
                    raw_value = "".join(filter(str.isdigit, line))
                    if len(raw_value) == 4:
                        val = float(raw_value)
                        adc_values.append(val)
                        
                        # Print progress every 1000 samples
                        if len(adc_values) % 1000 == 0:
                            print(f"[{len(adc_values)}/{samples_per_batch}] Samples collected...")
                except Exception as e:
                    pass # Ignore corrupted serial lines

        # --- PROCESS AND SAVE DATA ---
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
        print(f"\n*** Data saved successfully! (Total recordings this session: {drop_count}) ***")

        # 3. Visualization
        label_text = "YES (Drop)" if CURRENT_CLASS_LABEL == 1 else "NO (Noise)"
        
        # --- ADC Value Plot (Time Domain) ---
        ax1.clear()
        ax1.plot(time_axis, data, color='blue', alpha=0.8)
        ax1.set_title(f"Raw ADC Value Plot | Label: {label_text} | Max ADC: {np.max(data):.0f}")
        ax1.set_xlabel("Time (Seconds)")
        ax1.set_ylabel("ADC Raw Value (0-4095)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # --- FFT Plot (Frequency Domain) ---
        ax2.clear()
        ax2.plot(freqs, fft_magnitude, color='red', alpha=0.8)
        ax2.set_title(f"Frequency Spectrum (FFT) | Total Energy: {fft_energy:.0f}")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.pause(0.01) # Updates the plot without freezing the script

except KeyboardInterrupt:
    print(f"\nExiting... Total recordings collected this session: {drop_count}")
finally:
    ser.close()