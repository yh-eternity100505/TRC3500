import serial
import time
import numpy as np
import matplotlib.pyplot as plt

# Serial configuration
ser = serial.Serial(port='COM6', baudrate=115200, timeout=1)
time.sleep(2) 

# --- FFT Parameters ---
samples_per_batch = 1024  # Use powers of 2 for FFT (e.g., 128, 256, 512)
adc_values = []
start_time = 0  # Stopwatch variable to track actual sampling rate

print("--- FFT Mode Started ---")
print(f"Waiting for data on COM6. Collecting {samples_per_batch} samples...")

try:
    plt.ion() # Turn on interactive mode for live plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            try:
                # Extracts numeric value directly (Assuming STM32 sends just the number now)
                raw_value = "".join(filter(str.isdigit, line))
                if raw_value:
                    adc_y = float(raw_value)
                    adc_values.append(adc_y)
                    
                    # Start the stopwatch on the very first sample
                    if len(adc_values) == 1:
                        start_time = time.time()

                    print(f"[{len(adc_values)}/{samples_per_batch}] Collected ADC: {adc_y}")
                
                if len(adc_values) == samples_per_batch:
                    # --- DYNAMIC FREQUENCY CALCULATION ---
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    actual_fs = (samples_per_batch - 1) / elapsed_time
                    
                    print("\n" + "="*40)
                    print(f"--> ACTUAL SAMPLING RATE: {actual_fs:.2f} Hz <--")
                    print("="*40 + "\n")

                    # 1. Prepare Time-Series Data
                    data = np.array(adc_values)
                    
                    # --- Data Features for Impact Classification ---
                    max_value = np.max(data) 
                    max_index = np.argmax(data)
                    min_value = np.min(data)
                    min_index = np.argmin(data)
                    
                    peak_to_peak = max_value - min_value
                    
                    # Calculate Total Energy (Sum of absolute deviation from baseline)
                    baseline = np.mean(data)
                    data_centered = data - baseline 
                    total_energy = np.sum(np.abs(data_centered))
                    
                    # 2. Perform FFT
                    # Remove DC offset to see vibrations clearly
                    data_detrended = data - baseline 
                    fft_result = np.fft.fft(data_detrended)
                    fft_magnitude = np.abs(fft_result)[:samples_per_batch // 2]
                    
                    # Use the REAL calculated frequency here, not a guessed number!
                    freqs = np.fft.fftfreq(samples_per_batch, 1/actual_fs)[:samples_per_batch // 2]

                    # --- Find the MAXIMUM frequency magnitude in the FFT ---
                    fft_max_index = np.argmax(fft_magnitude)
                    fft_max_freq = freqs[fft_max_index]
                    fft_max_mag = fft_magnitude[fft_max_index]

                    # 3. Plotting
                    ax1.clear()
                    ax1.plot(data)
                    # Updated title to show the new features for your classification algorithm
                    ax1.set_title(f"Time Domain | Pk-Pk Swing: {peak_to_peak} | Total Energy: {total_energy:.0f}")
                    ax1.set_ylabel("ADC Raw Value")

                    # Annotate the max and min on the time-domain graph
                    ax1.plot(max_index, max_value, 'ro')  # Red dot on max
                    ax1.plot(min_index, min_value, 'go')  # Green dot on min

                    ax2.clear()
                    ax2.plot(freqs, fft_magnitude, color='red')
                    ax2.set_title(f"Frequency Domain (FFT) - Peak Freq: {fft_max_freq:.2f} Hz (Mag: {fft_max_mag:.2f})")
                    ax2.set_xlabel("Frequency (Hz)")
                    ax2.set_ylabel("Magnitude")

                    # --- Annotate the peak on the FFT graph ---
                    ax2.plot(fft_max_freq, fft_max_mag, 'bo')  # Blue dot on FFT peak
                    ax2.annotate(f'{fft_max_freq:.2f} Hz', 
                                 xy=(fft_max_freq, fft_max_mag), 
                                 xytext=(10, 10) 
                    )

                    plt.pause(0.01)
                    
                    # Clear for next batch
                    adc_values = []

                    filename = f"Test {time.strftime('%H-%M-%S')}.png"
                    plt.savefig(filename)
                    
                    print("Batch complete. Displaying plot...")
                    plt.show(block=True)

                    # --- Stop the loop after the first batch ---
                    print("Target samples reached. Stopping...")
                    break
                    
            except Exception as e:
                print(f"Error processing: {e}")

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    ser.close()