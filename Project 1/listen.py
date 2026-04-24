import serial
import time
import json

# Replace 'COM4' with the actual port your STM32 is connected to
ser = serial.Serial(
    port='COM4',
    baudrate=115200,
    timeout=1  # seconds
)

time.sleep(2)  # give some time for STM32 to reset

# Load calibration parameters
with open('params.txt', 'r') as f:
    lines = f.readlines()
    saved_m = float(lines[0])
    saved_c = float(lines[1])

# --- NEW: Arrays to hold the batch data ---
adc_values = []
ml_values = []
samples_per_batch = 40 

try:
    mode = input("Select operation mode: Calibration(C)/Present(P): ").strip().upper()
    
    if mode == "C":
        print("--- Calibration Mode Started ---")
        while True:
            if ser.in_waiting > 0:  
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(line)
                
    elif mode == "P":
        print(f"Using Calibration: m={saved_m}, c={saved_c}")
        print("--- Present Mode Started ---")
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Check if the line contains the individual ADC data
                if "ADC Value =" in line:
                    try:
                        raw_value = line.split('=')[1].strip()
                        adc_y = float(raw_value)
                        ml_x = (adc_y - saved_c) / saved_m
                        ml_x += 3
                        # Add the values to our arrays
                        adc_values.append(adc_y)
                        ml_values.append(ml_x)
                        
                        print(f"Raw ADC: {adc_y} | Calculated Volume: {ml_x:.2f} ml")
                        
                        # --- NEW: Check if we have collected 40 samples ---
                        if len(adc_values) == samples_per_batch:
                            # Calculate the average from the stored arrays
                            mean_adc = sum(adc_values) / samples_per_batch
                            mean_ml = sum(ml_values) / samples_per_batch
                            
                            print("-" * 50)
                            print(f"BATCH COMPLETE! AVERAGE ADC: {mean_adc:.2f} | AVERAGE VOLUME: {mean_ml:.2f} ml")
                            print("-" * 50)
                            
                            # Clear the arrays so they are empty for the next batch of 40
                            adc_values.clear()
                            ml_values.clear()
                            
                    except (ValueError, IndexError):
                        print(f"Skipping noisy data: {line}")

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    ser.close()