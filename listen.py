import serial
import time
import json

# Replace 'COM3' with the actual port your STM32 is connected to
ser = serial.Serial(
    port='COM4',
    baudrate=115200,
    timeout=1  # seconds
)

time.sleep(2)  # give some time for STM32 to reset

# How to load it later:
with open('params.txt', 'r') as f:
    lines = f.readlines()
    saved_m = float(lines[0])
    saved_c = float(lines[1])

try:
    while True:
        if ser.in_waiting > 0:  # check if data is available
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)
            
                
except KeyboardInterrupt:
    print("Exiting...")

ser.close()