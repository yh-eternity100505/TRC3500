import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Define the CNN Architecture (MUST exactly match training) ---
class PinDropMultiInputCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(PinDropMultiInputCNN, self).__init__()
        
        # --- Time-Series Feature Extractor ---
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16) 
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32) 
        
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        
        # --- The Merger (Fully Connected) ---
        # The CNN outputs 32 features. We add 1 feature for Energy, so the input is 33.
        self.fc1 = nn.Linear(32 + 1, 64) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_time, x_energy):
        # 1. Process the time series through the CNN
        c = self.pool(F.relu(self.bn1(self.conv1(x_time))))
        c = self.pool(F.relu(self.bn2(self.conv2(c))))
        
        # Pool down and flatten
        c = self.global_pool(c)
        c = torch.flatten(c, 1) # Shape: [Batch, 32]
        
        # 2. Ensure energy is properly shaped [Batch, 1]
        e = x_energy.view(-1, 1) 
        
        # 3. Merge: Concatenate CNN features with the Energy feature
        # This combines shape awareness with total impact force!
        merged = torch.cat((c, e), dim=1) # Shape: [Batch, 33]
        
        # 4. Final Classification
        x = F.relu(self.fc1(merged))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. Load the Trained Weights ---
print("Loading Neural Network...")
model = PinDropMultiInputCNN()

# Load weights and map to CPU (in case you trained on a GPU)
model.load_state_dict(torch.load("pindrop_cnn_weights.pth", map_location=torch.device('cpu')))
model.eval() # CRITICAL: Sets the model to evaluation mode (turns off dropout)
print("Model loaded successfully!\n")

# Map your numerical labels back to physical states
class_names = {
    0: "10cm Height / 10cm Distance",
    1: "10cm Height / 30cm Distance",
    2: "30cm Height / 10cm Distance",
    3: "30cm Height / 30cm Distance"
}

# --- 3. Serial Configuration ---
# Update to your actual COM port
ser = serial.Serial(port='COM6', baudrate=115200, timeout=1)
time.sleep(2) 

samples_per_batch = 10000
adc_values = []

print("--- Live CNN Classification Started ---")
print("Waiting for coin drop...")

try:
    plt.ion() # Turn on interactive mode for live plotting
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            try:
                # Extract numeric value
                raw_value = "".join(filter(str.isdigit, line))
                if len(raw_value) == 4:
                    adc_y = float(raw_value)
                    adc_values.append(adc_y)
                    
                    if len(adc_values) % 500 == 0:
                        print(f"[{len(adc_values)}/{samples_per_batch}] Samples collected...")

                # When a full drop is collected (Use >= for safety!)
                if len(adc_values) >= samples_per_batch:
                    
                    # 1. Prepare Time Data (Grab exactly 10,000 just in case)
                    data = np.array(adc_values[:samples_per_batch])
                    
                    # 2. Normalize Time Data
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    if std_val == 0: std_val = 1e-8 
                    data_scaled = (data - mean_val) / std_val
                    
                    # 3. Calculate Energy Data (Matching your dataset generation)
                    data_centered = data - np.mean(data)
                    fft_result = np.fft.fft(data_centered)
                    fft_magnitude = np.abs(fft_result)[:samples_per_batch // 2]
                    live_energy = np.sum(np.square(fft_magnitude)) / samples_per_batch
                    
                    # 4. Convert BOTH to PyTorch Tensors
                    tensor_time = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    tensor_energy = torch.tensor([[live_energy]], dtype=torch.float32) # Shape [1, 1]
                    
                    # 5. Neural Network Inference
                    with torch.no_grad(): 
                        # ---> CRITICAL FIX: Pass BOTH time and energy into the model <---
                        output = model(tensor_time, tensor_energy)
                        
                        # Get the highest probability prediction
                        predicted_class = torch.argmax(output, dim=1).item()
                        
                        # Calculate confidence percentage
                        probabilities = F.softmax(output, dim=1)[0]
                        confidence = probabilities[predicted_class].item() * 100
                    
                    category_name = class_names.get(predicted_class, "Unknown")
                    
                    print("\n" + "="*50)
                    print(f">>> PREDICTION: {category_name}")
                    print(f">>> CONFIDENCE: {confidence:.1f}%")
                    print("="*50 + "\n")

                    # 6. Live Plotting
                    ax1.clear()
                    ax1.plot(data)
                    ax1.set_title(f"CNN Prediction: {category_name} (Conf: {confidence:.1f}%) | Energy: {live_energy:.0f}")
                    ax1.set_ylabel("ADC Raw Value")
                    ax1.set_xlabel("Sample Index")
                    plt.pause(0.01)
                    
                    # Clear array to instantly listen for the next drop
                    adc_values = []
                    print("Waiting for next drop...")
                    break
                        
            except Exception as e:
                print(f"Error processing: {e}")

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    ser.close()