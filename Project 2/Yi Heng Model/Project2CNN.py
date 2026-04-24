import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. The Multi-Input CNN Architecture ---
class PinDropMultiInputCNN(nn.Module):
    def __init__(self, sequence_length=1024, num_classes=4):
        super(PinDropMultiInputCNN, self).__init__()
        
        # Head 1: 1D Convolution for Time-Series Shape
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        
        cnn_flattened_size = 32 * (sequence_length // 16) 
        
        # The Merge: We add +1 to the linear input size to make room for the Energy feature
        self.fc1 = nn.Linear(cnn_flattened_size + 1, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_time, x_energy):
        # 1. Process the time series through the CNN
        c = self.pool(F.relu(self.conv1(x_time)))
        c = self.pool(F.relu(self.conv2(c)))
        c = torch.flatten(c, 1) # Flattens to [Batch_Size, cnn_flattened_size]
        
        # 2. Ensure energy is properly shaped [Batch_Size, 1]
        e = x_energy.view(-1, 1) 
        
        # 3. Merge: Concatenate CNN features with the Energy feature
        merged = torch.cat((c, e), dim=1) 
        
        # 4. Final Classification
        x = F.relu(self.fc1(merged))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. Data Loading & Preparation ---
def load_and_prep_data(csv_filename="pindrop_dataset.csv"):
    print("Loading dataset...")
    # Load the CSV
    df = pd.read_csv(csv_filename)
    
    # Extract labels (Col 0), Time Series (Cols 1 to 1024), and Energy (Col 1025)
    y = df['label'].values
    X_time = df.iloc[:, 1:1025].values 
    X_energy = df['energy'].values 
    
    # --- NORMALIZATION ---
    # 1. Time Series: Subtract DC offset and divide by fixed ADC max (2048)
    # This preserves the physical amplitude difference between 10cm and 30cm drops
    X_time_scaled = np.zeros_like(X_time, dtype=float)
    for i in range(len(X_time)):
        X_time_scaled[i] = (X_time[i] - np.mean(X_time[i])) / 2048.0 
        
    # 2. Energy: Values are massive, so we use Standard Scaling (Z-score)
    scaler_energy = StandardScaler()
    X_energy_scaled = scaler_energy.fit_transform(X_energy.reshape(-1, 1)).flatten()
    
    # --- SPLITTING ---
    # We split all three arrays simultaneously to keep them perfectly synced
    X_time_train, X_time_test, X_energy_train, X_energy_test, y_train, y_test = train_test_split(
        X_time_scaled, X_energy_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- CONVERT TO TENSORS ---
    X_time_train_tensor = torch.tensor(X_time_train, dtype=torch.float32).unsqueeze(1) # Shape: (N, 1, 1024)
    X_energy_train_tensor = torch.tensor(X_energy_train, dtype=torch.float32)          # Shape: (N)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_time_test_tensor = torch.tensor(X_time_test, dtype=torch.float32).unsqueeze(1)
    X_energy_test_tensor = torch.tensor(X_energy_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create PyTorch DataLoaders (Notice we pass 3 tensors now!)
    train_dataset = TensorDataset(X_time_train_tensor, X_energy_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_time_test_tensor, X_energy_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Data ready: {len(X_time_train)} train samples, {len(X_time_test)} test samples.")
    return train_loader, test_loader, scaler_energy

# --- 3. The Training Loop ---
def train_model():
    # Setup
    train_loader, test_loader, scaler_energy = load_and_prep_data()
    model = PinDropMultiInputCNN(sequence_length=1024, num_classes=4)
    
    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    epochs = 200 # Since the network has more features to balance, you might need 100-200 epochs
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Unpack the 3 variables from our DataLoader
        for inputs_time, inputs_energy, labels in train_loader:
            optimizer.zero_grad() 
            
            # Forward pass: Feed BOTH inputs into the model
            outputs = model(inputs_time, inputs_energy)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # --- Evaluate on Test Set ---
        model.eval() 
        correct_test = 0
        total_test = 0
        with torch.no_grad(): 
            for inputs_time, inputs_energy, labels in test_loader:
                outputs = model(inputs_time, inputs_energy)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        test_acc = 100 * correct_test / total_test
        
        # Print update every 10 epochs to keep the console clean
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

    # Save the trained weights
    torch.save(model.state_dict(), "pindrop_cnn_weights.pth")
    print("\nTraining Complete! Model saved as 'pindrop_cnn_weights.pth'")
    
    # Save the energy scaler so the live script knows how to normalize incoming energy

    joblib.dump(scaler_energy, 'energy_scaler.pkl')
    print("Energy scaler saved as 'energy_scaler.pkl'")

if __name__ == "__main__":
    train_model()