import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. The Multi-Input CNN Architecture ---
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

# --- 2. Data Loading & Preparation ---
def load_and_prep_data(csv_filename="pindrop_dataset5.csv"):
    print("Loading dataset...")
    df = pd.read_csv(csv_filename)
    
    # Extract labels (Column 0)
    y = df.iloc[:, 0].values
    
    # Extract Time-Series (Columns 1 to second-to-last)
    X_time = df.iloc[:, 1:-1].values
    
    # Extract Energy (The very last column)
    X_energy = df.iloc[:, -1].values.reshape(-1, 1) 
    
    # --- Normalization ---
    # We must scale time and energy separately because energy numbers are much larger
    scaler_time = StandardScaler()
    X_time_scaled = scaler_time.fit_transform(X_time)
    
    scaler_energy = StandardScaler()
    X_energy_scaled = scaler_energy.fit_transform(X_energy)
    
    # Split the data, keeping time and energy pairs together
    X_time_train, X_time_test, X_en_train, X_en_test, y_train, y_test = train_test_split(
        X_time_scaled, X_energy_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to Tensors
    X_time_train_tensor = torch.tensor(X_time_train, dtype=torch.float32).unsqueeze(1)
    X_en_train_tensor = torch.tensor(X_en_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_time_test_tensor = torch.tensor(X_time_test, dtype=torch.float32).unsqueeze(1)
    X_en_test_tensor = torch.tensor(X_en_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoaders (Now packing 3 things: time, energy, and label)
    train_dataset = TensorDataset(X_time_train_tensor, X_en_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_time_test_tensor, X_en_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Data ready: {len(X_time_train)} train samples, {len(X_time_test)} test samples.")
    return train_loader, test_loader

# --- 3. The Training Loop ---
def train_model():
    train_loader, test_loader = load_and_prep_data()
    model = PinDropMultiInputCNN(num_classes=4)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    epochs = 300 
    
    # --- NEW: Variable to track the best test accuracy ---
    best_test_acc = 0.0
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Unpack all 3 items from the loader
        for time_inputs, energy_inputs, labels in train_loader:
            optimizer.zero_grad() 
            
            # Add noise only to the time-series (keep energy absolute)
            noise = torch.randn_like(time_inputs) * 0.05 
            time_inputs = time_inputs + noise

            # Forward pass: Feed BOTH inputs
            outputs = model(time_inputs, energy_inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # --- Evaluate ---
        model.eval() 
        correct_test = 0
        total_test = 0
        with torch.no_grad(): 
            for time_inputs, energy_inputs, labels in test_loader:
                outputs = model(time_inputs, energy_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        test_acc = 100 * correct_test / total_test
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")
        
        # --- NEW: Save the model if it's the best one we've seen so far ---
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "pindrop_cnn_weights_best.pth")
            print(f"  -> [Update] Best model saved! (Test Acc: {best_test_acc:.1f}%)")

    # Also save the final epoch just in case you want to compare it later
    torch.save(model.state_dict(), "pindrop_cnn_weights_last.pth")
    print(f"\nTraining Complete! Best model saved with {best_test_acc:.1f}% accuracy.")

    # --- 4. Final Evaluation & Confusion Matrix ---
    print("\n--- Loading BEST Model for Final Evaluation ---")
    # --- NEW: Load the best weights before generating the report ---
    model.load_state_dict(torch.load("pindrop_cnn_weights_best.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for time_inputs, energy_inputs, labels in test_loader:
            outputs = model(time_inputs, energy_inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    target_names = ["10cm D / 10cm H", "10cm D / 30cm H", "30cm D / 10cm H", "30cm D / 30cm H"]
    print(classification_report(all_labels, all_preds, target_names=target_names))

if __name__ == "__main__":
    train_model()