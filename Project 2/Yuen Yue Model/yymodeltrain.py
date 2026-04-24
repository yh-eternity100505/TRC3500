import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from scipy.signal import spectrogram

# --- 1. Load Data ---
print("Loading dataset...")
df = pd.read_csv('pindrop_dataset5.csv')
X_raw = df.filter(like='sample_').values
y = df['label'].values

# Determine input size for live test consistency
samples_per_batch = X_raw.shape[1]
print(f"Detected {samples_per_batch} samples per drop in CSV.")
print(f"IMPORTANT: Make sure 'samples_per_batch' in livetest.py is set to {samples_per_batch}!\n")

# --- 2. Convert 1D signals to 2D Spectrograms ---
def create_spectrograms(data):
    specs = []
    for i in range(len(data)):
        # Generate spectrogram: fs is sampling rate (e.g., 1000Hz)
        f, t, Sxx = spectrogram(data[i], fs=1000, nperseg=256, noverlap=128)
        # Log scale helps the CNN see lower intensity patterns
        Sxx_log = np.log(Sxx + 1e-10) 
        specs.append(Sxx_log)
    return np.array(specs)

print("Converting signals to spectrograms...")
X_2d = create_spectrograms(X_raw)

# Reshape for CNN: (Samples, Height, Width, 1 Channel)
X_2d = X_2d.reshape(X_2d.shape[0], X_2d.shape[1], X_2d.shape[2], 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, stratify=y)

# --- 3. Build 2D CNN Model ---
model = models.Sequential([
    Input(shape=(X_2d.shape[1], X_2d.shape[2], 1)), # Modern Keras standard
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# --- 4. Train Model ---
print("\nEpoch | Training Acc | Test Acc")
print("-" * 35)
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=2 
)

# --- 5. Save the model ---
model.save('pindrop_model.keras') # Updated to Keras V3 format
print("Model saved to disk as pindrop_model.keras")