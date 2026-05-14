import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ModelCheckpoint # Import Checkpoint
from sklearn.model_selection import train_test_split
from scipy.signal import spectrogram
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. Load Data ---
print("Loading dataset...")
df = pd.read_csv('pindrop_dataset5.csv')
X_raw = df.filter(like='sample_').values
y = df['label'].values

samples_per_batch = X_raw.shape[1]

# --- 2. Convert 1D signals to 2D Spectrograms ---
def create_spectrograms(data):
    specs = []
    for i in range(len(data)):
        f, t, Sxx = spectrogram(data[i], fs=1000, nperseg=256, noverlap=128)
        Sxx_log = np.log(Sxx + 1e-10) 
        specs.append(Sxx_log)
    return np.array(specs)

print("Converting signals to spectrograms...")
X_2d = create_spectrograms(X_raw)
X_2d = X_2d.reshape(X_2d.shape[0], X_2d.shape[1], X_2d.shape[2], 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, stratify=y)

# --- 3. Build 2D CNN Model ---
model = models.Sequential([
    Input(shape=(X_2d.shape[1], X_2d.shape[2], 1)), 
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

# --- 4. Setup Checkpoint to Save Best Model ---
# This monitors 'val_loss' and saves only when it improves (gets lower)
checkpoint = ModelCheckpoint(
    'pindrop_model_best.keras', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

# --- 5. Train Model ---
print("\nStarting Training...")
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint], # Add the callback here
    verbose=2 
)

# --- 6. Load the Best Model for Evaluation ---
# Crucial: This ensures the confusion matrix represents the best performing state
print("\nReloading the best model weights for final evaluation...")
model = tf.keras.models.load_model('pindrop_model_best.keras')

# --- 7. Final Evaluation and Confusion Matrix ---
print("\n--- Final Model Evaluation (Best Weights) ---")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

target_names = ["10cm D / 10cm H", "10cm D / 30cm H", "30cm D / 10cm H", "30cm D / 30cm H"]
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))