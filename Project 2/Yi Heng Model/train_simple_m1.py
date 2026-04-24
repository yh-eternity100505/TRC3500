import pandas as pd
import numpy as np
import scipy.fft as fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- Configuration ---
CSV_FILE = 'pindrop_dataset4.csv'
FS = 2000  # Your STM32 sampling frequency in Hz

def extract_features(df):
    print("Extracting features... this might take a few seconds.")
    # Identify sample columns (ignoring label and the raw 'energy' column if present)
    sample_cols = [c for c in df.columns if c.startswith('sample_')]
    
    # Clip data to valid 12-bit ADC range to remove any weird hardware glitches
    df_samples = df[sample_cols].clip(0, 4095)
    
    # Create an empty dataframe to hold our new condensed features
    features = pd.DataFrame()
    features['label'] = df['label']
    
    # Feature 1: Peak-to-Peak Amplitude
    print("Calculating Peak-to-Peak Amplitude...")
    features['ptp_amp'] = df_samples.max(axis=1) - df_samples.min(axis=1)
    
    # Feature 2: Signal Energy (Sum of squared mean-centered amplitudes)
    print("Calculating Signal Energy...")
    # We subtract the mean (DC offset/1.65V bias) so we only measure the vibration
    means = df_samples.mean(axis=1).values[:, np.newaxis]
    centered_samples = df_samples.values - means
    features['energy'] = np.sum(centered_samples ** 2, axis=1)
    
    # Feature 3: Peak Frequency using Fast Fourier Transform (FFT)
    print("Calculating Peak Frequency via FFT...")
    def get_peak_freq(row_idx):
        sig = centered_samples[row_idx]
        yf = np.abs(fft.rfft(sig))
        xf = fft.rfftfreq(len(sig), 1/FS)
        # Find the index of the maximum frequency spike
        idx = np.argmax(yf)
        return xf[idx]
        
    features['peak_freq'] = [get_peak_freq(i) for i in range(len(centered_samples))]
    
    return features

def main():
    # 1. Load Data
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_FILE}'. Make sure it is in the same folder.")
        return

    # 2. Extract Features
    features_df = extract_features(df)
    
    # 3. Prepare Data for Machine Learning
    X = features_df[['ptp_amp', 'energy', 'peak_freq']]
    y = features_df['label']
    
    # Split into 80% training and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train the Classifier
    print("\nTraining Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Evaluate the Model
    y_pred = clf.predict(X_test)
    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Check which feature was the most useful
    print("\nFeature Importances (What the model relied on most):")
    for name, importance in zip(X.columns, clf.feature_importances_):
        print(f"{name}: {importance * 100:.2f}%")
        
    # 7. Save model for live testing later
    joblib.dump(clf, 'pindrop_rf_model.pkl')
    print("\nModel saved as 'pindrop_rf_model.pkl'")

if __name__ == "__main__":
    main()