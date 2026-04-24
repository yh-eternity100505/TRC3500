import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_FILE = "pindrop_dataset.csv"
MODEL_FILE = "coin_drop_model.pkl"


def load_dataset(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Show first few rows
    print("Dataset preview:")
    print(df.head())

    # Check label column exists
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Features = all columns except label
    X = df.drop(columns=["label"]).values

    # Labels = label column
    y = df["label"].values

    print(f"\nDataset shape: {df.shape}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y


def main():
    # Load dataset
    X, y = load_dataset(DATA_FILE)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Optional cross-validation
    print("\nRunning cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print(f"Average CV accuracy: {cv_scores.mean() * 100:.2f}%")

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved as: {MODEL_FILE}")


if __name__ == "__main__":
    main()