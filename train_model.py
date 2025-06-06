"""
Training script for a simple classification model
This will create a model that classifies iris flowers
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def train_model():
    """Train a neural network classifier on the Iris dataset"""

    print("Loading Iris dataset...")
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training neural network...")
    # Create and train the neural network
    model = MLPClassifier(
        hidden_layer_sizes=(10, 5),
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the model and scaler
    joblib.dump(model, 'models/iris_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    # Save feature names and target names
    model_info = {
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist(),
        'accuracy': accuracy
    }

    import json
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f)

    print("Model saved successfully!")
    return model, scaler, model_info


if __name__ == "__main__":
    train_model()