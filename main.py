"""
Main deployment script for Cerebrium
This file contains the prediction function that will be deployed
"""

import joblib
import json
import numpy as np
from typing import Dict, List

# Global variables to store the loaded models
model = None
scaler = None
model_info = None


def init():
    """Initialize the model when the container starts"""
    global model, scaler, model_info

    print("Loading model and scaler...")
    try:
        # Load the trained model and scaler
        model = joblib.load('models/iris_classifier.pkl')
        scaler = joblib.load('models/scaler.pkl')

        # Load model information
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)

        print("Model loaded successfully!")
        print(f"Model accuracy: {model_info['accuracy']:.4f}")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e


def predict(item: Dict) -> Dict:
    """
    Main prediction function that will be called by Cerebrium

    Args:
        item: Dictionary containing the input features

    Returns:
        Dictionary containing the prediction results
    """
    try:
        # Extract features from input
        features = item.get('features', [])

        if not features:
            return {
                'error': 'No features provided. Please provide features as a list.',
                'example': {
                    'features': [5.1, 3.5, 1.4, 0.2]
                }
            }

        if len(features) != 4:
            return {
                'error': 'Expected 4 features but got {}'.format(len(features)),
                'feature_names': model_info['feature_names']
            }

        # Convert to numpy array and reshape for single prediction
        features_array = np.array(features).reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        # Get the predicted class name
        predicted_class = model_info['target_names'][prediction]

        # Create confidence scores for all classes
        confidence_scores = {}
        for i, class_name in enumerate(model_info['target_names']):
            confidence_scores[class_name] = float(prediction_proba[i])

        return {
            'prediction': predicted_class,
            'prediction_index': int(prediction),
            'confidence': float(max(prediction_proba)),
            'all_probabilities': confidence_scores,
            'input_features': {
                feature_name: feature_value
                for feature_name, feature_value in zip(model_info['feature_names'], features)
            }
        }

    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'input_received': item
        }


def health_check() -> Dict:
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_accuracy': model_info['accuracy'] if model_info else None,
        'feature_names': model_info['feature_names'] if model_info else None,
        'target_classes': model_info['target_names'] if model_info else None
    }


# Example usage for testing locally
if __name__ == "__main__":
    # Initialize the model
    init()

    # Test prediction
    test_input = {
        'features': [5.1, 3.5, 1.4, 0.2]  # Example iris setosa
    }

    result = predict(test_input)
    print("Test prediction result:")
    print(json.dumps(result, indent=2))

    # Test health check
    health = health_check()
    print("\nHealth check result:")
    print(json.dumps(health, indent=2))