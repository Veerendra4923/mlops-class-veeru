

import joblib
import json
import numpy as np
from typing import Dict, List
model = None
scaler = None
model_info = None


def init():
    global model, scaler, model_info

    print("Loading model and scaler...")
    try:
        model = joblib.load('models/iris_classifier.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)

        print("Model loaded successfully!")
        print(f"Model accuracy: {model_info['accuracy']:.4f}")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e


def predict(item: Dict) -> Dict:
    try:
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

        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        predicted_class = model_info['target_names'][prediction]
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

if __name__ == "__main__":
  
    init()

   
    test_input = {
        'features': [5.1, 3.5, 1.4, 0.2]  
    }

    result = predict(test_input)
    print("Test prediction result:")
    print(json.dumps(result, indent=2))

    # Test health check
    health = health_check()
    print("\nHealth check result:")
    print(json.dumps(health, indent=2))
