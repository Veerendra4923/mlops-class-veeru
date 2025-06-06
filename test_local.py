"""
Local testing script to verify your model works before deployment
"""

import json
from main import init, predict, health_check


def test_model():
    """Test the model locally"""
    print("=" * 50)
    print("TESTING MODEL LOCALLY")
    print("=" * 50)

    # Initialize the model
    print("1. Initializing model...")
    try:
        init()
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return

    # Test health check
    print("\n2. Testing health check...")
    health = health_check()
    print("Health check result:")
    print(json.dumps(health, indent=2))

    # Test predictions with different iris samples
    test_cases = [
        {
            'name': 'Iris Setosa',
            'features': [5.1, 3.5, 1.4, 0.2]
        },
        {
            'name': 'Iris Versicolor',
            'features': [6.4, 3.2, 4.5, 1.5]
        },
        {
            'name': 'Iris Virginica',
            'features': [6.3, 3.3, 6.0, 2.5]
        }
    ]

    print("\n3. Testing predictions...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input features: {test_case['features']}")

        result = predict({'features': test_case['features']})

        if 'error' in result:
            print(f"❌ Prediction failed: {result['error']}")
        else:
            print(f"✅ Predicted class: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   All probabilities: {result['all_probabilities']}")

    # Test error handling
    print("\n4. Testing error handling...")

    # Test with wrong number of features
    error_test = predict({'features': [1.0, 2.0]})  # Only 2 features instead of 4
    if 'error' in error_test:
        print("✅ Error handling works for wrong input size")
    else:
        print("❌ Error handling failed")

    # Test with no features
    error_test2 = predict({})
    if 'error' in error_test2:
        print("✅ Error handling works for missing features")
    else:
        print("❌ Error handling failed")

    print("\n" + "=" * 50)
    print("LOCAL TESTING COMPLETED!")
    print("=" * 50)


if __name__ == "__main__":
    test_model()