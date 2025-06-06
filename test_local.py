

import json
from main import init, predict, health_check


def test_model():
    print("=" * 50)
    print("TESTING MODEL LOCALLY")
    print("=" * 50)

    print("1. Initializing model...")
    try:
        init()
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return
    print("\n2. Testing health check...")
    health = health_check()
    print("Health check result:")
    print(json.dumps(health, indent=2))
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
    print("\n4. Testing error handling...")
    error_test = predict({'features': [1.0, 2.0]}) 
    if 'error' in error_test:
        print("✅ Error handling works for wrong input size")
    else:
        print("❌ Error handling failed")
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
