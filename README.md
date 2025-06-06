# mlops-class-veeru
# MLOPs Take Home - Iris Classification on Cerebrium

## Project Overview
This project demonstrates deploying a neural network classification model on Cerebrium's serverless GPU platform. The model classifies iris flowers into three species based on their measurements.

## Prerequisites
- Python 3.8+
- Git
- GitHub account
- Cerebrium account (sign up at https://www.cerebrium.ai/)

## Project Structure
```
mlops-cerebrium-project/
├── main.py                 # Main deployment script
├── train_model.py         # Model training script
├── test_local.py          # Local testing script
├── cerebrium.toml         # Cerebrium configuration
├── requirements.txt       # Python dependencies
├── models/               # Directory for saved models
│   ├── iris_classifier.pkl
│   ├── scaler.pkl
│   └── model_info.json
└── README.md             # This file
```

## Step-by-Step Deployment

### 1. Setup Environment
```bash
# Clone your repository
git clone https://github.com/yourusername/mlops-classification-deployment.git
cd mlops-classification-deployment

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This will:
- Load the Iris dataset
- Train a neural network classifier
- Save the model, scaler, and metadata to the `models/` directory

### 3. Test Locally
```bash
python test_local.py
```
This ensures your model works correctly before deployment.

### 4. Set Up Cerebrium

#### Install Cerebrium CLI
```bash
pip install cerebrium
```

#### Login to Cerebrium
```bash
cerebrium login
```
Follow the prompts to authenticate with your Cerebrium account.

#### Deploy the Model
```bash
cerebrium deploy
```

### 5. Test the Deployed Model

After deployment, you'll get an endpoint URL. Test it using:

```python
import requests
import json

# Replace with your actual endpoint URL
url = "https://your-endpoint.cerebrium.ai/predict"

# Test data
test_data = {
    "features": [5.1, 3.5, 1.4, 0.2]  # Iris setosa example
}

response = requests.post(url, json=test_data)
print(json.dumps(response.json(), indent=2))
```

### 6. API Usage Examples

#### Prediction Request
```json
{
    "features": [5.1, 3.5, 1.4, 0.2]
}
```

#### Prediction Response
```json
{
    "prediction": "setosa",
    "prediction_index": 0,
    "confidence": 0.9999,
    "all_probabilities": {
        "setosa": 0.9999,
        "versicolor": 0.0001,
        "virginica": 0.0000
    },
    "input_features": {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
}
```

#### Health Check
```bash
curl https://your-endpoint.cerebrium.ai/health
```

## Model Information

### Dataset
- **Name**: Iris Dataset
- **Features**: 4 numerical features (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 species (setosa, versicolor, virginica)
- **Samples**: 150 total samples

### Model Architecture
- **Type**: Multi-layer Perceptron (Neural Network)
- **Hidden Layers**: 2 layers with 10 and 5 neurons
- **Activation**: ReLU (default)
- **Preprocessing**: StandardScaler for feature normalization

### Performance
- **Training Accuracy**: ~96-98%
- **Test Accuracy**: ~95-100% (depends on random split)

## Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure you ran `python train_model.py` first
   - Check that `models/` directory exists with all required files

2. **Cerebrium authentication issues**
   - Run `cerebrium login` again
   - Check your API key in Cerebrium dashboard

3. **Deployment failures**
   - Verify `cerebrium.toml` configuration
   - Check all dependencies are listed correctly
   - Ensure Python version compatibility

4. **Prediction errors**
   - Verify input format matches expected schema
   - Check that you're sending exactly 4 features
   - Ensure features are numerical values

### Testing Different Iris Samples

```python
# Iris Setosa (typically small petals)
setosa_sample = [5.1, 3.5, 1.4, 0.2]

# Iris Versicolor (medium-sized)
versicolor_sample = [6.4, 3.2, 4.5, 1.5]

# Iris Virginica (typically large petals)
virginica_sample = [6.3, 3.3, 6.0, 2.5]
```

## Video Recording Guide

For your code walkthrough video, cover:

1. **Project Overview** (2-3 minutes)
   - Explain the problem and solution
   - Show project structure

2. **Code Walkthrough** (15-20 minutes)
   - `train_model.py`: Model training process
   - `main.py`: Deployment logic and prediction function
   - `cerebrium.toml`: Configuration explanation
   - Local testing with `test_local.py`

3. **Deployment Process** (10-15 minutes)
   - Cerebrium CLI setup
   - Deployment command
   - Monitoring deployment status

4. **Testing and Validation** (5-10 minutes)
   - Test the deployed endpoint
   - Show different prediction examples
   - Demonstrate error handling

5. **Wrap-up** (2-3 minutes)
   - Summary of what was accomplished
   - Potential improvements or next steps

## Next Steps / Improvements

- Add model versioning
- Implement A/B testing
- Add monitoring and logging
- Scale to larger datasets
- Add more sophisticated preprocessing
- Implement model retraining pipeline

## Resources

- [Cerebrium Documentation](https://docs.cerebrium.ai/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Iris Dataset Information](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
