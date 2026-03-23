# RI-OncoScan: AI-Based Breast Cancer Classification

A Java-based machine learning project for breast cancer prediction using a simple neural network.

## Features

- **Data Loading**: Reads CSV datasets with breast cancer features
- **Neural Network**: Custom implementation with one hidden layer and sigmoid activation
- **Training & Evaluation**: Trains on 80% of data, evaluates on 20% test set
- **Predictions**: Saves detailed predictions to CSV with probabilities and accuracy

## Dataset Format

The project expects a CSV file with the following structure:

```csv
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,...
842302,M,17.99,10.38,122.8,1001.0,0.1184,...
842517,B,20.57,17.77,132.9,1326.0,0.08474,...
```

- **id**: Unique identifier (ignored by model)
- **diagnosis**: M (malignant) or B (benign), or 1/0
- **Features**: 30 numeric features (radius_mean, texture_mean, etc.)

## Usage

### Prerequisites
- Java 17 or higher
- VS Code with Java extension (recommended)

### Running Tests

Compile and run unit tests:
```bash
javac SimpleTestRunner.java
java SimpleTestRunner
```

Expected output:
```
Running RIOncoScan Unit Tests...

✓ DataSplit test passed
✓ NeuralNetwork evaluation test passed
✓ DataLoader test passed

Test Results: 3/3 tests passed
All tests passed! ✓
```

### Example Output
```
Loading dataset: cancer_data.csv
Loaded 569 samples with 30 features
Training set: 455 samples
Test set: 114 samples
Epoch 1/1500 - loss 0.693147
Epoch 100/1500 - loss 0.234567
...
Epoch 1500/1500 - loss 0.123456
Test Accuracy: 94.74%
Predictions saved to predictions.csv
```

## Output Files

- **predictions.csv**: Contains prediction results
  ```csv
  Index,Predicted,Actual,Probability,Correct
  0,1,1,0.9876,true
  1,0,0,0.1234,true
  ```

## Model Details

- **Architecture**: Input → Hidden Layer (24 neurons) → Output (1 neuron)
- **Activation**: Sigmoid
- **Training**: Gradient descent with binary cross-entropy loss
- **Normalization**: Z-score normalization of features

## Dataset Source

This project is designed to work with the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, available from:
- UCI Machine Learning Repository
- Kaggle datasets
- Convert Excel/ARFF to CSV format

## Project Structure

- `RIOncoScan.java`: Main class and all components
  - `CancerDataLoader`: CSV loading and preprocessing
  - `CancerNeuralNetwork`: Neural network implementation
  - `PredictionSaver`: CSV output writer
  - `DataSplit`: Train/test splitting utility
- `SimpleTestRunner.java`: Unit tests for key components
- `README.md`: This documentation

## VS Code Integration

The project includes `.vscode` configuration for easy debugging and running:
- Press `F5` to debug
- Press `Ctrl+F5` to run
- Press `Ctrl+Shift+B` to build

## License

This project is for educational purposes.