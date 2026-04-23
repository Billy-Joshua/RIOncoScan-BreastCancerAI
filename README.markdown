# RIOncoScan-BreastCancerAI

## Overview
RIOncoScan is a Java-based AI project for predicting breast cancer. 
It uses a neural network to classify tumors as **benign** or **malignant**. 
The system can load real-world datasets, train a model, evaluate accuracy, and save predictions to CSV.

 Features
- Loads CSV datasets with tumor features and labels (M/B)
- Splits dataset into training and testing sets
- Neural network with one hidden layer
- Trains model with gradient descent
- Evaluates test accuracy
- Saves predictions to `predictions.csv`

 Dataset
- Source: [WDBC dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- CSV format:  
`ID, Feature1, Feature2, ..., Feature30, Diagnosis (M/B)`
- Optional: remove missing/invalid rows, normalize features

 Usage
```bash
# Compile Java files
javac RIOncoScan.java

 Run the project (default input: cancer_data.csv)
java RIOncoScan

# Optional: specify input and output files
java RIOncoScan wdbc.data predictions.csv
pure JAVA
