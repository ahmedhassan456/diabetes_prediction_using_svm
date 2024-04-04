# Diabetes Prediction with SVM

## Overview
This project utilizes the Support Vector Machine (SVM) algorithm to predict the onset of diabetes based on diagnostic measurements.

## Dataset
The dataset comprises medical attributes such as glucose levels, blood pressure, insulin levels, BMI, and more, along with a binary outcome indicating the presence or absence of diabetes.

## Dependencies
- Numpy
- Pandas
- Scikit-learn

## Installation
```bash
pip install numpy pandas scikit-learn
```

## Usage
To run the prediction model, execute the following:
```bash
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load the dataset
diabetes_dataset = pd.read_csv('data/diabetes.csv')
# Add your data preprocessing and model training code here
```

## Model
The SVM model is trained on the preprocessed dataset to classify patients as diabetic or non-diabetic.

## Results
The model’s performance is evaluated using metrics such as accuracy score, confusion matrix, and F1 score.

## Conclusion
This SVM-based model demonstrates the potential of machine learning in predicting diabetes effectively.
