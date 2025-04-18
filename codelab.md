summary: Learn how to build a RAG pipeline using Mistral OCR, chunking, embeddings, Pinecone, ChromaDB, and a Streamlit UI.
id: rag-data-pipeline
categories: Data Engineering, Cloud, AWS, RAG
status: Published
authors: Hishita Thakkar
feedback link: https://your-feedback-link.com
# H-1B Visa Approval Prediction Machine Learning Project

## Introduction

Welcome to the H-1B Visa Approval Prediction Machine Learning project codelab! In this tutorial, you will learn how to build a comprehensive machine learning pipeline to predict the approval status of H-1B visa applications. This project covers data loading, preprocessing, feature engineering, model training, evaluation, and deployment.

---

### What you'll learn

- How to explore and preprocess a real-world dataset
- Techniques to handle high cardinality categorical features
- Methods to address class imbalance using SMOTE
- Building and evaluating multiple classification models
- Saving and deploying a trained model

---

### Prerequisites

- Basic knowledge of Python and pandas
- Familiarity with machine learning concepts
- Jupyter Notebook environment (optional but recommended)

---

Let's get started!

---

## Setup

### Step 1: Clone the repository and prepare your environment

```bash
# Clone the repository (replace URL with actual repo if available)
git clone https://github.com/yourusername/h1b-visa-approval-prediction.git
cd h1b-visa-approval-prediction

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

> **Note:** The dataset file `h1b_2011_2016.csv` should be placed in the project root directory.

---

## Data Loading and Initial Exploration

### Step 2: Load the dataset

Load the H-1B visa application data using pandas.

```python
import pandas as pd

# Load dataset
h1b_data = pd.read_csv('h1b_2011_2016.csv')

# Display first few rows
print(h1b_data.head())
```

### Step 3: Explore key columns

Inspect important columns such as:

- `CASE_STATUS` (visa application status)
- `EMPLOYER_NAME`
- `SOC_NAME` (Standard Occupational Classification)
- `JOB_TITLE`
- `FULL_TIME_POSITION`
- `PREVAILING_WAGE`
- `YEAR`
- `WORKSITE` (city and state)
- `LON` and `LAT` (geographical coordinates)

```python
print(h1b_data.columns)
print(h1b_data['CASE_STATUS'].value_counts())
print(h1b_data[['WORKSITE', 'LON', 'LAT']].head())
```

> **Tip:** Understanding the data distribution helps in planning preprocessing and feature engineering.

---

## Data Cleaning and Preprocessing

### Step 4: Clean the dataset

- Drop unnecessary columns like `Unnamed: 0`.
- Remove rows with missing values.
- Split `WORKSITE` into `CITY` and `STATE`.
- Binarize the target variable `CASE_STATUS` into `ACCEPT_REJECT`.

```python
# Drop unnamed column
h1b_data = h1b_data.drop(columns=['Unnamed: 0'], errors='ignore')

# Drop rows with missing values
h1b_data = h1b_data.dropna()

# Split WORKSITE into CITY and STATE
h1b_data[['CITY', 'STATE']] = h1b_data['WORKSITE'].str.split(',', expand=True)

# Binarize CASE_STATUS
h1b_data['ACCEPT_REJECT'] = h1b_data['CASE_STATUS'].apply(lambda x: 1 if x == 'CERTIFIED' else 0)

print(h1b_data[['CASE_STATUS', 'ACCEPT_REJECT']].head())
```

### Step 5: Clean text columns

- Remove punctuation
- Convert text to uppercase

```python
import string

def clean_text(text):
    if isinstance(text, str):
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.upper()
    return text

text_columns = ['SOC_NAME', 'JOB_TITLE', 'EMPLOYER_NAME', 'CITY']
for col in text_columns:
    h1b_data[col] = h1b_data[col].apply(clean_text)

print(h1b_data[text_columns].head())
```

### Step 6: Handle high cardinality categorical features

Group low-frequency categories into "OTHERS" to reduce noise.

```python
for col in ['SOC_NAME', 'JOB_TITLE', 'EMPLOYER_NAME', 'CITY']:
    freq = h1b_data[col].value_counts(normalize=True)
    mask = h1b_data[col].isin(freq[freq < 0.01].index)
    h1b_data.loc[mask, col] = 'OTHERS'

print(h1b_data[text_columns].nunique())
```

### Step 7: Encode categorical variables

Convert categorical columns to pandas category dtype and encode as integer codes.

```python
for col in text_columns + ['STATE']:
    h1b_data[col] = h1b_data[col].astype('category').cat.codes

print(h1b_data.dtypes)
```

---

## Feature Selection and Multicollinearity Analysis

### Step 8: Analyze feature relationships

Calculate correlation and covariance matrices.

```python
corr_matrix = h1b_data.corr()
print(corr_matrix['ACCEPT_REJECT'].sort_values(ascending=False))
```

### Step 9: Calculate Variance Inflation Factor (VIF)

Detect multicollinearity and drop features with high VIF.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

features = h1b_data.drop(columns=['CASE_STATUS', 'ACCEPT_REJECT', 'WORKSITE'])

vif_data = pd.DataFrame()
vif_data['feature'] = features.columns
vif_data['VIF'] = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]

print(vif_data)

# Drop features with high VIF
features = features.drop(columns=['YEAR', 'FULL_TIME_POSITION', 'LON', 'LAT'])
```

### Step 10: Select top features using SelectKBest

```python
from sklearn.feature_selection import SelectKBest, chi2

X = features
y = h1b_data['ACCEPT_REJECT']

selector = SelectKBest(chi2, k=6)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print('Selected features:', list(selected_features))
```

---

## Data Splitting and Balancing

### Step 11: Split data into training and testing sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

print('Training set size:', X_train.shape)
print('Test set size:', X_test.shape)
```

### Step 12: Address class imbalance with SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print('Balanced training set class distribution:')
print(pd.Series(y_train_bal).value_counts())
```

> **Warning:** Balancing the dataset helps models learn minority classes better but be cautious of overfitting.

---

## Model Building

### Step 13: Train multiple classification models

Train KNN, Decision Tree, Random Forest, and XGBoost models.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Initialize models
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

# Train models
knn.fit(X_train_bal, y_train_bal)
dt.fit(X_train_bal, y_train_bal)
rf.fit(X_train_bal, y_train_bal)

# XGBoost with custom modelfit function

def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, seed=42)
        alg.set_params(n_estimators=cvresult.shape[0])
    alg.fit(X_train, y_train, eval_metric='auc')

xgb_model = xgb.XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42)
modelfit(xgb_model, X_train_bal, y_train_bal)
```

---

## Model Evaluation

### Step 14: Evaluate models on test set

Calculate confusion matrix, accuracy, precision, recall, F1 score, and classification report.

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

models = {'KNN': knn, 'Decision Tree': dt, 'Random Forest': rf, 'XGBoost': xgb_model}

for name, model in models.items():
    y_pred = model.predict(X_test[selected_features])
    print(f'--- {name} Model Evaluation ---')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    # ROC Curve
    y_proba = model.predict_proba(X_test[selected_features])[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()
```

> **Insight:** XGBoost typically achieves the best performance with ~91.6% accuracy and AUC ~0.83.

---

## Prediction Examples

### Step 15: Predict visa approval probabilities

Use trained models to predict acceptance probabilities for sample inputs.

```python
sample_input = X_test[selected_features].iloc[0:3]

print('Random Forest Predictions:')
print(rf.predict_proba(sample_input))

print('XGBoost Predictions:')
print(xgb_model.predict_proba(sample_input))
```

---

## Model Persistence

### Step 16: Save the final model

Save the Random Forest model for future use.

```python
import pickle

with open('h1b_prediction_model_rf2.pk', 'wb') as f:
    pickle.dump(rf, f)

print('Model saved as h1b_prediction_model_rf2.pk')
```

---

## Summary

Congratulations! You have completed the H-1B Visa Approval Prediction project codelab. You have learned how to:

- Load and preprocess a complex dataset
- Handle high cardinality categorical features
- Address class imbalance with SMOTE
- Train and evaluate multiple classification models
- Save a trained model for deployment

Feel free to explore the backend and frontend components (`backend/app.py` and `index.html`) to build a web application interface for your model.

---

## Contact and Contribution

For questions, suggestions, or contributions, please open an issue or submit a pull request in the repository.

Thank you for exploring the H-1B Visa Approval Prediction project!
