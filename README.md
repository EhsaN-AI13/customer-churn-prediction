# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the IBM Telco Customer Churn dataset.

The project focuses on data cleaning, exploratory data analysis, preprocessing, model comparison, cross-validation, threshold optimization, feature importance, and error analysis.

---

## Project Overview

Customer churn prediction is a binary classification problem where the goal is to identify customers who are likely to leave a service.

In this project, the main objective is not only to build a predictive model, but also to analyze its behavior and optimize the decision threshold to improve churn detection.

The final model prioritizes **churn recall** because missing a customer who is likely to churn can be more costly than incorrectly flagging a customer.

---

## Dataset

The project uses the **Telco Customer Churn** dataset provided by IBM.

Dataset source:

https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

The dataset contains:

- 7,043 customers
- 21 columns
- 19 predictive features after removing the customer ID and target
- Binary target: `Churn`

Target distribution:

| Churn | Customers | Percentage |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

---

## Project Workflow

The project follows an end-to-end machine learning workflow:

1. Dataset Understanding
2. Data Quality and Cleaning
3. Exploratory Data Analysis
4. Feature and Target Definition
5. Train/Test Split
6. Feature Type Identification
7. Preprocessing Pipeline
8. Baseline Logistic Regression
9. Cross-Validation
10. Random Forest
11. Hyperparameter Tuning
12. Model Comparison
13. Threshold Optimization
14. Final Evaluation
15. ROC and Precision-Recall Analysis
16. Feature Importance
17. Error Analysis

---

## Data Cleaning

The `TotalCharges` column initially contained 11 blank values.

These values were converted to numeric format and missing values were handled appropriately.

The dataset was also checked for duplicate rows.

Results:

- Blank `TotalCharges` values: 11
- Duplicate rows: 0
- Final missing values: 0

---

## Exploratory Data Analysis

### Contract Type vs Churn

Churn rate varies significantly across contract types.

| Contract | No Churn | Churn |
|---|---:|---:|
| Month-to-month | 57.29% | 42.71% |
| One year | 88.73% | 11.27% |
| Two year | 97.17% | 2.83% |

Customers with month-to-month contracts show substantially higher churn rates.

### Tenure vs Churn

Customers who churn tend to have shorter tenure.

| Churn | Mean Tenure | Median Tenure |
|---|---:|---:|
| No | 37.57 | 38 |
| Yes | 17.98 | 10 |

This indicates a strong relationship between customer tenure and churn behavior.

---

## Feature and Target Definition

The customer ID was removed because it does not provide meaningful predictive information.

The target variable is:

```text
Churn
The input features consist of 19 predictive variables after removing the customer ID and target column.

Train/Test Split

The dataset was split using a stratified train/test split:

Training set: 80%
Test set: 20%
Random state: 42
Stratification: enabled

Final sizes:

Training: 5,634 samples
Test: 1,409 samples

Stratification preserved the original class distribution.

Preprocessing

A Scikit-learn preprocessing pipeline was used to handle numerical and categorical features.

Numerical Features

The numerical features were:

SeniorCitizen
tenure
MonthlyCharges
TotalCharges

The numerical preprocessing included:

Median imputation
StandardScaler
Categorical Features

Categorical features were processed using:

Most-frequent imputation
One-hot encoding
handle_unknown="ignore"

After preprocessing, the original 19 features were transformed into 45 processed features.

All preprocessing steps were integrated into a Pipeline and ColumnTransformer to reduce the risk of data leakage.

Models

Two main classification models were evaluated:

Logistic Regression

Logistic Regression was used as the baseline model and later selected as the final model.

Random Forest

Random Forest was evaluated as a nonlinear model for comparison.

Hyperparameter tuning was performed using GridSearchCV.

Baseline Logistic Regression

The baseline Logistic Regression model used the default classification threshold of 0.50.

Test results:

Metric	Score
Accuracy	80.55%
Precision	65.72%
Recall	55.88%
F1 Score	60.40%
ROC-AUC	84.21%
Random Forest

The baseline Random Forest achieved:

Metric	Score
Accuracy	78.35%
Precision	61.86%
Recall	48.13%
F1 Score	54.14%
ROC-AUC	82.06%
Hyperparameter Tuning

Random Forest was optimized using GridSearchCV.

Best configuration:

n_estimators = 200
max_depth = 10
min_samples_split = 5

Best cross-validation F1 score:

0.5747

Test performance:

Metric	Score
Accuracy	80.70%
Precision	67.23%
Recall	53.21%
F1 Score	59.40%
ROC-AUC	83.92%

Although the tuned Random Forest achieved slightly higher accuracy and precision, Logistic Regression provided better recall, F1 Score, and ROC-AUC.

Model Comparison

Based on the evaluation results, Logistic Regression was selected as the final model.

The main reason is that the project prioritizes identifying customers who are likely to churn.

Therefore, recall is particularly important.

Threshold Optimization

The default classification threshold of 0.50 was not assumed to be optimal.

Instead, out-of-fold predictions were generated on the training set using 5-fold Stratified Cross-Validation.

Several classification thresholds were evaluated using Precision, Recall, and F1 Score.

The best F1 Score among the evaluated thresholds was obtained at:

Final threshold = 0.35

The threshold was selected using only training data through out-of-fold predictions.

The test set was kept untouched until the final evaluation.

Final Model

The final model consists of:

Logistic Regression
+
Preprocessing Pipeline
+
Optimized Decision Threshold = 0.35

Final test results:

Metric	Score
Accuracy	76.44%
Precision	54.32%
Recall	70.59%
F1 Score	61.40%
ROC-AUC	84.21%

The optimized threshold increased recall from 55.88% to 70.59%.

This means the final model identifies more potential churn cases, while accepting a reduction in precision and accuracy.

Confusion Matrix

At the final threshold of 0.35:

True Negative  = 813
False Positive = 222
False Negative = 110
True Positive  = 264

The number of false negatives decreased from 165 at the default threshold to 110 after threshold optimization.

Feature Importance

Logistic Regression coefficients were analyzed to understand which features were most strongly associated with model predictions.

The strongest features included:

Tenure
Contract type
Internet service type
Monthly charges
Total charges
Paperless billing

Positive coefficients increase the model's predicted probability of churn, while negative coefficients decrease it.

These coefficients represent model associations and should not be interpreted as causal relationships.

Error Analysis

False negatives were analyzed separately because they represent customers who actually churned but were predicted as non-churn.

There were:

110 False Negatives

Among these customers:

62.7% had month-to-month contracts
42.7% used DSL internet service
29.1% used mailed check as payment method

Compared with all customers who churned, false-negative customers showed:

Higher average tenure
Lower average monthly charges
Higher average total charges

These findings describe the model's error patterns and do not imply causality.

Project Structure
customer-churn-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── customer-churn-prediction.ipynb
│
├── data/
│   └── .gitkeep
│
└── results/
    └── figures/

The original dataset is not included in this repository.

It can be downloaded from the IBM dataset source provided above.

Installation

Clone the repository:

git clone https://github.com/EhsaN-AI13/customer-churn-prediction.git
cd customer-churn-prediction

Install dependencies:

pip install -r requirements.txt

Launch Jupyter Notebook:

jupyter notebook

Then open:

notebooks/customer-churn-prediction.ipynb
Technologies
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Logistic Regression
Random Forest
GridSearchCV
Cross-Validation
One-Hot Encoding
StandardScaler
Pipeline
ColumnTransformer
ROC-AUC
Precision / Recall / F1
Threshold Optimization
Limitations

The project has several limitations:

The dataset is a standard public benchmark dataset.
The final threshold was optimized for F1 Score rather than a business-specific cost function.
Only a limited set of classification models was evaluated.
The evaluation uses a single held-out test split.
Feature importance describes model associations rather than causal relationships.
Production deployment and monitoring are not included in this project.
Future Improvements

Potential improvements include:

Cost-sensitive threshold optimization
XGBoost / LightGBM comparison
Probability calibration
SHAP-based model explainability
Business-cost-based evaluation
FastAPI deployment
Docker containerization
Production monitoring
Model drift detection
Conclusion

This project demonstrates an end-to-end approach to customer churn prediction.

Instead of focusing only on accuracy, the project evaluates multiple models, uses cross-validation, performs hyperparameter tuning, analyzes model errors, and optimizes the classification threshold based on the objective of detecting more potential churn cases.

The final solution uses Logistic Regression with an optimized threshold of 0.35 and achieves:

70.59% Recall, 61.40% F1 Score, and 84.21% ROC-AUC on the held-out test set.