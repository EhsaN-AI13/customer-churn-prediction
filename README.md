# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the IBM Telco Customer Churn dataset.

This project covers the complete workflow from data preprocessing and model evaluation to threshold optimization, automated testing, FastAPI serving, and Docker containerization.

The final system exposes the trained machine learning model through a REST API and is designed as a production-oriented machine learning project.

---

## Project Overview

Customer churn prediction is a binary classification problem where the objective is to identify customers who are likely to leave a service.

The project focuses not only on model training, but also on:

* Data cleaning
* Exploratory data analysis
* Feature preprocessing
* Model evaluation
* Threshold optimization
* Error analysis
* Model persistence
* REST API development
* Automated testing
* Docker containerization
* Reproducible dependency management

The final classification threshold was adjusted from `0.50` to `0.35` to increase churn detection recall.

---

## Dataset

The project uses the **Telco Customer Churn** dataset provided by IBM.

Dataset source:

https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

Dataset characteristics:

* 7,043 customers
* 21 columns
* 19 predictive features after removing `customerID` and the target
* Binary target: `Churn`

### Target Distribution

| Churn | Customers | Percentage |
| ----- | --------: | ---------: |
| No    |     5,174 |     73.46% |
| Yes   |     1,869 |     26.54% |

---

## Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

1. Dataset understanding
2. Data quality analysis
3. Data cleaning
4. Exploratory data analysis
5. Feature and target definition
6. Train/test split
7. Numerical and categorical feature identification
8. Preprocessing pipeline
9. Logistic Regression
10. Random Forest comparison
11. Hyperparameter tuning
12. Model evaluation
13. Threshold optimization
14. Final test evaluation
15. Error analysis
16. Model persistence
17. FastAPI deployment
18. Automated testing
19. Docker containerization

---

## Data Cleaning

The `TotalCharges` column initially contained blank values.

These values were converted to numeric format using:

```python
pd.to_numeric(..., errors="coerce")
```

Missing numerical values are handled through the preprocessing pipeline.

The dataset was also checked for duplicate rows.

Results:

* Blank `TotalCharges` values: 11
* Duplicate rows: 0

---

## Preprocessing

The project uses Scikit-learn `Pipeline` and `ColumnTransformer` to keep preprocessing and model inference consistent.

### Numerical Features

The numerical features include:

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `TotalCharges`

Numerical preprocessing:

* Median imputation
* StandardScaler

### Categorical Features

Categorical preprocessing:

* Most-frequent imputation
* One-hot encoding
* `handle_unknown="ignore"`

The preprocessing steps are integrated directly into the machine learning pipeline.

This helps prevent preprocessing inconsistencies between training and inference.

---

## Model

Several classification approaches were evaluated during the development process.

The final model is:

```text
Preprocessing Pipeline
        +
Logistic Regression
        +
Decision Threshold = 0.35
```

Logistic Regression was selected for the final implementation because it provided strong overall classification performance while allowing the decision threshold to be adjusted for the project's churn-detection objective.

---

## Threshold Optimization

The default classification threshold of `0.50` was evaluated against a lower threshold.

The final threshold used by the deployed API is:

```text
0.35
```

At this threshold, customers with:

```text
churn_probability >= 0.35
```

are classified as:

```text
Churn = Yes
```

This increases recall for the churn class while producing more false positives.

The threshold is therefore a business-oriented decision parameter rather than a property of the underlying probability model.

---

## Final Model Evaluation

The final model was evaluated on a held-out test set containing:

```text
1,409 samples
```

### Threshold = 0.35

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 76.51% |
| Precision | 54.43% |
| Recall    | 70.59% |
| F1 Score  | 61.47% |
| ROC-AUC   | 84.19% |

The ROC-AUC remains independent of the selected classification threshold.

### Threshold Comparison

| Metric   | Threshold 0.50 | Threshold 0.35 |
| -------- | -------------: | -------------: |
| Accuracy |         80.55% |           76.5 |
