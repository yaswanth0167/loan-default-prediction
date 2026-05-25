---

# README 2 — Loan Default Prediction Model

```markdown
# Loan Default Prediction Model

## Overview

This project focuses on predicting the Probability of Default (PD) for borrowers using Machine Learning techniques. The model helps financial institutions estimate potential credit risk and expected losses based on customer financial information.

This project was completed as part of the J.P. Morgan Quantitative Research Virtual Experience Program by Forage.

---

## Objectives

- Predict borrower default probability
- Estimate expected financial losses
- Analyze customer financial behavior
- Build a credit risk prediction system
- Visualize model performance

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

---

## Dataset Features

- customer_id
- credit_lines
- loan_amt_outstanding
- total_debt_outstanding
- income
- years_emp
- fico_score
- default

---

## Machine Learning Workflow

1. Load loan dataset
2. Data preprocessing
3. Feature selection
4. Train-test split
5. Logistic Regression model training
6. Prediction and evaluation
7. Expected loss estimation

---

## Formula Used

Expected Loss = PD × LGD × EAD

Where:
- PD = Probability of Default
- LGD = Loss Given Default
- EAD = Exposure at Default

---

## Features

- Loan default prediction
- Probability estimation
- Expected loss calculation
- ROC Curve analysis
- Classification metrics

---

## Visualizations Included

- ROC Curve
- Loan Default Distribution
- Prediction Performance Metrics

---

## Model Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC-AUC Score

---

## How to Run

```bash
python loan_default_prediction.py


Project Structure
loan-default-prediction/ │ 
├── loan_default_prediction.py 
├── loan_data.csv 
├── README.md