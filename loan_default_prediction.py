import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

data = pd.read_csv("loan_data.csv")

print(data.head())

print(data.info())

print(data.isnull().sum())


X = data.drop(["customer_id", "default"], axis=1)

y = data["default"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

auc_score = roc_auc_score(y_test, probabilities)

print("ROC-AUC Score:", auc_score)


fpr, tpr, thresholds = roc_curve(y_test, probabilities)

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label="ROC Curve")

plt.plot([0, 1], [0, 1])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()


data['default'].value_counts().plot(
    kind='bar',
    figsize=(6, 4)
)

plt.title("Loan Default Distribution")

plt.xlabel("Default")

plt.ylabel("Count")

plt.grid(True)

plt.show()


def expected_loss(
    credit_lines,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_emp,
    fico_score,
    recovery_rate=0.10
):

    input_data = pd.DataFrame({
        'credit_lines': [credit_lines],
        'loan_amt_outstanding': [loan_amt_outstanding],
        'total_debt_outstanding': [total_debt_outstanding],
        'income': [income],
        'years_emp': [years_emp],
        'fico_score': [fico_score]
    })

    pd_probability = model.predict_proba(input_data)[:, 1][0]

    lgd = 1 - recovery_rate

    expected_loss_value = (
        pd_probability
        * lgd
        * loan_amt_outstanding
    )

    return {
        "Probability_of_Default": pd_probability,
        "Expected_Loss": expected_loss_value
    }


result = expected_loss(
    credit_lines=2,
    loan_amt_outstanding=5000,
    total_debt_outstanding=10000,
    income=60000,
    years_emp=5,
    fico_score=650
)

print("\nPrediction Result:")
print(result)