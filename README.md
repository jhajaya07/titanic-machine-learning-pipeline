# 🚢 Titanic Machine Learning Pipeline (Task 5)

This project builds a complete **Machine Learning Pipeline** on the Titanic dataset using Python and Scikit-learn.  
The goal is to predict passenger survival using preprocessing, Logistic Regression, evaluation metrics, cross-validation, and model persistence.

---

## 📌 Project Objectives

- Clean and preprocess Titanic dataset
- Handle missing values using imputation
- Encode categorical features
- Scale numerical features
- Train a Logistic Regression classifier
- Evaluate model performance using multiple metrics
- Perform cross-validation for reliable results
- Save and reload the trained model

---

## 🧹 Data Preprocessing

### Selected Features
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

### Target Variable
- Survived

### Missing Value Handling
- Numeric features (`Age`, `Fare`) → filled using **median**
- Categorical feature (`Embarked`) → filled using **most frequent value**

### Feature Engineering & Transformation
- One-Hot Encoding for categorical variables
- Standard Scaling for numeric features

---

## 🤖 Machine Learning Model

### Model Used
- Logistic Regression (`max_iter=1000`)

### Why Logistic Regression?
- Simple and interpretable baseline model
- Effective for binary classification problems
- Fast and efficient

---

## 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- ROC-AUC Score

---

## 🔁 Cross Validation

Performed **5-Fold Cross Validation** using ROC-AUC scoring to ensure reliable performance estimation.

---

## 💾 Model Persistence

The trained pipeline was saved using:

```python
joblib.dump(model, 'model.joblib')
