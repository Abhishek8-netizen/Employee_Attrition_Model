# IBM Employee Attrition Prediction System

## Project Overview

This project is a **Machine Learning–powered Employee Attrition Prediction system** built using the IBM HR Analytics dataset. It combines **data cleaning, exploratory data analysis, ML models, evaluation metrics, and an interactive Streamlit web application** to help organizations understand and predict employee attrition.

The application allows users to:

* Upload or use a built-in dataset
* Clean and preprocess HR data
* Visualize key attrition patterns
* Train and evaluate ML models
* Identify key factors influencing attrition
* Predict whether an employee is likely to leave

---

## Features

* Dataset upload & preprocessing
* Automated data cleaning & encoding
* Interactive visualizations (bar charts, heatmaps, KDE plots, etc.)
* Machine Learning models:

  * Logistic Regression
  * Decision Tree
  * Random Forest (feature importance)
* Model evaluation with accuracy, confusion matrix, ROC curve & AUC
* Insights on key attrition drivers
* Real-time employee attrition prediction
* Interactive Streamlit dashboard

---

## Machine Learning Pipeline

1. **Data Cleaning**

   * Removal of duplicates and unnecessary columns
   * Handling missing values
   * Encoding categorical variables

2. **Exploratory Data Analysis (EDA)**

   * Attrition distribution
   * Income vs Attrition
   * Age and tenure patterns
   * Overtime and commute impact

3. **Model Training**

   * Train-test split (70/30)
   * SMOTE for class imbalance
   * Feature scaling

4. **Evaluation**

   * Accuracy Score
   * Confusion Matrix
   * Precision, Recall, F1-score
   * ROC Curve & AUC

5. **Prediction**

   * Trained Logistic Regression model saved using Pickle
   * Real-time predictions via Streamlit UI

---

## Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * pandas, numpy
  * matplotlib, seaborn
  * scikit-learn
  * imbalanced-learn (SMOTE)
  * streamlit
* **Model Persistence:** pickle

---

## Key Insights

* **OverTime** is the strongest predictor of attrition
* Employees with **lower income** show higher attrition rates
* **Newer employees** are more likely to leave
* **Long commuting distance** increases attrition probability
* Younger employees tend to switch jobs more frequently

---

## Model Performance (Sample)

* Logistic Regression Accuracy: ~77%
* Strong recall for attrition class after SMOTE balancing
* ROC-AUC shows good class separation

---

## License

This project is for educational and academic purposes.

---
