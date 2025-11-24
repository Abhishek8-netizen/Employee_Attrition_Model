import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset


def logistic_regression(num):
    df = pd.read_csv('IBM_Employee_HR_Cleaned.csv')
    # Drop the first unnamed index column, if it exists
    if df.columns[0].startswith('Unnamed: 0'):
        df = df.drop(df.columns[0], axis=1)

    # ============================================
    # Encoding categorical variables
    # ============================================

    label = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label.fit_transform(df[col])

    # ============================================
    # Define target and feature variables
    # ============================================

    y = df['Attrition']                     # Target
    X = df.drop('Attrition', axis=1)        # Features

    # ============================================
    # Train-Test Split (70% Train – 30% Test)
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    smote_model = SMOTE(random_state=42)
    X_train_resample, y_train_resample = smote_model.fit_resample(X_train, y_train)

    # ============================================
    # Feature Scaling
    # ============================================

    scaler = StandardScaler()
    X_train_resample = scaler.fit_transform(X_train_resample)
    X_test = scaler.transform(X_test)

    # ============================================
    # Logistic Regression Model
    # ============================================

    log_model = LogisticRegression(max_iter=1000,class_weight="balanced")

    # Train the model
    log_model.fit(X_train_resample, y_train_resample)

    # Predict
    y_pred = log_model.predict(X_test)
    
    # ============================================
    # Evaluation Metrics
    # ============================================

    y_prob = log_model.predict_proba(X_test)[:,1]

    auc_score = roc_auc_score(y_test,y_prob)

    fpr, tpr, thresholds = roc_curve(y_test,y_prob)
    

    acc_logistic = accuracy_score(y_test, y_pred)
    print("\n✅ ACCURACY SCORE:")
    print(accuracy_score(y_test, y_pred))

    con_matrix_logistic = confusion_matrix(y_test, y_pred)
    print("\n📌 CONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))

    class_report_logistic = classification_report(y_test, y_pred, output_dict=True)
    classReportLogistic = pd.DataFrame(class_report_logistic).transpose()
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))

    if num==0:
        return (acc_logistic,con_matrix_logistic,classReportLogistic)
    else:
        return (fpr,tpr,auc_score)


def decision_tree():
    df = pd.read_csv('IBM_Employee_HR_Cleaned.csv')
    # ------------------------------------------------------------
    # Decision Tree Classifier on IBM Employee HR Cleaned Dataset
    # ------------------------------------------------------------

    # Load dataset

    # Drop unnamed index column if exists
    if df.columns[0].startswith('Unnamed: 0'):
        df = df.drop(df.columns[0], axis=1)

    # ============================================
    # Encoding categorical variables
    # ============================================

    label = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label.fit_transform(df[col])

    # ============================================
    # Define target and feature variables
    # ============================================

    y = df['Attrition']                     # Target
    X = df.drop('Attrition', axis=1)        # Features

    # ============================================
    # Train-Test Split (70% Train – 30% Test)
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    smote_model = SMOTE(random_state=42)
    X_train_resample, y_train_resample = smote_model.fit_resample(X_train,y_train)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # ============================================
    # Decision Tree Model
    # ============================================

    tree_model = DecisionTreeClassifier(
        criterion='gini',        # or "entropy"
        max_depth=None,          # can tune later
        random_state=42,
        class_weight="balanced"
    )

    # Train the model
    tree_model.fit(X_train_resample, y_train_resample)

    # Predict
    y_pred = tree_model.predict(X_test)

    # ============================================
    # Evaluation Metrics
    # ============================================

    acc_decision = accuracy_score(y_test, y_pred)
    print("\nDECISION TREE ACCURACY:")
    print(accuracy_score(y_test, y_pred))

    conf_matrix_decision = confusion_matrix(y_test, y_pred)
    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))

    class_report_decision = classification_report(y_test, y_pred,output_dict=True)
    classReportDecision = pd.DataFrame(class_report_decision).transpose()
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))

    return(acc_decision,conf_matrix_decision,classReportDecision)

def random_forest():
    df = pd.read_csv("IBM_Employee_HR_Cleaned.csv")

    # Drop unnamed index column
    if df.columns[0].startswith('Unnamed:'):
        df = df.drop(df.columns[0], axis=1)

    # Label encode categorical variables
    label = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label.fit_transform(df[col])

    # Target + features
    y = df['Attrition']
    X = df.drop('Attrition', axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)

    # Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_resample, y_train_resample)

    # Extract feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return importance



def train_and_save_logistic_model():
    df = pd.read_csv('IBM_Employee_HR_Cleaned.csv')

    df = df[["Age","Attrition", "MonthlyIncome",
             "YearsAtCompany","OverTime_No","OverTime_Yes"]]
      
    if df.columns[0].startswith('Unnamed: 0'):
        df = df.drop(df.columns[0], axis=1)

    label = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label.fit_transform(df[col])

    y = df['Attrition']
    X = df.drop('Attrition', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resample)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced",random_state=42)
    model.fit(X_train_scaled, y_train_resample)

    y_pred = model.predict(X_test_scaled)

    # Save model and scaler
    with open("logistic_model.pkl","wb") as f:
        pickle.dump(model, f)

    with open ("scaler.pkl","wb") as g:
        pickle.dump(scaler, open("scaler.pkl", "wb"))

    # Accuracy for display
    acc = accuracy_score(y_test, y_pred)
    return acc


def predict_attrition(input_list):

    with open("logistic_model.pkl","rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl","rb") as g:
        scaler = pickle.load(g)

    scaled_input = scaler.transform([input_list])
    prediction = model.predict(scaled_input)[0]
    return prediction


logistic_regression(0)
print()
decision_tree()

