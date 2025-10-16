"""
svm.py
-------
This script trains three Support Vector Machine (SVM) classifiers
(Linear, RBF, and Polynomial) using the best hyperparameters previously found
from `svm_scheduler.py`.

Workflow:
  1. Convert the Excel sheet into a CSV (if it isn't already in CSV format)
  2. Load and preprocess the Dry Bean dataset
  3. Split the data into training and test sets
  4. Scale features using the training set
  5. Train and evaluate SVM models
  6. Display metrics (accuracy, F1-scores, confusion matrices)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ============================================================
# Helper functions
# ============================================================

def report_line(tag, acc, f1m, f1w):
    """Nicely formatted printout of model performance metrics."""
    print(f"{tag} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(Model="None", Macro_F1_Test=None, Macro_F1_Train=None, Accuracy=None):
    """Appends model performance results to Results.csv"""
    filename = Path("Results.csv")

    data = {
        "Model" : [Model],
        "Macro_F1 (Test)" : [Macro_F1_Test],
        "Macro_F1 (Train)": [Macro_F1_Train],
        "Accuracy (Test)" : [Accuracy]
    }

    df_new = pd.DataFrame(data)

    if filename.exists():
        df_new.to_csv(filename, index=False, mode='a', header=False)
    else:
        df_new.to_csv(filename, index=False, mode='w', header=True)

def displaying_the_confusion_matrix(y_pred, y_test_or_train, name, classes, t):
    """Displays and optionally saves a confusion matrix plot."""
    rev_classes = {value : key for key, value in classes.items()}
    label_names = [rev_classes[i] for i in sorted(rev_classes.keys())]

    cm = confusion_matrix(y_test_or_train, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", ax=plt.gca())
    plt.title(f"Confusion Matrix - {name} ({t})")
    plt.tight_layout()

    # Uncomment this if you want the plot to be saved
    #saving_name = f"{name} ({t}).png"
    #plt.savefig(saving_name)

    plt.show()

def print_results(y_train, y_test, yhat_tr, yhat_te, name, best_param, classes):
    """
    Prints full training/testing evaluation results for a model.
    Includes accuracy, F1 scores, confusion matrices, and classification report.
    """ 
    # ----------------------- TRAIN ----------------------
    print("\n=== TRAIN ===")
    print(f"Accuracy: {accuracy_score(y_train, yhat_tr):.3f}")
    print(f"F1 (macro): {f1_score(y_train, yhat_tr, average='macro'):.3f}")
    print(f"F1 (weighted): {f1_score(y_train, yhat_tr, average="weighted"):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_train, yhat_tr))
    displaying_the_confusion_matrix(y_pred=yhat_tr, y_test_or_train=y_train, name=name, classes=classes, t="TRAIN")

    # ----------------------- TEST ----------------------
    print("\n=== TEST ===")
    print(f"Accuracy: {accuracy_score(y_test, yhat_te):.3f}")
    print(f"F1 (macro): {f1_score(y_test, yhat_te, average='macro'):.3f}")
    print(f"F1 (weighted): {f1_score(y_test, yhat_te, average="weighted"):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, yhat_te))
    displaying_the_confusion_matrix(y_pred=yhat_te, y_test_or_train=y_test, name=name, classes=classes, t="TEST")
    print("\nClassification report (TEST):")
    print(classification_report(y_test, yhat_te, digits=3))

    f1m_test = f1_score(y_test, yhat_te, average='macro')
    f1m_train = f1_score(y_train, yhat_tr, average='macro')
    f1w_test = f1_score(y_test, yhat_te, average="weighted")
    acc = accuracy_score(y_test, yhat_te)
    report_line(tag=best_param, acc=acc, f1m=f1m_test, f1w=f1w_test)

    # Uncomment this if you want the results to be saved to an external CSV file
    # saving_results(Model=best_param, Macro_F1_Test=f1m_test, Macro_F1_Train=f1m_train, Accuracy=acc)


# ============================================================
# 1) Excel sheet to CSV
# ============================================================

# Convert the excel sheet to a CSV if the CSV does not exist
filename = Path("Dry_Bean_Dataset.csv")

if not filename.exists():
    # Load Excel file
    df = pd.read_excel("Dry_Bean_Dataset.xlsx", sheet_name="Dry_Beans_Dataset")

    # Save as CSV
    df.to_csv("Dry_Bean_Dataset.csv", index=False, encoding="utf-8-sig")


# ============================================================
# 2) Load the Dry Bean Dataset
# ============================================================

df = pd.read_csv("Dry_Bean_Dataset.csv")

# ============================================================
# 3) Encode the categorical class column
# ============================================================

# Manual mapping of bean types to numeric labels
classes = {
    "SEKER"    : 1,
    "BARBUNYA" : 2, 
    "BOMBAY"   : 3, 
    "CALI"     : 4, 
    "HOROZ"    : 5,   
    "SIRA"     : 6,
    "DERMASON" : 7
}

df["Class_num"] = df["Class"].map(classes)

# ============================================================
# 4) Train/test split (80/20 stratified)
# ============================================================

X = df.drop(columns=["Class", "Class_num"]).to_numpy()
y = df["Class_num"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# 5) Feature scaling (fit on train only)
# ============================================================

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# 6) Linear SVM 
# ============================================================

print("\n=== Linear SVM ===")
C = 86
linear_svm_model = LinearSVC(C=C, max_iter=20000, random_state=42)

# Fit on the training data
linear_svm_model.fit(X_train_s, y_train)

# Make predictions on the training and the test set
yhat_tr = linear_svm_model.predict(X_train_s)
yhat_te = linear_svm_model.predict(X_test_s)

best_param = f"SVM (Linear) C={C}"

# Print Results (Confusion Matrix, Classification Report, F1_Scores, Accuracy)
print_results(y_train=y_train, y_test=y_test, yhat_tr=yhat_tr, yhat_te=yhat_te, name="Linear SVM", classes=classes, best_param=best_param)

# ============================================================
# 7) RBF SVM 
# ============================================================

print("\n=== RBF SVM ===")
C = 16
gamma = "scale"
rbf_svm_model = SVC(kernel="rbf", C=C, gamma=gamma)

# Fit on the training data
rbf_svm_model.fit(X_train_s, y_train)

# Make predictions on the training and the test set
yhat_tr = rbf_svm_model.predict(X_train_s)
yhat_te = rbf_svm_model.predict(X_test_s)

best_param = f"SVM (RBF) C={C}, gamma={gamma}"
print_results(y_train=y_train, y_test=y_test, yhat_tr=yhat_tr, yhat_te=yhat_te, name="RBF SVM", classes=classes, best_param=best_param)

# ============================================================
# 8) Polynomial SVM 
# ============================================================

print("\n=== Polynomial SVM ===")
C = 3
degree = 3
gamma = "scale"
coef0 = 1
poly_svm_model = SVC(kernel="poly", C=C, degree=degree, gamma=gamma, coef0=coef0)

# Fit on the training data
poly_svm_model.fit(X_train_s, y_train)

# Make predictions on the training and the test set
yhat_tr = poly_svm_model.predict(X_train_s)
yhat_te = poly_svm_model.predict(X_test_s)

best_param = f"SVM (Poly) C={C}, degree={degree}, gamma={gamma}, coef0={coef0}"
print_results(y_train=y_train, y_test=y_test, yhat_tr=yhat_tr, yhat_te=yhat_te, name="Polynomial SVM", classes=classes, best_param=best_param)