import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

###############################################################
####################### Helper Functions  #####################
###############################################################

def report_line(tag, acc, f1m, f1w):
    print(f"{tag} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(Model="None", Macro_F1=None, Accuracy=None):
    filename = Path("Results_labelEncoder.csv")

    data = {
        "Model" : [Model],
        "Macro_F1 (Test)" : [Macro_F1],
        "Accuracy (Test)" : [Accuracy]
    }

    df_new = pd.DataFrame(data)

    if filename.exists():
        df_new.to_csv(filename, index=False, mode='a', header=False)
    else:
        df_new.to_csv(filename, index=False, mode='w', header=True)

###############################################################
################ 1) Load the Dry Bean Dataset #################
###############################################################

df = pd.read_csv("Dry_Bean_Dataset.csv")

###############################################################
################## 2) Transform the dataset ###################
###############################################################

# Transform the categorical class column into numerical digits
encoder = LabelEncoder()
y = encoder.fit_transform(df["Class"])

###############################################################
#################### 3)  Train/test split  ####################
###############################################################

X = df.drop(columns=["Class"]).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

###############################################################
############## 4)  Scale (fit on train only)  #################
###############################################################

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

###############################################################
######################## 5)  Linear SVM  ######################
###############################################################

print("\n=== Linear SVM ===")
# Create a Linear SVM classifier
#C = 30
#C = 10
#C = 60
C = 20000000000
linear_svm_model = LinearSVC(C=C, max_iter=20000, random_state=42)

# Fit on the training data
linear_svm_model.fit(X_train_s, y_train)

# Make predictions on the test set
y_pred = linear_svm_model.predict(X_test_s)

# Evaluate on the test set
print("Confusion Matrix (TEST):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (TEST):")
print(classification_report(y_test, y_pred, digits=3))

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
f1w = f1_score(y_test, y_pred, average="weighted")
best_param = f"SVM (Linear) C={C}"
report_line(tag=best_param, acc=acc, f1m=f1m, f1w=f1w)
saving_results(Model=best_param, Macro_F1=f1m, Accuracy=acc)

###############################################################
##################### 6)  RBF SVM  ############################
###############################################################

print("\n=== RBF SVM ===")
# Define the SVM model
#C = 30
C = 4
gamma = "scale"
rbf_svm_model = SVC(kernel="rbf", C=C, gamma=gamma)
#rbf_svm_model = SVC(kernel="rbf", C=60, gamma=0.01)
#rbf_svm_model = SVC(kernel="rbf", C=10, gamma=0.01)
#rbf_svm_model = SVC(kernel="rbf", C=10, gamma="scale")
#rbf_svm_model = SVC(kernel="rbf", C=30, gamma="scale")

# Fit on the training data
rbf_svm_model.fit(X_train_s, y_train)

# Make predictions on the test set
y_pred = rbf_svm_model.predict(X_test_s)

# Evaluate on the test set
print("Confusion Matrix (TEST):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (TEST):")
print(classification_report(y_test, y_pred, digits=3))

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
f1w = f1_score(y_test, y_pred, average="weighted")
best_param = f"SVM (RBF) C={C}, gamma={gamma}"
report_line(tag=best_param, acc=acc, f1m=f1m, f1w=f1w)
saving_results(Model=best_param, Macro_F1=f1m, Accuracy=acc)

###############################################################
################ 7)  Polynomial SVM Schedule  #################
###############################################################

print("\n=== Polynomial SVM ===")
# Define the SVM model
C = 40
degree = 2
gamma = "scale"
coef0 = 1
poly_svm_model = SVC(kernel="poly", C=C, degree=degree, gamma=gamma, coef0=coef0)
#C = 3
#poly_svm_model = SVC(kernel="poly", C=3, degree=2, gamma="scale", coef0=1)
#C = 1
#poly_svm_model = SVC(kernel="poly", C=C, degree=2, gamma=0.1, coef0=1)
#C = 1
#poly_svm_model = SVC(kernel="poly", C=C, degree=3, gamma="scale", coef0=1)
#C = 3
#poly_svm_model = SVC(kernel="poly", C=C, degree=3, gamma="scale", coef0=1)

# Fit on the training data
poly_svm_model.fit(X_train_s, y_train)

# Make predictions on the test set
y_pred = poly_svm_model.predict(X_test_s)

# Evaluate on the test set
print("Confusion Matrix (TEST):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (TEST):")
print(classification_report(y_test, y_pred, digits=3))

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
f1w = f1_score(y_test, y_pred, average="weighted")
best_param = f"SVM (Poly) C={C}, degree={degree}, gamma={gamma}, coef0={coef0}"
report_line(tag=best_param, acc=acc, f1m=f1m, f1w=f1w)
saving_results(Model=best_param, Macro_F1=f1m, Accuracy=acc)