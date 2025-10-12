import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score

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
#################### 5)  Helper Function  #####################
###############################################################

def report_line(tag, acc, f1m, f1w):
    print(f"{tag:<30} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(CV="None", SVM = "None", C=None, degree=None, gamma=None, coeff0=None):
    filename = Path("Best_SVM_Hyperparameters_LabelEncoder.csv")

    data = {
        "Cross Validation" : [CV],
        "SVM" : [SVM],
        "C" : [C],
        "degree" : [degree],
        "gamma" : [gamma],
        "coef0" : [coeff0] 
    }

    df_new = pd.DataFrame(data)

    if filename.exists():
        df_new.to_csv(filename, index=False, mode='a', header=False)
    else:
        df_new.to_csv(filename, index=False, mode='w', header=True)

"""
I learned about GridSearchCV
GriSearchCV is pretty much like choosing hyperparameters like C, alpha, or Gamma manually through a for loop
GridSearchCV does the hyperparameter tuning automatically. First this easier, also GridSearchCV does "K-fold-cross-validation" internally
K-fold cross-validation = split your data into k folds, train k times leaving out each fold once,
and average the performance → gives a more reliable measure than a single train/test split.
For example:
If k = 5
The dataset is split into 5 equal folds (chunks).
Each iteration uses 4 folds for training and 1 fold for testing/validation.
You repeat this 5 times, each time leaving out a different fold as the test set.
So it looks like this:

Iteration	Training folds	        Test fold
1	        Fold 2, 3, 4, 5	        Fold 1
2	        Fold 1, 3, 4, 5	        Fold 2
3	        Fold 1, 2, 4, 5	        Fold 3
4	        Fold 1, 2, 3, 5	        Fold 4
5	        Fold 1, 2, 3, 4	        Fold 5

Each fold gets used once as test, and 4 times as part of training.
GridSearchCV is also better than a for loop because it gives you the ability to do parallelization (n_jobs=-1) i.e use all your cores where as with a for loop you only use one core.
This helps speed up the hyperparameter tuning.
"""

###############################################################
############# 6)  K-fold Stratified Cross Validation  #########
###############################################################

# k-fold cross-validation on the Training set only
#cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
###############################################################
################## 7)  Linear SVM Schedule  ###################
###############################################################
print("\n=== Linear SVM schedule ===")

# Define the linear svm parameter grid
linear_param_grid = {
    #'C' : [0.01, 0.1, 0.3, 1, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'C' : [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 20000000000],
    'max_iter' : [20000],
    'dual': [False]
}


# Create Linear SVM GridSearchCV object
linear_clf_grid = GridSearchCV(
    LinearSVC(random_state=42),
    linear_param_grid,
    scoring='f1_macro',         # Prefer Macro-F1 for evaluation since the dataset is imbalanced.
    cv=cross_validation,        # cross validation
    verbose=1                   # show progress
)

# Fit on the training data
linear_clf_grid.fit(X_train_s, y_train)

# Predict on the test set using the best estimator
# The best estimator is the actual trained model (estimator) that achieved the best score on cross-validation for the hyperparameter combination that was specified.
best_lin_ctf = linear_clf_grid.best_estimator_
pred = best_lin_ctf.predict(X_test_s)

# Evaluate on the test set
acc = accuracy_score(y_test, pred)
f1m = f1_score(y_test, pred, average="macro")
f1w = f1_score(y_test, pred, average="weighted")
tag = "Linear SVM (GridSearchCV best)"
report_line(tag, acc, f1m, f1w)
# Best parameters and score
print("Mean cross-validated score of the best_estimator:", linear_clf_grid.best_score_)
print(f"Best Linear params: {linear_clf_grid.best_params_['C']}")
saving_results(CV="GridSearchCV", SVM="Linear", C=linear_clf_grid.best_params_['C'])

###############################################################
################### 8)  RBF SVM Schedule  #####################
###############################################################

print("\n=== RBF SVM schedule ===")
# Define the rbf svm parameter grid
rbf_param_grid = {
    #'C' : [0.01, 0.1, 0.3, 1, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'C' : [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5],
    'gamma' : ["scale"]
}

# Define the SVM model
rbf_svc = SVC(kernel="rbf")

# Create RBF SVM GridSearchCV object
rbf_clf_grid = GridSearchCV(
    estimator=rbf_svc,
    param_grid=rbf_param_grid,
    scoring='f1_macro',         # Prefer Macro-F1 for evaluation since the dataset is imbalanced.
    cv=cross_validation,        # cross validation
    n_jobs=-1,                  # use all CPU cores
    verbose=2                   # show progress
)

# Fit on the training data
rbf_clf_grid.fit(X_train_s, y_train)

# Predict on the test set using the best estimator
# The best estimator is the actual trained model (estimator) that achieved the best score on cross-validation for the hyperparameter combination that was specified.
best_rbf_ctf = rbf_clf_grid.best_estimator_
pred = best_rbf_ctf.predict(X_test_s)

# Evaluate on the test set
acc = accuracy_score(y_test, pred)
f1m = f1_score(y_test, pred, average="macro")
f1w = f1_score(y_test, pred, average="weighted")
tag = "RBF SVM (GridSearchCV best)"
report_line(tag, acc, f1m, f1w)
# Best parameters and score
print("Mean cross-validated score of the best_estimator:", rbf_clf_grid.best_score_)
print(f"Best RBF params: C={rbf_clf_grid.best_params_['C']}, gamma={rbf_clf_grid.best_params_['gamma']}")
saving_results(CV="GridSearchCV", SVM="RBF", C=rbf_clf_grid.best_params_['C'], gamma=rbf_clf_grid.best_params_['gamma'])

###############################################################
################ 9)  Polynomial SVM Schedule  #################
###############################################################

print("\n=== Polynomoal SVM schedule ===")
# Define the poly svm parameter grid
poly_param_grid = {
    #'C' : [0.01, 0.1, 0.3, 1, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'C' : [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    'degree' : [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #'gamma' :  ["scale", 0.001, 0.01, 0.1, 1, 10],
    'gamma' :  ["scale"],
    'coef0' : [0, 1]
}

# Define the SVM model
poly_svc = SVC(kernel="poly")

# Create RBF SVM GridSearchCV object
poly_clf_grid = GridSearchCV(
    estimator=poly_svc,
    param_grid=poly_param_grid,
    scoring='f1_macro',         # Prefer Macro-F1 for evaluation since the dataset is imbalanced.
    cv=cross_validation,        # cross validation
    n_jobs=-1,                  # use all CPU cores
    verbose=2                   # show progress
)

# Fit on the training data
poly_clf_grid.fit(X_train_s, y_train)

# Predict on the test set using the best estimator
# The best estimator is the actual trained model (estimator) that achieved the best score on cross-validation for the hyperparameter combination that was specified.
best_poly_ctf = poly_clf_grid.best_estimator_
pred = best_poly_ctf.predict(X_test_s)

# Evaluate on the test set
acc = accuracy_score(y_test, pred)
f1m = f1_score(y_test, pred, average="macro")
f1w = f1_score(y_test, pred, average="weighted")
tag = "Poly SVM (GridSearchCV best)"
report_line(tag, acc, f1m, f1w)
# Best parameters and score
print("Mean cross-validated score of the best_estimator:", poly_clf_grid.best_score_)
print(f"Best Poly params: C={poly_clf_grid.best_params_['C']}, degree={poly_clf_grid.best_params_['degree']}, gamma={poly_clf_grid.best_params_['gamma']}, coef0={poly_clf_grid.best_params_['coef0']}")
saving_results(CV="GridSearchCV", SVM="Poly", C=poly_clf_grid.best_params_['C'], degree=poly_clf_grid.best_params_['degree'], gamma=poly_clf_grid.best_params_['gamma'], coeff0=poly_clf_grid.best_params_['coef0'])