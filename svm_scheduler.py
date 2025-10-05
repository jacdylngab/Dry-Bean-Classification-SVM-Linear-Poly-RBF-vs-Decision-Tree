import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
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

classes = {
    "SEKER"    : 1,
    "BARBUNYA" : 2, 
    "BOMBAY"   : 3, 
    "CALI"     : 4, 
    "HOROZ"    : 5,   
    "SIRA"     : 6,
    "DERMASON" : 7
}

# Add a new column for the numerical Class column
if "Class_num" not in df.columns:
    df["Class_num"] = df["Class"].map(classes)

# Save back to the same CSV
df.to_csv("Dry_Bean_Dataset.csv", index=False)


###############################################################
#################### 3)  Train/test split  ####################
###############################################################

X = df.drop(columns=["Class"]).to_numpy()
y = df["Class_num"].to_numpy()

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

def report_line(tag, acc, f1m, f1w):
    print(f"{tag:<30} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(CV="None", SVM = "None", C=None, degree=None, gamma=None, coeff0=None):
    filename = Path("Best_Hyperparameters.csv")

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
"""
###############################################################
################## 5)  Linear SVM Schedule  ###################
###############################################################
print("\n=== Linear SVM schedule ===")

# Define the linear svm parameter grid
linear_param_grid = {
    'C' : [0.01, 0.1, 0.3, 1, 3, 10, 30, 60, 100],
    'max_iter' : [20000],
    'dual': [False]
}

# Create Linear SVM GridSearchCV object
linear_clf_grid = GridSearchCV(
    LinearSVC(random_state=42),
    linear_param_grid,
    scoring='f1_macro',         # Prefer Macro-F1 for evaluation since the dataset is imbalanced.
    cv=5,                       # 5-fold cross validation
    verbose=1                   # show progress
)

# Fit on the training data
linear_clf_grid.fit(X_train_s, y_train)

# Convert cv_results_ to a DataFrame for easy viewing
#results = pd.DataFrame(linear_clf_grid.cv_results_)

# Show relevant columns
#print(results[['param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']])

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
print("Best F1-Macro score:", linear_clf_grid.best_score_)
print(f"Best Linear by F1-macro: {linear_clf_grid.best_params_['C']}")
saving_results(CV="GridSearchCV", SVM="Linear", C=linear_clf_grid.best_params_['C'])

"""
lin_Cs = [0.01, 0.1, 0.3, 1, 3, 10, 30, 60, 100]
print("\n=== Linear SVM schedule ===")
best_lin = (None, -1.0)
for C in lin_Cs:
    lin_clf = LinearSVC(C=C, max_iter=20000, random_state=42)  # add dual=False if you like
    lin_clf.fit(X_train_s, y_train)
    pred = lin_clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    f1w = f1_score(y_test, pred, average="weighted")
    tag = f"Linear C={C:g}"
    report_line(tag, acc, f1m, f1w)
    if f1m > best_lin[1]:
        best_lin = (tag, f1m)
print(f"Best Linear by F1-macro: {best_lin[0]}")
saving_results(CV="Manually", SVM= "Linear", C=best_lin[0])
"""




###############################################################
################### 6)  RBF SVM Schedule  #####################
###############################################################

print("\n=== RBF SVM schedule ===")
# Define the rbf svm parameter grid
rbf_param_grid = {
    'C' : [0.01, 0.1, 0.3, 1, 3, 10, 30, 60, 100],
    'gamma' : ["scale", 0.001, 0.01, 0.1, 1, 10]
}

# Define the SVM model
rbf_svc = SVC(kernel="rbf")

# Create RBF SVM GridSearchCV object
rbf_clf_grid = GridSearchCV(
    estimator=rbf_svc,
    param_grid=rbf_param_grid,
    scoring='f1_macro',         # Prefer Macro-F1 for evaluation since the dataset is imbalanced.
    cv=5,                       # 5-fold cross validation
    n_jobs=-1,                   # use all CPU cores
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
print("Best F1-Macro score:", rbf_clf_grid.best_score_)
print(f"Best RBF by F1-macro: C={rbf_clf_grid.best_params_['C']}, gamma={rbf_clf_grid.best_params_['gamma']}")
saving_results(CV="GridSearchCV", SVM="RBF", C=rbf_clf_grid.best_params_['C'], gamma=rbf_clf_grid.best_params_['gamma'])

"""
rbf_Cs     = [0.01, 0.1, 0.3, 1, 3, 10, 30, 60, 100]
rbf_gammas =  ["scale", 0.001, 0.01, 0.1, 1, 10]

print("\n=== RBF SVM schedule ===")
best_rbf = (None, -1.0)
for C, gamma in product(rbf_Cs, rbf_gammas):
    clf = SVC(kernel="rbf", C=C, gamma=gamma)
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    f1w = f1_score(y_test, pred, average="weighted")
    tag = f"RBF C={C:g}, γ={gamma}"
    report_line(tag, acc, f1m, f1w)
    if f1m > best_rbf[1]:
        best_rbf = (tag, f1m)
print(f"Best RBF by F1-macro: {best_rbf[0]}")
saving_results(CV="Manually", SVM="RBF", C=best_rbf[0], gamma=best_rbf[0])
"""
###############################################################
################ 7)  Polynomial SVM Schedule  #################
###############################################################

print("\n=== Polynomoal SVM schedule ===")
# Define the poly svm parameter grid
poly_param_grid = {
    'C' : [0.01, 0.1, 0.3, 1, 3, 10, 30, 60, 100],
    'degree' : [2, 3],
    'gamma' :  ["scale", 0.001, 0.01, 0.1, 1, 10],
    'coef0' : [0, 1]
}

# Define the SVM model
poly_svc = SVC(kernel="poly")

# Create RBF SVM GridSearchCV object
poly_clf_grid = GridSearchCV(
    estimator=poly_svc,
    param_grid=poly_param_grid,
    scoring='f1_macro',         # Prefer Macro-F1 for evaluation since the dataset is imbalanced.
    cv=5,                       # 5-fold cross validation
    n_jobs=-1,                   # use all CPU cores
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
print("Best F1-Macro score:", poly_clf_grid.best_score_)
print(f"Best Poly by F1-macro: C={poly_clf_grid.best_params_['C']}, degree={poly_clf_grid.best_params_['degree']}, gamma={poly_clf_grid.best_params_['gamma']}, coef0={poly_clf_grid.best_params_['coef0']}")
saving_results(CV="GridSearchCV", SVM="Poly", C=poly_clf_grid.best_params_['C'], degree=poly_clf_grid.best_params_['degree'], gamma=poly_clf_grid.best_params_['gamma'], coeff0=poly_clf_grid.best_params_['coef0'])

'''
poly_Cs     = [0.01, 0.1, 0.3, 1, 3, 10, 30, 60, 100]
poly_degs   = [2, 3]
poly_gammas =  ["scale", 0.001, 0.01, 0.1, 1, 10]
poly_coef0s = [0, 1]

print("\n=== Polynomial SVM schedule ===")
best_poly = (None, -1.0)
for C, degree, gamma, coef0 in product(poly_Cs, poly_degs, poly_gammas, poly_coef0s):
    clf = SVC(kernel="poly", C=C, degree=degree, gamma=gamma, coef0=coef0)
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    f1w = f1_score(y_test, pred, average="weighted")
    tag = f"Poly C={C:g}, d={degree}, γ={gamma}, c0={coef0}"
    report_line(tag, acc, f1m, f1w)
    if f1m > best_poly[1]:
        best_poly = (tag, f1m)
print(f"Best Poly by F1-macro: {best_poly[0]}")
saving_results(CV="Manually", SVM="Poly", C=best_poly[0], degree=best_poly[0], gamma=best_poly[0], coeff0=best_poly[0])
'''