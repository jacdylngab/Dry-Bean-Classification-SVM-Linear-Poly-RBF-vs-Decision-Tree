# The goal of this script is to find the best hyperparameters for Linear, Polynomial, and RBF SVM models
# It takes in a CSV file, applies necessary data prepocessing and transformations,
# and uses GridSearchCV and RepeatedStratifiedKFold to find the best hyperparameters of the Linear, Polynomial, and RBF SVM models
#
# The workflow includes:
#   - Loading and transforming the dataset
#   - Splitting the dataset into training and testing sets
#   - Scaling the features for SVM models using the training set only
#   - Performing GridSearchCV and RepeatedStratifiedKFold to find the best hyperparameters
#
# I was saving results on CSV files for better visualization and understanding
# I have the line commented out. Uncomment it if you want the results in a CSV file

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
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

df["Class_num"] = df["Class"].map(classes)


###############################################################
#################### 3)  Train/test split  ####################
###############################################################

X = df.drop(columns=["Class", "Class_num"]).to_numpy()
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

###############################################################
#################### 5)  Helper Function  #####################
###############################################################

def report_line(tag, acc, f1m, f1w):
    print(f"{tag:<30} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(CV="None", SVM = "None", C=None, degree=None, gamma=None, coeff0=None):
    filename = Path("Best_SVM_Hyperparameters.csv")

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

def print_evaluations(y_test, pred, tag):
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    f1w = f1_score(y_test, pred, average="weighted")
    report_line(tag, acc, f1m, f1w)

###############################################################
############# 6)  K-fold Stratified Cross Validation  #########
###############################################################

# k-fold cross-validation on the Training set only
cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

###############################################################
################## 7)  Linear SVM Schedule  ###################
###############################################################

print("\n=== Linear SVM schedule ===")
# Define the linear svm parameter grid
linear_param_grid = {
    # This is the first range of C's I tried
    #'C' : [0.01, 0.1, 0.3, 1, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    # This is the second range of C's I tried
    #'C'  :[80, 80.5, 81, 81.5, 82, 82.5, 83, 83.5, 84, 84.5, 85, 85.5, 86, 86.5, 87, 87.5, 88, 88.5, 89, 89.5],
    # This was my final range of C's I tried.
    'C'  :[86, 86.1, 86.2, 86.3, 86.4, 86.5, 86.7, 86.8, 86.9],
    'max_iter' : [20000],
    'dual': [False]
}


# Create Linear SVM GridSearchCV object
linear_clf_grid = GridSearchCV(
    estimator=LinearSVC(random_state=42),
    param_grid=linear_param_grid,
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
tag = "Linear SVM (GridSearchCV best)"
print_evaluations(y_test=y_test, pred=pred, tag=tag)

# Best parameters and score
print("Mean cross-validated score of the best_estimator:", linear_clf_grid.best_score_)
print(f"Best Linear params: {linear_clf_grid.best_params_['C']}")
#saving_results(CV="GridSearchCV", SVM="Linear", C=linear_clf_grid.best_params_['C'])

###############################################################
################### 8)  RBF SVM Schedule  #####################
###############################################################

print("\n=== RBF SVM schedule ===")
# Define the rbf svm parameter grid
rbf_param_grid = {
    # This is the first range of C's I tried
    #'C' : [0.01, 0.1, 0.3, 1, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    # This is the second range of C's I tried
    #'C' : [10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5],
    # This was my final range of C's I tried.
    'C' : [16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9],
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
tag = "RBF SVM (GridSearchCV best)"
print_evaluations(y_test=y_test, pred=pred, tag=tag)

# Best parameters and score
print("Mean cross-validated score of the best_estimator:", rbf_clf_grid.best_score_)
print(f"Best RBF params: C={rbf_clf_grid.best_params_['C']}, gamma={rbf_clf_grid.best_params_['gamma']}")
#saving_results(CV="GridSearchCV", SVM="RBF", C=rbf_clf_grid.best_params_['C'], gamma=rbf_clf_grid.best_params_['gamma'])

###############################################################
################ 9)  Polynomial SVM Schedule  #################
###############################################################

print("\n=== Polynomoal SVM schedule ===")
# Define the poly svm parameter grid
poly_param_grid = {
    # This is the first range of C's I tried
    #'C' : [0.01, 0.1, 0.3, 1, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    # This is the second range of C's I tried
    #'C' : [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5],
    # This was my final range of C's I tried.
    'C' : [3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9],
    'degree' : [2, 3, 4, 5, 6, 7, 8, 9, 10],
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
tag = "Poly SVM (GridSearchCV best)"
print_evaluations(y_test=y_test, pred=pred, tag=tag)

# Best parameters and score
print("Mean cross-validated score of the best_estimator:", poly_clf_grid.best_score_)
print(f"Best Poly params: C={poly_clf_grid.best_params_['C']}, degree={poly_clf_grid.best_params_['degree']}, gamma={poly_clf_grid.best_params_['gamma']}, coef0={poly_clf_grid.best_params_['coef0']}")
#saving_results(CV="GridSearchCV", SVM="Poly", C=poly_clf_grid.best_params_['C'], degree=poly_clf_grid.best_params_['degree'], gamma=poly_clf_grid.best_params_['gamma'], coeff0=poly_clf_grid.best_params_['coef0'])