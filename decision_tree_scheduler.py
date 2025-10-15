import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

###############################################################
################ 1) Load the Dry Bean Dataset #################
###############################################################

df = pd.read_csv("Dry_Bean_Dataset.csv")

###############################################################
#################### 2)  Train/test split  ####################
###############################################################

X = df.drop(columns=["Class"]).to_numpy()
y = df["Class"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

###############################################################
#################### 3)  Helper Function  #####################
###############################################################

def report_line(tag, acc, f1m, f1w):
    print(f"{tag:<30} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(criterion=None, max_depth=None, max_leaf_nodes=None, min_samples_split=None, ccp_alpha=None):
    filename = Path("Best_Tree_Hyperparameters_Manual.csv")

    data = {
        "criterion"         : [criterion],
        "max_depth"         : [max_depth],
        "max_leaf_nodes"    : [max_leaf_nodes],
        "min_samples_split" : [min_samples_split],
        "ccp_alpha"         : [ccp_alpha]
    }

    df_new = pd.DataFrame(data)

    if filename.exists():
        df_new.to_csv(filename, index=False, mode='a', header=False)
    else:
        df_new.to_csv(filename, index=False, mode='w', header=True)

###############################################################
########### 4) Parameter Grid & Combos  #######################
###############################################################

param_grid = [
    # Allow: (A) vary max_depth with max_leaf_nodes=None
    {
        "criterion"         : ['gini', 'entropy'],
        #"max_depth"         : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "max_depth"         : [None, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "max_leaf_nodes"    : [None],
        "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
        #"ccp_alpha"         : [0.0, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5]
        #"ccp_alpha"         : [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
        "ccp_alpha"         : [0.0, 0.0004, 0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00046, 0.00047, 0.00047, 0.00048, 0.00049]
    },
    # Allow: (B) Vary max_leaf_nodes with max_depth=None
    {
        "criterion"         : ['gini', 'entropy'],
        "max_depth"         : [None],
        #"max_leaf_nodes"    : [None, 6, 9, 12, 15, 18, 21, 24],
        #"max_leaf_nodes"    : [None, 6, 9, 12, 15, 18, 21, 24, 27, 30, 40, 50, 60, 70, 80, 90, 100],
        "max_leaf_nodes"    : [None, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
        "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
        #"ccp_alpha"         : [0.0, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5]
        #"ccp_alpha"         : [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
        "ccp_alpha"         : [0.0, 0.0004, 0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00046, 0.00047, 0.00047, 0.00048, 0.00049]
    },
    # Baseline: (C) both unlimited, various mss/alpha
    {
        "criterion"         : ['gini', 'entropy'],
        "max_depth"         : [None],
        "max_leaf_nodes"    : [None],
        "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9, 10],
        #"ccp_alpha"         : [0.0, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.5]
        #"ccp_alpha"         : [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
        "ccp_alpha"         : [0.0, 0.0004, 0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00046, 0.00047, 0.00047, 0.00048, 0.00049]
    }
]

###############################################################
############# 5)  K-fold Stratified Cross Validation  #########
###############################################################

# 5-fold cross-validation on the Training set only
#cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

###############################################################
##################### 6)  GridSearchCV  #######################
###############################################################

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1_macro',         # Prefer Macro-F1 for evaluation since the dataset is imbalanced.
    cv=cross_validation,        # cross validation
    n_jobs=-1,                  # use all CPU cores for speed
    verbose=2,                   # show progress
    return_train_score=True
)

# Fit on the training data
grid_search.fit(X_train, y_train)

# Predict on the test set using the best estimator
# The best estimator is the actual trained model (estimator) that achieved the best score on cross-validation for the hyperparameter combination that was specified.
best_clf = grid_search.best_estimator_

yhat_tr = best_clf.predict(X_train)
yhat_te = best_clf.predict(X_test)

'''
# 3) Parameter grid
grid_max_depth       = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grid_max_leaf_nodes  = [None, 6, 9, 12, 15, 18, 21, 24]
grid_min_samples_split = [2, 10, 20, 30, 40, 50]
grid_ccp_alpha       = [0.0, 0.0005, 0.001, 0.01] # light post-pruning

# 4) CV setup (more stable than single 5-fold)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

def combos():
    # Allow: (A) vary max_depth with max_leaf_nodes=None
    for max_depth, mss, alpha in product(grid_max_depth, grid_min_samples_split, grid_ccp_alpha):
        yield {"max_depth": max_depth, "max_leaf_nodes": None,
               "min_samples_split": mss, "ccp_alpha": alpha}
    # Allow: (B) vary max_leaf_nodes with max_depth=None
    for max_leaf_nodes, mss, alpha in product(grid_max_leaf_nodes, grid_min_samples_split, grid_ccp_alpha):
        yield {"max_depth": None, "max_leaf_nodes": max_leaf_nodes,
               "min_samples_split": mss, "ccp_alpha": alpha}
    # Baseline: (C) both unlimited, various mss/alpha
    for mss, alpha in product(grid_min_samples_split, grid_ccp_alpha):
        yield {"max_depth": None, "max_leaf_nodes": None,
               "min_samples_split": mss, "ccp_alpha": alpha}

results = []
print("=== Repeated 5-fold CV across filtered hyperparameter grid ===")
for params in combos():
    clf = DecisionTreeClassifier(random_state=42, **params)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)
    mean_, std_ = scores.mean(), scores.std()
    results.append({"params": params, "mean_acc": mean_, "std_acc": std_})

# Prefer higher acc, then simpler model
def simplicity_key(p):
    # simpler = shallower depth, fewer leaves limit (prefer having just one limiter), larger min_samples_split, larger ccp_alpha
    depth_score = 0 if p["max_depth"] is None else p["max_depth"]      # lower better
    leaf_score  = 0 if p["max_leaf_nodes"] is None else p["max_leaf_nodes"]  # lower better
    # Penalize if both limiters present (we never allow it, but keep just in case)
    both_penalty = 1 if (p["max_depth"] is not None and p["max_leaf_nodes"] is not None) else 0
    return (both_penalty, depth_score, leaf_score, -p["min_samples_split"], -p["ccp_alpha"])

best = max(results, key=lambda r: (r["mean_acc"], -1*(
    # invert simplicity_key lexicographically for tie-break (smaller is simpler)
    -simplicity_key(r["params"])[0],
    -simplicity_key(r["params"])[1],
    -simplicity_key(r["params"])[2],
    -simplicity_key(r["params"])[3],
    -simplicity_key(r["params"])[4]
)))

print("\n=== Best config by CV accuracy ===")
print(f"Params: {best['params']}, CV mean acc={best['mean_acc']:.3f} (std={best['std_acc']:.3f})")

# Retrain best on full TRAIN and evaluate
best_clf = DecisionTreeClassifier(random_state=42, **best["params"])
best_clf.fit(X_train, y_train)

yhat_tr = best_clf.predict(X_train)
yhat_te = best_clf.predict(X_test)
'''

print(f"Best Hyperparameters: criterion={grid_search.best_params_['criterion']}, max_depth={grid_search.best_params_['max_depth']} | max_leaf_nodes={grid_search.best_params_['max_leaf_nodes']} | min_samples_split={grid_search.best_params_['min_samples_split']} | ccp_alpha={grid_search.best_params_["ccp_alpha"]}")
saving_results(criterion=grid_search.best_params_['criterion'], max_depth=grid_search.best_params_['max_depth'], max_leaf_nodes=grid_search.best_params_['max_leaf_nodes'], min_samples_split=grid_search.best_params_['min_samples_split'], ccp_alpha=grid_search.best_params_["ccp_alpha"])

print("\n=== TRAIN ===")
print(f"Accuracy: {accuracy_score(y_train, yhat_tr):.3f}")
print(f"F1 (macro): {f1_score(y_train, yhat_tr, average='macro'):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_train, yhat_tr))

print("\n=== TEST ===")
print(f"Accuracy: {accuracy_score(y_test, yhat_te):.3f}")
print(f"F1 (macro): {f1_score(y_test, yhat_te, average='macro'):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_test, yhat_te))
print("\nClassification report (TEST):")
print(classification_report(y_test, yhat_te, digits=3))