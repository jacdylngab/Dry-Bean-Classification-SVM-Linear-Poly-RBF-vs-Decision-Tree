Project 2 README:

The project includes scripts for both hyperparameter tuning (schedulers) and final evaluation using the best parameters found.

Description of the files:
svm_scheduler.py:
    Performs grid search over multiple SVM parameters (kernel, C, gamma, etc.) to find the best hyperparameters. Takes a long time to run.
decision_tree_scheduler.py:
    Performs grid search over Decision Tree parameters (max_depth, ccp_alpha, etc.) to find the best hyperparameters. Also time-intensive.
svm.py:
    Uses the best SVM hyperparameters found to train and evaluate the SVM models quickly. Outputs accuracy, F1 scores, classification report, and confusion matrices.
decision_tree.py:
    Uses the best Decision Tree hyperparameters found to train and evaluate the model. Outputs accuracy, F1 scores, classification report, and confusion matrices.
f1_macro_bar_chart.py
    Plots the F1 macro test and F1 macro train bar charts

Required libraries:
    - seaborn
    - pandas
    - scikit-learn
    - matplotlib

How to Run
1. Download the Dry Bean Dataset
    Place the Dry_Bean_Dataset.xlsx in the same directory as the scripts (src/)

2. Run schedulers (optional, slow):
    python3 svm_scheduler.py
    python3 decision_tree_scheduler.py

3. Run final models (recommended, fast)
    python3 svm.py
    python3 decision_tree.py
