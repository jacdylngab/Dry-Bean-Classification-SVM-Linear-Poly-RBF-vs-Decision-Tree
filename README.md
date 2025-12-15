# Dry Bean Classification: SVM (Linear/Poly/RBF) vs Decision Tree

## 📌 Project Description
This project uses the **Dry Bean dataset** to compare three Support Vector Machine (SVM) kernels—**Linear**, **Polynomial**, and **RBF**—against a **Decision Tree classifier**. The goal is to build a reproducible experiment, tune hyperparameters, evaluate each model, and present concise results.  

The purpose of this project is to learn to:
- Prepare and scale data appropriately for different models
- Tune hyperparameters using cross-validation
- Evaluate models using metrics like **Macro-F1**, **Accuracy**, **Precision**, and **Recall**
- Visualize performance using confusion matrices and bar charts

---

## 🗂 Project Structure

The project includes scripts for both hyperparameter tuning (schedulers) and final evaluation using the best parameters found.

| File | Description |
|------|-------------|
|`svm_scheduler.py` | Performs grid search over multiple SVM parameters (kernel, C, gamma, etc.) to find the best hyperparameters. Takes a long time to run. |
|`decision_tree_scheduler.py` | Performs grid search over Decision Tree parameters (max_depth, ccp_alpha, etc.) to find the best hyperparameters. Also time-intensive. |
|`svm.py` | Uses the best SVM hyperparameters found to train and evaluate the SVM models quickly. Outputs accuracy, F1 scores, classification report, and confusion matrices. |
|`decision_tree.py` | Uses the best Decision Tree hyperparameters found to train and evaluate the model. Outputs accuracy, F1 scores, classification report, and confusion matrices. |
|`f1_macro_bar_chart.py` | Plots the F1 macro test and F1 macro train bar charts |

---
## ▶️ How to Run

How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the Dry Bean Dataset
```
Place the Dry_Bean_Dataset.xlsx in the same directory as the scripts
```

3. Run schedulers (optional, slow):
```bash
python3 svm_scheduler.py
```
```bash
python3 decision_tree_scheduler.py
```

3. Run final models (recommended, fast)
```bash
python3 svm.py
```
```bash
python3 decision_tree.py
```
