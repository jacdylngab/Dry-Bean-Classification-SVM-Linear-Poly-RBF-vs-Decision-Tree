# This code uses the best hyperparameters found from the decision_tree_scheduler.py script
# It takes in a CSV file and trains a Decision Tree Classifier model based on the best hyperparameters found
#
# The workflow includes:
#   - Loading and transforming the dataset
#   - Splitting the dataset into training and testing sets
#   - Training and evaluating the Decision Tree Classifer model

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

###############################################################
####################### Helper Function  ######################
###############################################################

def report_line(tag, acc, f1m, f1w):
    print(f"{tag:<30} | ACC: {acc:.3f} | F1-macro: {f1m:.3f} | F1-weighted: {f1w:.3f}")

def saving_results(Model="None", Macro_F1_Test=None, Macro_F1_Train=None, Accuracy=None):
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

def displaying_the_confusion_matrix(y_pred, y_test_or_train, model, t):
    cm = confusion_matrix(y_test_or_train, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", ax=plt.gca())
    plt.title(f"Confusion Matrix - Decision Tree ({t})")
    plt.tight_layout()
    saving_name = f"Decision Tree ({t}).png"
    #plt.savefig(saving_name)
    plt.show()

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
############# 3)  Decision Tree Classifier  ###################
###############################################################

# Best parameters found
max_depth = None
max_leaf_nodes = None
min_samples_split = 2
ccp_alpha = 0.00041

decision_tree_clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, ccp_alpha=ccp_alpha, random_state=42)

# Fit on the training data
decision_tree_clf.fit(X_train, y_train)

# Make predictions on the training and the test set
yhat_tr = decision_tree_clf.predict(X_train)
yhat_te = decision_tree_clf.predict(X_test)


print("\n=== TRAIN ===")
print(f"Accuracy: {accuracy_score(y_train, yhat_tr):.3f}")
print(f"F1 (macro): {f1_score(y_train, yhat_tr, average='macro'):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_train, yhat_tr))
displaying_the_confusion_matrix(y_pred=yhat_tr, y_test_or_train=y_train, model=decision_tree_clf, t="TRAIN")

print("\n=== TEST ===")
print(f"Accuracy: {accuracy_score(y_test, yhat_te):.3f}")
print(f"F1 (macro): {f1_score(y_test, yhat_te, average='macro'):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_test, yhat_te))
displaying_the_confusion_matrix(y_pred=yhat_te, y_test_or_train=y_test, model=decision_tree_clf, t="TEST")
print("\nClassification report (TEST):")
print(classification_report(y_test, yhat_te, digits=3))

best_param = f"Decision Tree max_depth={max_depth}, max_leaf_nodes={max_leaf_nodes}, min_samples_split={min_samples_split}, ccp_alpha={ccp_alpha}"
f1m_test = f1_score(y_test, yhat_te, average='macro')
f1m_train = f1_score(y_train, yhat_tr, average='macro')
acc = accuracy_score(y_test, yhat_te)
# Uncomment this if you want the results to be saved to an external CSV file
#saving_results(Model=best_param, Macro_F1_Test=f1m_test, Macro_F1_Train=f1m_train, Accuracy=acc)