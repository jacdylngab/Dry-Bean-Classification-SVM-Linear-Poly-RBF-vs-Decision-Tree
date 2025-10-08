import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split, GridSearchCV
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
class_names = encoder.classes_.tolist()

print(f"numeric labels: {y}")
print(f"numeric labels type: {type(y)}")
print(f"class names: {class_names}")
