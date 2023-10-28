# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import preprocess_data, split_train_dev_test, get_accuracy, tune_hparams, get_hparam_combinations, train_model, load_hparams_from_json, delete_files_shell
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from joblib import load
import pandas as pd
import argparse
import json
from sklearn.tree import DecisionTreeClassifier


# Load the MNIST dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train SVM
production_model = SVC(kernel='linear')
production_model.fit(X_train, y_train)
production_predictions = production_model.predict(X_test)
production_accuracy = accuracy_score(y_test, production_predictions)

# Train Decision Tree
candidate_model = DecisionTreeClassifier()
candidate_model.fit(X_train, y_train)
candidate_predictions = candidate_model.predict(X_test)
candidate_accuracy = accuracy_score(y_test, candidate_predictions)

# Confusion matrix between production and candidate predictions
confusion_matrix_all = confusion_matrix(production_predictions, candidate_predictions)

# Calculate macro-average F1 score
f1_macro = f1_score(y_test, candidate_predictions, average='macro')

print("\nQuiz 2 Results\n")
print("Production Model's Accuracy:", production_accuracy)
print("Candidate Model's Accuracy:", candidate_accuracy)
print("\nConfusion Matrix (Production vs Candidate):")
print(confusion_matrix_all)
print("\nMacro-average F1 Score:", f1_macro)

# Find indices where production model's predictions are correct but candidate model's predictions are wrong
correct_in_production_not_in_candidate = ((production_predictions == y_test) & (candidate_predictions != y_test))

# Create a 2x2 confusion matrix
true_positive = sum(correct_in_production_not_in_candidate & (y_test == production_predictions))
false_negative = sum(~correct_in_production_not_in_candidate & (y_test == production_predictions))
false_positive = sum(correct_in_production_not_in_candidate & (y_test != production_predictions))
true_negative = sum(~correct_in_production_not_in_candidate & (y_test != production_predictions))

confusion_matrix_subset = [[true_positive, false_negative],
                           [false_positive, true_negative]]

print("\nConfusion Matrix (Samples Correct in Production but Not in Candidate):")
print(confusion_matrix_subset)
