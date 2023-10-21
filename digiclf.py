
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

###############################################################################

# Removing old model files
extn_paths = [
    {'extn': '*.joblib', 'path': './models/'},
]
delete_files_shell(extn_paths)

# Holds Hparam combinations for all Classifiers 
clf_param_dict = {}

parser = argparse.ArgumentParser(description='Process hyperparameters from JSON file.')
parser.add_argument('json_file', type=str, help='Path to the JSON file containing hyperparameters')

args = parser.parse_args()
json_file_path = args.json_file

# Load hyperparameters from JSON
hyperparameters_data = load_hparams_from_json(json_file_path)

# Extract hyperparameter ranges
svm_hparams = hyperparameters_data['svm']
tree_hparams = hyperparameters_data['tree']

# Create a list of dictionaries for all hparam combinations
svm_param_combinations = get_hparam_combinations(svm_hparams)
tree_param_combinations = get_hparam_combinations(tree_hparams)

clf_param_dict['svm'] = svm_param_combinations
clf_param_dict['tree'] = tree_param_combinations

# No of Runs
no_of_runs = 5

# Define Data split
test_size = [0.2]
dev_size = [0.2]

# Data loading
digits = datasets.load_digits()
data = digits.images

# Define x & y dataframes
x = data
y = digits.target

results_dict_list = []

for run_no in range(no_of_runs):

    run_results = {}
    for size_test in test_size:
        for size_dev in dev_size:

            size_train = 1-(size_test+size_dev)
            # Generate Train, Dev and Test splits.
            x_train, x_dev, x_test, y_train, y_dev, y_test = split_train_dev_test(x, y, size_test, size_dev)

            for clf in clf_param_dict:

                param_combinations = clf_param_dict[clf]
                best_hparams, best_acc_so_far, model_path = tune_hparams(x_train, y_train, x_dev, y_dev, param_combinations, clf, size_train, size_dev, size_test)

                # Load Best model
                best_model = load(model_path)

                train_acc = get_accuracy(best_model, x_train, y_train)
                dev_acc = get_accuracy(best_model, x_dev, y_dev)
                test_acc = get_accuracy(best_model, x_test, y_test)

                print (f"\n\nmodel_type={clf} test_size={size_test} dev_size={size_dev} train_size={size_train} train_acc={train_acc} dev_acc={dev_acc} test_acc={test_acc}")
                
                print ("\nBest Hparam combinations are:")
                print ("Best Hparama: ", best_hparams)
                print ("Best Model: ", best_model)
                print ("Best Accuracy: ", best_acc_so_far)
                print ("Model Path: ", model_path)

                run_results = {'model_type' : clf, 'run_index' : run_no, 'train_acc' : train_acc, 'dev_acc' : dev_acc, 'test_acc' : test_acc}
                results_dict_list.append(run_results)

results_df = pd.DataFrame(results_dict_list)

print(results_df.groupby('model_type').describe().T)

# Quiz 2 ********************************************************

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
print()




