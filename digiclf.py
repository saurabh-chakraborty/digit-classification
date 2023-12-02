
# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import preprocess_data, split_train_dev_test, get_accuracy, tune_hparams, get_hparam_combinations, train_model, load_hparams_from_json, delete_files_shell, write_results_to_file
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

# # Removing old model files
# extn_paths = [
#     {'extn': '*.joblib', 'path': './models/'},
# ]
# delete_files_shell(extn_paths)

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
logistic_hparams = hyperparameters_data['lr']

# Create a list of dictionaries for all hparam combinations
svm_param_combinations = get_hparam_combinations(svm_hparams)
tree_param_combinations = get_hparam_combinations(tree_hparams)
logistic_param_combinations = get_hparam_combinations(logistic_hparams)

clf_param_dict['svm'] = svm_param_combinations
clf_param_dict['tree'] = tree_param_combinations
clf_param_dict['lr'] = logistic_param_combinations

random_state = 42
num_threads = 1


# No of Experiemnt Runs per combination fo hyper-params
no_of_runs = 2

# Define Data split, canb be a list to iterate
test_size = [0.1, 0.2]
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
            x_train, x_dev, x_test, y_train, y_dev, y_test = split_train_dev_test(x, y, size_test, size_dev, random_state)

            for clf in clf_param_dict:

                param_combinations = clf_param_dict[clf]
                best_hparams, best_acc_so_far, model_path, best_f1_so_far = tune_hparams(x_train, y_train, x_dev, y_dev, param_combinations, clf, size_train, size_dev, size_test, random_state, num_threads)

                # Load Best model
                best_model = load(model_path)

                train_acc, train_f1, train_mean_accuracy, train_std_accuracy = get_accuracy(best_model, x_train, y_train)
                dev_acc, dev_f1, dev_mean_accuracy, dev_std_accuracy = get_accuracy(best_model, x_dev, y_dev)
                test_acc, test_f1, test_mean_accuracy, test_std_accuracy = get_accuracy(best_model, x_test, y_test)

                print (f"\n\nmodel_type={clf} test_size={size_test} dev_size={size_dev} train_size={size_train} train_acc={train_acc} dev_acc={dev_acc} test_acc={test_acc}")
                
                print ("\nHparam combinations are:")
                print ("Hparama: ", best_hparams)
                print ("Model: ", best_model)
                print ("Accuracy: ", best_acc_so_far)
                print ("F1: ", best_f1_so_far)
                print ("Mean (5 CV): ", train_mean_accuracy)
                print ("Std (5 CV): ", train_std_accuracy)
                print ("Model Path: ", model_path)

                run_results = {'model_type' : clf, 'run_index' : run_no, 'train_acc' : train_acc, 'dev_acc' : dev_acc, 'test_acc' : test_acc, 'model_path' : model_path}
                results_dict_list.append(run_results)


results_df = pd.DataFrame(results_dict_list)

best_test_acc = 0
best_model_path = ''
for modl in results_dict_list:
    
    if modl['test_acc'] > best_test_acc:
        best_test_acc = modl['test_acc']
        best_model_path = modl['model_path']

print("\nBest Model Path =", best_model_path)
print("\nBest Model Test Accuracy =", best_test_acc)

print()
print(results_df.groupby('model_type').describe().T)






