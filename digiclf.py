
# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import preprocess_data, split_train_dev_test, get_accuracy, tune_hparams, get_hparam_combinations, train_model
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import load
import pandas as pd
###############################################################################

# Holds Hparam combinations for all Classifiers 
clf_param_dict = {}

# SVM
# Define SVM hyperparameter ranges
gamma_ranges = [0.0001, 0.001, 0.01, 1, 10]
C_ranges = [0.1, 1, 2, 5, 10]
svm_hparams = {}
svm_hparams['gamma'] = gamma_ranges
svm_hparams['C'] = C_ranges

# Create a list of dictionaries for all hparam combinations
svm_param_combinations = get_hparam_combinations(svm_hparams)
clf_param_dict['svm'] = svm_param_combinations

# Decision Tree
# Define Tree hyperparameter ranges
max_depth_ranges = [5, 10, 15, 20, 50, 100]
tree_hparams = {}
tree_hparams['max_depth'] = max_depth_ranges

# Create a list of dictionaries for all hparam combinations
tree_param_combinations = get_hparam_combinations(tree_hparams)
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

