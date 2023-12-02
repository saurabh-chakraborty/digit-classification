
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.svm import SVC
from joblib import dump
from itertools import product
import argparse
import json
import subprocess
from sklearn.metrics import f1_score
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np


# Flatten the images
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))

    # Apply unit normalization using StandardScaler
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)

    return data_normalized

# Split data into train, dev and test subsets
def split_train_dev_test(X, y, test_size, dev_size, seed):

    # Generate Test splits
    x_remaining, x_test, y_remaining, y_test = train_test_split(
        X, y, test_size=test_size, shuffle = True, random_state=seed)
    
    
    # Calculate Dev size
    size_remaining = 1 - test_size
    dev_size_adjusted = dev_size / size_remaining

    # Generate Train and Dev splits
    x_train, x_dev, y_train, y_dev = train_test_split(
        x_remaining, y_remaining, test_size=dev_size_adjusted, shuffle = True, random_state=seed)
    
    # Data preprocessing
    x_train = preprocess_data(x_train)
    x_dev = preprocess_data(x_dev)
    x_test = preprocess_data(x_test)
    
    return x_train, x_dev, x_test, y_train, y_dev, y_test


# Train a model of choice, pass the model parameters
def train_model(X, y, model_params, clf_type, random_state_value, num_threads):

    if clf_type == "svm":
        # Create SVC classifier with specified model_params
        # clf = svm.SVC
        clf = SVC(**model_params)
        # Training the model
        clf.fit(X, y)

    if clf_type == "tree":
        # Create DecisionTree classifier with specified model_params
        # clf = tree.DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=random_state_value, **model_params)
        # Training the model
        clf.fit(X, y)
    
    if clf_type == "lr":
        # Create Logistic Regression classifier with specified model_params
        clf = LogisticRegression(random_state=random_state_value, **model_params)
        # Training the model
        clf.fit(X, y)

    # model = clf(**model_params)
    
    return clf


# Predict & Eval
def predict_and_eval(model, x, y):
    # Predict the value of the digit on the test subset
    predicted = model.predict(x)

    # Below we visualize the first 4 test samples and show their predicted digit value in the title.
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y, predicted)}\n"
    )

    return predicted

# Reporting Metrics
def report_CM(y, predicted):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

# Compute Model Accuracy & F1 score
def get_accuracy(model, x, y):
    predicted = model.predict(x)
    accuracy = metrics.accuracy_score(y, predicted)
    # Calculate macro F1 score
    macro_f1 = f1_score(y, predicted, average='macro')

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, x, y, cv=5, scoring='accuracy')
    # Calculate mean and std of the cross-validation scores
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    return round(accuracy, 3), round(macro_f1, 3), mean_accuracy, std_accuracy

# Hyper-parameter Tuning & Selection of best Hparams
def tune_hparams(X_train, Y_train, X_dev, y_dev, param_combinations, clf_type, train_size, dev_size, test_size, random_state_value, num_threads):
    best_acc_so_far = -1
    best_f1_so_far = 0

    for param_combination in param_combinations:
            # Train model with cur_gamma & cur_C
            cur_model = train_model(X_train, Y_train, param_combination, clf_type, random_state_value, num_threads)
            # Get accuracy metric on Dev set
            cur_accuracy, cur_f1, mean_accuracy, std_accuracy = get_accuracy(cur_model, X_dev, y_dev)
            # Select the best Hparams based on accuracy metric using Dev set
            if cur_accuracy > best_acc_so_far:
                best_acc_so_far = cur_accuracy
                best_f1_so_far = cur_f1
                best_hparams = param_combination
                best_model = cur_model
    
    # model_path = './models/' + 'm22aie239_' + clf_type + '_' + 'train_' + str(train_size) + '_' + 'dev_' + str(dev_size) + '_' + 'test_' + str(test_size) + '_' +"_".join(["{}_{}".format(x,y) for x,y in best_hparams.items()]) + ".joblib"
    # model_path = './models/' + 'm22aie239_' + clf_type + '_' +"_".join(["{}_{}".format(x,y) for x,y in best_hparams.items()]) + ".joblib"
    model_path = './models/' + 'm22aie239_' + clf_type + '_' +"_".join(["{}".format(y) for x,y in best_hparams.items()]) + ".joblib"


    # save the best_model
    dump(best_model, model_path)

    return best_hparams, best_acc_so_far, model_path, best_f1_so_far

# Get Hparam combinations
def get_hparam_combinations(dict_of_param_lists):
    # Generate all combinations of parameter values
    param_combinations = list(product(*dict_of_param_lists.values()))

    # Create dictionaries for each combination
    hyperparameter_combinations = []
    for combination in param_combinations:
        hyperparameters = {}
        for param_name, param_value in zip(dict_of_param_lists.keys(), combination):
            hyperparameters[param_name] = param_value
        hyperparameter_combinations.append(hyperparameters)

    return hyperparameter_combinations

# Load initial data
def load_data():

    # Data loading
    digits = datasets.load_digits()
    data = digits.images

    # Define x & y dataframes
    X = data
    y = digits.target

    return X, y 

# Load data from json file
def load_hparams_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        hyperparameters_data = json.load(json_file)
    return hyperparameters_data

# Delete unwanted files using shell command
def delete_files_shell(folder_extn_dict):

    for item in folder_extn_dict:
        extn = item.get('extn', '*.*')
        folder_path = item.get('path', './')

        # Construct the rm command with the extension and folder path
        rm_command = f'rm {folder_path}{extn}'

        # Run the command using subprocess
        # subprocess.run(rm_command, shell=True)
        subprocess.run(rm_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


        # print(f"Successfully removed {extn} files")

# Function to write results into a text file
def write_results_to_file(clf_name, random_state, accuracy_score_value, f1_score_value, model_path):
    # Create the results folder if it doesn't exist
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)

    # Define the file path with the specified format
    file_name = f"{clf_name}_{random_state}.txt"
    file_path = os.path.join(results_folder, file_name)

    # Write the results to the text file
    with open(file_path, "w") as file:
        file.write(f"accuracy: {accuracy_score_value}\n")
        file.write(f"f1: {f1_score_value}\n")
        file.write(f"model path: {model_path}\n")