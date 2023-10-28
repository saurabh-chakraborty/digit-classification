
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm, tree
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from joblib import dump
from itertools import product
import argparse
import json
import subprocess


# Flatten the images
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into train, dev and test subsets
def split_train_dev_test(X, y, test_size, dev_size):

    # Generate Test splits
    x_remaining, x_test, y_remaining, y_test = train_test_split(
        X, y, test_size=test_size, shuffle = True)

    # Calculate Dev size
    size_remaining = 1 - test_size
    dev_size_adjusted = dev_size / size_remaining

    # Generate Train and Dev splits
    x_train, x_dev, y_train, y_dev = train_test_split(
        x_remaining, y_remaining, test_size=dev_size_adjusted, shuffle = True)
    
    # Data preprocessing
    x_train = preprocess_data(x_train)
    x_dev = preprocess_data(x_dev)
    x_test = preprocess_data(x_test)
    
    return x_train, x_dev, x_test, y_train, y_dev, y_test


# Train a model of choice, pass the model parameters
def train_model(X, y, model_params, clf_type):

    if clf_type == "svm":
        # Create SVC classifier with specified model_params
        clf = svm.SVC

    if clf_type == "tree":
        # Create DecisionTree classifier with specified model_params
        clf = tree.DecisionTreeClassifier


    model = clf(**model_params)
    # Training the model
    model.fit(X, y)
    return model


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

# Compute Model Accuracy
def get_accuracy(model, x, y):
    predicted = model.predict(x)
    accuracy = metrics.accuracy_score(y, predicted)
    return round(accuracy, 3)

# Hyper-parameter Tuning & Selection of best Hparams
def tune_hparams(X_train, Y_train, X_dev, y_dev, param_combinations, clf_type, train_size, dev_size, test_size):
    best_acc_so_far = -1

    for param_combination in param_combinations:
            # Train model with cur_gamma & cur_C
            cur_model = train_model(X_train, Y_train, param_combination, clf_type)
            # Get accuracy metric on Dev set
            cur_accuracy = get_accuracy(cur_model, X_dev, y_dev)
            # Select the best Hparams based on accuracy metric using Dev set
            if cur_accuracy > best_acc_so_far:
                best_acc_so_far = cur_accuracy
                best_hparams = param_combination
                best_model = cur_model
    
    model_path = './models/' + clf_type + '_' + 'train_' + str(train_size) + '_' + 'dev_' + str(dev_size) + '_' + 'test_' + str(test_size) + '_' +"_".join(["{}_{}".format(x,y) for x,y in best_hparams.items()]) + ".joblib"
    
    # save the best_model
    dump(best_model, model_path)

    return best_hparams, best_acc_so_far, model_path

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
        subprocess.run(rm_command, shell=True)

        print(f"Successfully removed {extn} files")
