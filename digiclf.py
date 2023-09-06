
# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import preprocess_data, split_train_dev_test, get_accuracy, tune_hparams

###############################################################################
# Define hyperparameter ranges
gamma_ranges = [0.0001, 0.001, 0.01, 1, 10]
C_ranges = [0.1, 1, 2, 5, 10]

# Create a list of dictionaries for all hparam combinations
param_combinations = []

for gamma in gamma_ranges:
    for C in C_ranges:
        param_combinations.append({"gamma": gamma, "C": C})

test_size = [0.1, 0.2, 0.3]
dev_size = [0.1, 0.2, 0.3]

# Data loading
digits = datasets.load_digits()
data = digits.images

# Define x & y dataframes
x = data
y = digits.target

for size_test in test_size:
    for size_dev in dev_size:

        # Generate Train, Dev and Test splits.
        x_train, x_dev, x_test, y_train, y_dev, y_test = split_train_dev_test(x, y, size_test, size_dev)

        # Data preprocessing
        x_train = preprocess_data(x_train)
        x_dev = preprocess_data(x_dev)
        x_test = preprocess_data(x_test)

        optimal_gamma, optimal_C, best_model, best_acc_so_far = tune_hparams(x_train, y_train, x_dev, y_dev, param_combinations)
        
        train_acc = get_accuracy(best_model, x_train, y_train)
        dev_acc = get_accuracy(best_model, x_dev, y_dev)
        test_acc = get_accuracy(best_model, x_test, y_test)

        print (f"\n\ntest_size={size_test} dev_size={size_dev} train_size={1-(size_test+size_dev)} train_acc={train_acc} dev_acc={dev_acc} test_acc={test_acc}")
        
        print ("\nBest Hparam combinations are:")
        print ("Optimal Gamma value: ", optimal_gamma)
        print ("Optimal C value: ", optimal_C)
        print ("Best Model: ", best_model)
        print ("Best Accuracy: ", best_acc_so_far)








