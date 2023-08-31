
# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import split_data, preprocess_data, train_model, predict_and_eval, report_CM

###############################################################################


# Step 1: Data loading
digits = datasets.load_digits()

# Step 2: Data splitting
data = digits.images

# Defines split ratios
ratio_train = 0.8
ratio_val = 0.1
ratio_test = 0.1

# Define x & y datarames
x = data
y = digits.target

# Generate t\Test split.
x_remaining, x_test, y_remaining, y_test = split_data(x, y, test_size=ratio_test)

# Define Val ratio
ratio_remaining = 1 - ratio_test
ratio_val_adjusted = ratio_val / ratio_remaining

# Generate Train and Val splits.
x_train, x_val, y_train, y_val = split_data(x_remaining, y_remaining, test_size=ratio_val_adjusted)


# Step 3: Data preprocessing
x_train = preprocess_data(x_train)
x_val = preprocess_data(x_val)
x_test = preprocess_data(x_test)

# Step 4: Model training
model = train_model(x_train, y_train, {'gamma': 0.001}, "svm")

# Step 5: Model Prediction & Evaluation using Validation dataset
predicted_val = predict_and_eval(model, x_val, y_val)

# Step 6: Report Metrics using Testing dataset
predicted_test = predict_and_eval(model, x_test, y_test)
report_CM(y_test, predicted_test)






