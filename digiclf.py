
# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import preprocess_data, split_train_dev_test, train_model, predict_and_eval, report_CM

###############################################################################


# Step 1: Data loading
digits = datasets.load_digits()

# Step 2: Data splitting
data = digits.images

# Defines split sizes
# ratio_train = 0.8
size_dev = 0.1
size_test = 0.1

# Define x & y dataframes
x = data
y = digits.target

# Generate Train, Dev and Test splits.
x_train, x_dev, x_test, y_train, y_dev, y_test = split_train_dev_test(x, y, size_test, size_dev)

# Step 3: Data preprocessing
x_train = preprocess_data(x_train)
x_dev = preprocess_data(x_dev)
x_test = preprocess_data(x_test)

# Step 4: Model training
model = train_model(x_train, y_train, {'gamma': 0.001}, "svm")

# Step 5: Model Prediction & Evaluation using Test dataset
predicted_test = predict_and_eval(model, x_test, y_test)

# Step 6: Report Metrics using Test dataset
report_CM(y_test, predicted_test)






