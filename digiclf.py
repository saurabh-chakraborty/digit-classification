
# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import preprocess_data, split_train_dev_test, get_accuracy, tune_hparams, get_hparam_combinations, train_model
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
###############################################################################
# Define hyperparameter ranges
gamma_ranges = [0.0001, 0.001, 0.01, 1, 10]
C_ranges = [0.1, 1, 2, 5, 10]

# Create a list of dictionaries for all hparam combinations
param_combinations = []

param_combinations = get_hparam_combinations(gamma_ranges, C_ranges)

test_size = [0.1, 0.2, 0.3]
dev_size = [0.1, 0.2, 0.3]

# Data loading
digits = datasets.load_digits()
data = digits.images

# Define x & y dataframes
x = data
y = digits.target

# for size_test in test_size:
#     for size_dev in dev_size:

#         # Generate Train, Dev and Test splits.
#         x_train, x_dev, x_test, y_train, y_dev, y_test = split_train_dev_test(x, y, size_test, size_dev)

#         # Data preprocessing
#         x_train = preprocess_data(x_train)
#         x_dev = preprocess_data(x_dev)
#         x_test = preprocess_data(x_test)

#         optimal_gamma, optimal_C, best_model, best_acc_so_far = tune_hparams(x_train, y_train, x_dev, y_dev, param_combinations)
        
#         train_acc = get_accuracy(best_model, x_train, y_train)
#         dev_acc = get_accuracy(best_model, x_dev, y_dev)
#         test_acc = get_accuracy(best_model, x_test, y_test)

#         print (f"\n\ntest_size={size_test} dev_size={size_dev} train_size={1-(size_test+size_dev)} train_acc={train_acc} dev_acc={dev_acc} test_acc={test_acc}")
        
#         print ("\nBest Hparam combinations are:")
#         print ("Optimal Gamma value: ", optimal_gamma)
#         print ("Optimal C value: ", optimal_C)
#         print ("Best Model: ", best_model)
#         print ("Best Accuracy: ", best_acc_so_far)

# Q1 solution
# Split the dataset into training, dev, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)

print("\nThe number of total samples in the dataset (train + test + dev) = ", len(X_train) + len(X_test) + len(X_dev))

# Q2 solution
for image in data:
    height, width = image.shape
    print(f"\nImage Height = {height}  Width = {width}")
    break


# Q3 solution

image_sizes = [4, 6, 8]
# Example: Create an SVM classifier with an RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')

for size in image_sizes:
    resized_images = []
    new_shape = (size, size)

    # Resize image
    for image in data:
        
        # Convert the NumPy array to a Pillow Image object
        image_pillow = Image.fromarray(image)

        # Resize the image while maintaining the aspect ratio
        image_pillow.thumbnail(new_shape)

        # Convert the resized Pillow Image back to a NumPy array
        resized_image = np.array(image_pillow)
        resized_images.append(resized_image)

    # Convert the resized images to a NumPy array
    r_X = np.array([np.array(image) for image in resized_images])

    # Split the dataset into training, dev, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        r_X, y, test_size=0.20, random_state=42)
    
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.12, random_state=42)
    
    # Data preprocessing
    x_train = preprocess_data(X_train)
    x_dev = preprocess_data(X_dev)
    x_test = preprocess_data(X_test)

    # Train model
    model.fit(x_train, y_train)

    # predictions
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    dev_pred = model.predict(x_dev)
 
    # Evaluate the model's performance on the dev and test sets
    r_train_acc = accuracy_score(y_train, train_pred)
    r_dev_acc = accuracy_score(y_dev, dev_pred)
    r_test_acc = accuracy_score(y_test, test_pred)
    
    # Print the results
    # image size: 4x4 train_size: 0.7 dev_size: 0.1 test_size: 0.2 train_acc: xx.x dev_acc: xx.x test_acc: xx.x
    print(f"\nImage size: {size}x{size} Train size: {len(X_train)/len(data):.2f} Dev size: {len(X_dev)/len(data):.2f} Test size: {len(X_test)/len(data):.2f} Train accuracy: {r_train_acc:.2f} Dev accuracy: {r_dev_acc:.2f} Test accuracy: {r_test_acc:.2f}")









