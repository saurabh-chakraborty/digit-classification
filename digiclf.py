
# Import datasets, classifiers and performance metrics
from sklearn import datasets
from utils import preprocess_data, split_train_dev_test, get_accuracy, tune_hparams, get_hparam_combinations
from PIL import Image
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

print("\nThe number of total samples in the dataset (train + test + dev) = ", len(x_train) + len(x_test) + len(x_dev))

for image in data:
    height, width = image.shape
    print("Image Height, Width", height, width)


# Q3 solution

image_sizes = [4, 6, 8]

for size in image_sizes:
    resized_images = []

    

    for image in data:
        # Resize image
        resized_image = image.resize(size,size)
        resized_images.append(resized_image)

    r_X = resized_images

    # Split the dataset into training, dev, and test sets
    X_train, X_test, y_train, y_test = split_train_dev_test(
        r_X, y, test_size=0.2, random_state=42)
    
    X_train, X_dev, y_train, y_dev = split_train_dev_test(
        X_train, y_train, test_size=0.1, random_state=42)
 
    # Evaluate the model's performance on the dev and test sets
    r_train_acc = get_accuracy(y_train, best_model.predict(X_train))
    r_dev_acc = get_accuracy(y_dev, best_model.predict(X_dev))
    r_test_acc = get_accuracy(y_test, best_model.predict(X_test))
    
    # Print the results
    print(f"Image size: {size}x{size}")
    print(f"Train size: {len(X_train)/len(data):.2f}")
    print(f"Dev size: {len(X_dev)/len(data):.2f}")
    print(f"Test size: {len(X_test)/len(data):.2f}")
    print(f"Train accuracy: {r_train_acc:.2f}")
    print(f"Dev accuracy: {r_dev_acc:.2f}")
    print(f"Test accuracy: {r_test_acc:.2f}")
    print()






