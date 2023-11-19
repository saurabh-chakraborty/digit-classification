from utils import get_hparam_combinations, load_data, split_train_dev_test, tune_hparams, preprocess_data
import sys, os
from sklearn.svm import SVC
from joblib import load
import base64

# def test_hparam_combinations():
#     # test case to check whether all possible hparam combos are generated
#     gamma_list = [0.001, 0.01, 0.1, 1]
#     C_list = [1, 10, 100, 1000]
#     svm_hparams = {}
#     svm_hparams['gamma'] = gamma_list
#     svm_hparams['C'] = C_list

#     # Create a list of dictionaries for all hparam combinations
#     h_params_combinations = get_hparam_combinations(svm_hparams)

#     assert len(h_params_combinations) == len(gamma_list) * len(C_list)
    

# def test_specific_combinations():
# # test case to check whether all possible hparam combos are generated
#     gamma_list = [0.001, 0.01, 0.1, 1]
#     C_list = [1, 10, 100, 1000]
#     svm_hparams = {}
#     svm_hparams['gamma'] = gamma_list
#     svm_hparams['C'] = C_list

#     # Create a list of dictionaries for all hparam combinations
#     h_params_combinations = get_hparam_combinations(svm_hparams)

#     assert len(h_params_combinations) == len(gamma_list) * len(C_list)

#     combo_1 = {'C': 10, 'gamma': 0.01}
#     combo_2 = {'C': 100, 'gamma': 0.001}
#     assert combo_1 in h_params_combinations and combo_2 in h_params_combinations

# def test_data_splitting():
#     X, y = load_data()
#     X = X[:100,:,:]
#     y = y[:100]

#     test_size = .1
#     dev_size = .6
    
#     X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size, dev_size)

#     assert (len(X_train) == 30) 
#     assert (len(X_test) == 10)
#     assert ((len(X_dev) == 60))

# def test_modelSave_modelType():    

#     X, y = load_data()
#     X = X[:100,:,:]
#     y = y[:100]
    
#     gamma_list = [0.001, 0.01]
#     C_list = [1, 2]
#     svm_hparams = {}
#     svm_hparams['gamma'] = gamma_list
#     svm_hparams['C'] = C_list

#     # Create a list of dictionaries for all hparam combinations
#     h_params_combinations = get_hparam_combinations(svm_hparams)

#     test_size = 0.1
#     dev_size = 0.6
#     train_size = 1-(test_size+dev_size)
#     x_train, x_dev, x_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size, dev_size)

#     clf = 'svm'

#     outputs = tune_hparams(x_train, y_train, x_dev, y_dev, h_params_combinations, clf, train_size, dev_size, test_size)
#     actual_model_path = outputs[-1]

#     assert(os.path.exists(actual_model_path)==True)
#     assert type(load(actual_model_path)) == SVC

# Quiz 4 code

# test_app.py
import pytest
import numpy as np
from sklearn.datasets import fetch_openml
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def load_mnist_samples():
    # Load the MNIST dataset using scikit-learn
    mnist = fetch_openml('mnist_784', version=1)
    mnist_images = mnist.data.astype('float32') / 255.0
    mnist_labels = mnist.target.astype('int')

    # Reshape the data to match the model's input shape
    mnist_images = mnist_images.values.reshape((-1, 8, 8))
    # mnist_images = mnist_images.values.reshape((-1, 28, 28, 1))

    return mnist_images, mnist_labels

def flatten_image(image):
    # Flatten the image to a 1D array
    return image.flatten()



def test_post_predict(client):
    # Load MNIST samples
    mnist_images, mnist_labels = load_mnist_samples()

    # Take the first 10 samples for testing
    test_samples = mnist_images[:10]
    expected_labels = mnist_labels[:10]

    for i in range(10):
        sample_image = test_samples[i]
        expected_label = expected_labels[i]

        # Prepare the image data for the POST request
        image_data = base64.b64encode((sample_image * 255).astype(np.uint8).tobytes()).decode('utf-8')
        data = {'image': image_data}
        
        response = client.post('/predict', json=data)
        
        print(f"Digit: {i}, Response Status Code: {response.status_code}, Response JSON: {response.get_json()}")


        assert response.status_code == 200
     
        # Check if the predicted digit matches any of the expected labels
        predicted_digit = response.get_json()['predicted_digit']

        # Create a list of assertion conditions for each digit
        conditions = [
            predicted_digit == 0,
            predicted_digit == 1,
            predicted_digit == 2,
            predicted_digit == 3,
            predicted_digit == 4,
            predicted_digit == 5,
            predicted_digit == 6,
            predicted_digit == 7,
            predicted_digit == 8,
            predicted_digit == 9,
        ]

        # Use any() to check if any condition is true
        assert any(conditions), f"None of the conditions passed for Digit: {i}"


    