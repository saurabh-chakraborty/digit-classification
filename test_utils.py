from utils import get_hparam_combinations, load_data, split_train_dev_test, tune_hparams, preprocess_data
import sys, os
from sklearn.svm import SVC
from joblib import load
import base64
import pytest
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import unittest
import json
from flask import Flask
# from app.py import app


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
# import pytest
# import numpy as np
# from sklearn.datasets import fetch_openml
# from api.app import app

# @pytest.fixture
# def client():
#     app.config['TESTING'] = True
#     with app.test_client() as client:
#         yield client

# def load_mnist_samples():
#     # Load the MNIST dataset using scikit-learn
#     mnist = fetch_openml('mnist_784', version=1)
#     mnist_images = mnist.data.astype('float32') / 255.0
#     mnist_labels = mnist.target.astype('int')

#     # Reshape the data to match the model's input shape
#     mnist_images = mnist_images.values.reshape((-1, 8, 8))
#     # mnist_images = mnist_images.values.reshape((-1, 28, 28, 1))

#     return mnist_images, mnist_labels

# def flatten_image(image):
#     # Flatten the image to a 1D array
#     return image.flatten()



# def test_post_predict(client):
#     # Load MNIST samples
#     mnist_images, mnist_labels = load_mnist_samples()

#     # Take the first 10 samples for testing
#     test_samples = mnist_images[:10]
#     expected_labels = mnist_labels[:10]

#     for i in range(10):
#         sample_image = test_samples[i]
#         expected_label = expected_labels[i]

#         # Prepare the image data for the POST request
#         image_data = base64.b64encode((sample_image * 255).astype(np.uint8).tobytes()).decode('utf-8')
#         data = {'image': image_data}
        
#         response = client.post('/predict', json=data)
        
#         print(f"Digit: {i}, Response Status Code: {response.status_code}, Response JSON: {response.get_json()}")


#         assert response.status_code == 200
     
#         # Check if the predicted digit matches any of the expected labels
#         predicted_digit = response.get_json()['predicted_digit']

#         # Create a list of assertion conditions for each digit
#         conditions = [
#             predicted_digit == 0,
#             predicted_digit == 1,
#             predicted_digit == 2,
#             predicted_digit == 3,
#             predicted_digit == 4,
#             predicted_digit == 5,
#             predicted_digit == 6,
#             predicted_digit == 7,
#             predicted_digit == 8,
#             predicted_digit == 9,
#         ]

#         # Use any() to check if any condition is true
#         assert any(conditions), f"None of the conditions passed for Digit: {i}"


# test case to check that same random state gives same dataset splits

# def test_same_random_state_produces_same_split():
#     # Generate some example data
#     # X = np.array(range(100)).reshape(-1, 1)
#     # y = np.array(range(100))
#     X, y = load_data()

#     # Set the random state
#     random_state = 42

#     # Create two splits with the same random state
#     X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=random_state)
#     X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, random_state=random_state)

#     # Assert that the splits are exactly the same
#     assert np.array_equal(X_train_1, X_train_2)
#     assert np.array_equal(X_test_1, X_test_2)
#     assert np.array_equal(y_train_1, y_train_2)
#     assert np.array_equal(y_test_1, y_test_2)

# # test case to check that different random state gives different dataset splits

# def test_different_random_state_produces_different_split():
#     # Generate some example data
#     X = np.array(range(100)).reshape(-1, 1)
#     y = np.array(range(100))

#     # Set different random states
#     random_state_1 = 42
#     random_state_2 = 123

#     # Create splits with different random states
#     X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=random_state_1)
#     X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, random_state=random_state_2)

#     # Assert that the splits are different
#     assert not np.array_equal(X_train_1, X_train_2)
#     assert not np.array_equal(X_test_1, X_test_2)
#     assert not np.array_equal(y_train_1, y_train_2)
#     assert not np.array_equal(y_test_1, y_test_2)


def test_lr_tc_1():

    best_model_path = './models/m22aie239_lr_newton-cg.joblib'

    loaded_model = load(best_model_path)

    # Check if the loaded model is a logistic regression model
    if isinstance(loaded_model, LogisticRegression):
        print("The loaded model is a Logistic Regression model.")
    else:
        print("The loaded model is not a Logistic Regression model.")


def test_lr_tc_2():

    best_model_path = './models/m22aie239_lr_newton-cg.joblib'
    model_filename = 'm22aie239_lr_newton-cg.joblib'

    parts = model_filename.split('_')
    if len(parts) >= 2:
        expected_solver = parts[-2]  # Second-to-last part before '.joblib'
    else:
        expected_solver = ''

    loaded_model = load(best_model_path)

    # Check if the loaded model is a logistic regression model
    if isinstance(loaded_model, LogisticRegression):
        # Check if the solver names match
        if loaded_model.solver == expected_solver:
            print(f"The loaded model is a Logistic Regression model with the expected solver: {expected_solver}")
        else:
            print(f"Solver mismatch: Expected solver {expected_solver}, but loaded model has solver {loaded_model.solver}")
    else:
        print("The loaded model is not a Logistic Regression model.")

# API test case for models
# class TestPredictEndpoint(unittest.TestCase):

#     def setUp(self):
#         self.app = app.test_client()

#     def test_predict_svm_route(self):
#         image_vector = {"image": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}}  # Your provided image vector
#         response = self.app.post('/predict/svm', json=image_vector)
#         data = json.loads(response.data.decode('utf-8'))

#         self.assertEqual(response.status_code, 200)
#         self.assertIn('predicted_digit', data)

#     def test_predict_tree_route(self):
#         image_vector = {"image": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}}  # Your provided image vector
#         response = self.app.post('/predict/tree', json=image_vector)
#         data = json.loads(response.data.decode('utf-8'))

#         self.assertEqual(response.status_code, 200)
#         self.assertIn('predicted_digit', data)

#     def test_predict_lr_route(self):
#         image_vector = {"image": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}}  # Your provided image vector
#         response = self.app.post('/predict/lr', json=image_vector)
#         data = json.loads(response.data.decode('utf-8'))

#         self.assertEqual(response.status_code, 200)
#         self.assertIn('predicted_digit', data)

# if __name__ == '__main__':
#     unittest.main()

