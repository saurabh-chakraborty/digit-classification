from utils import get_hparam_combinations, load_data, split_train_dev_test, tune_hparams, preprocess_data
import sys, os
from sklearn.svm import SVC
from joblib import load

def test_hparam_combinations():
    # test case to check whether all possible hparam combos are generated
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    svm_hparams = {}
    svm_hparams['gamma'] = gamma_list
    svm_hparams['C'] = C_list

    # Create a list of dictionaries for all hparam combinations
    h_params_combinations = get_hparam_combinations(svm_hparams)

    assert len(h_params_combinations) == len(gamma_list) * len(C_list)
    

def test_specific_combinations():
# test case to check whether all possible hparam combos are generated
    gamma_list = [0.001, 0.01, 0.1, 1]
    C_list = [1, 10, 100, 1000]
    svm_hparams = {}
    svm_hparams['gamma'] = gamma_list
    svm_hparams['C'] = C_list

    # Create a list of dictionaries for all hparam combinations
    h_params_combinations = get_hparam_combinations(svm_hparams)

    assert len(h_params_combinations) == len(gamma_list) * len(C_list)

    combo_1 = {'C': 10, 'gamma': 0.01}
    combo_2 = {'C': 100, 'gamma': 0.001}
    assert combo_1 in h_params_combinations and combo_2 in h_params_combinations

def test_data_splitting():
    X, y = load_data()
    X = X[:100,:,:]
    y = y[:100]

    test_size = .1
    dev_size = .6
    
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size, dev_size)

    assert (len(X_train) == 30) 
    assert (len(X_test) == 10)
    assert ((len(X_dev) == 60))

def test_modelSave_modelType():    

    X, y = load_data()
    X = X[:100,:,:]
    y = y[:100]
    
    gamma_list = [0.001, 0.01]
    C_list = [1, 2]
    svm_hparams = {}
    svm_hparams['gamma'] = gamma_list
    svm_hparams['C'] = C_list

    # Create a list of dictionaries for all hparam combinations
    h_params_combinations = get_hparam_combinations(svm_hparams)

    test_size = 0.1
    dev_size = 0.6
    train_size = 1-(test_size+dev_size)
    x_train, x_dev, x_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_size, dev_size)

    clf = 'svm'

    outputs = tune_hparams(x_train, y_train, x_dev, y_dev, h_params_combinations, clf, train_size, dev_size, test_size)
    actual_model_path = outputs[-1]

    assert(os.path.exists(actual_model_path)==True)
    assert type(load(actual_model_path)) == SVC



    