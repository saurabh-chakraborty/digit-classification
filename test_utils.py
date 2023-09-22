# def test_data_splitting():
#     X, y = read_digits()
#     X = X[:100,:,:]
#     y = y[:100]

#     test_size = .1
#     dev_size = .6
#     train_size = 1 - test_size - dev_size

from utils import get_hparam_combinations
def test_hparam_combinations():
    # test case to check whether all possible hparam combos are generated
    gamma_list = []
    gamma_list = [0.001, 0.01, 0.1, 1]

    C_list = []
    C_list = [1, 10, 100, 1000]

    h_params_combinations = get_hparam_combinations(gamma_list, C_list)

    assert len(h_params_combinations) == len(gamma_list) * len(C_list)
    

def test_specific_combinations():
# test case to check whether all possible hparam combos are generated
    gamma_list = []
    gamma_list = [0.001, 0.01, 0.1, 1]

    C_list = []
    C_list = [1, 10, 100, 1000]

    h_params_combinations = get_hparam_combinations(gamma_list, C_list)

    assert len(h_params_combinations) == len(gamma_list) * len(C_list)

    combo_1 = {'C': 10, 'gamma': 0.01}
    combo_2 = {'C': 100, 'gamma': 0.001}
    assert combo_1 in h_params_combinations and combo_2 in h_params_combinations





    