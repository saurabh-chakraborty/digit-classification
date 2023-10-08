import json
from utils import delete_files_shell

# Removing old json files
extn_paths = [
    {'extn': '*.json', 'path': './json_files/'},
]
delete_files_shell(extn_paths)

# Define SVM hyperparameter ranges
gamma_ranges = [0.0001, 0.001, 0.01, 1, 10]
C_ranges = [0.1, 1, 2, 5, 10]
svm_hparams = {}
svm_hparams['gamma'] = gamma_ranges
svm_hparams['C'] = C_ranges

# Define Tree hyperparameter ranges
max_depth_ranges = [5, 10, 15, 20, 50, 100]
max_leaf_nodes_ranges = [2, 5, 10, 15, 20, 50, 75, 100, 200]
tree_hparams = {}
tree_hparams['max_depth'] = max_depth_ranges
tree_hparams['max_leaf_nodes'] = max_leaf_nodes_ranges

hyperparameters_data = {
    'svm': svm_hparams,
    'tree': tree_hparams
}

# File path to save the JSON file
json_path = './json_files/' + 'config.json'

# Write the JSON data to the file
with open(json_path, 'w') as json_file:
    json.dump(hyperparameters_data, json_file, indent=4)

print(f'JSON stored at {json_path}')
