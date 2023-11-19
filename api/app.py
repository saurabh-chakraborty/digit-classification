from flask import Flask, request, jsonify, session, render_template, make_response
from flask_session import Session
import numpy as np
import json
from joblib import load, dump
import os
import base64
from sklearn.ensemble import RandomForestClassifier 

model_name = 'tree_train_0.6_dev_0.2_test_0.2_max_depth_15_max_leaf_nodes_100.joblib'
predicted_result = 'None'

app = Flask(__name__)

app.secret_key = '#$@urav'  # Set a secret key for session management
app.config['SESSION_TYPE'] = 'filesystem'  # Choose the session storage type (filesystem in this case)
app.config['SESSION_PERMANENT'] = False  # Sessions are not permanent by default
Session(app)

@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val

@app.route("/sum/<x>/<y>")
def sum_num(x,y):
    sum = int(x) + int(y)
    return str(sum)

# @app.route("/predict", methods = ['POST'])
# def predict():
    
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     models_dir = os.path.join(current_dir, '..', 'models')
#     model_file_path = os.path.join(models_dir, model_name)

#     # Load model
#     model = load(model_file_path)

#     # Get the input data as a JSON object
#     data = request.get_json(force=True)
#     input_vector = np.array(data['image']).reshape(1, -1)
    
#     # Make predictions using the loaded model
#     prediction = model.predict(input_vector)
#     result = prediction[0]

#     # Use to print on console
#     return "\nPredicted Digit = " + str(result)


def predict_form_input(data):
   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    model_file_path = os.path.join(models_dir, model_name)

    # Load model
    model = load(model_file_path)

    input_vector = np.array(data['image']).reshape(1, -1)

    # Make predictions using the loaded model
    prediction = model.predict(input_vector)
    result = prediction[0]
    
    return str(result)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_input', methods=['GET','POST'])
def my_form_post():
    text1 = request.form['text1']
    data1 = json.loads(text1)
    output1 = predict_form_input(data1)

    text2 = request.form['text2']
    data2 = json.loads(text2)
    output2 = predict_form_input(data2)
    if(output1 == output2):
        res = 'True'
    else:
        res = 'False'
    result = {
        "output": res
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

@app.route('/predict_single_input', methods=['GET','POST'])
def form_post():
    text1 = request.form['text1']
    data1 = json.loads(text1)
    output1 = predict_form_input(data1)

    result = {
        "output": output1
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


# Quiz 4 related code

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models')
model_file_path = os.path.join(models_dir, model_name)

# Load model
model = load(model_file_path)


@app.route("/predict", methods=['POST'])
def predict():
    try:
        
        # Get the input data as a JSON object
        data = request.get_json(force=True)
        
        # Decode the base64-encoded image data
        image_data = base64.b64decode(data['image'])

        # Convert bytes to numpy array
        input_vector = np.frombuffer(image_data, dtype=np.uint8)

        # Reshape the data to match the model's input shape
        input_vector = input_vector.reshape(1, -1)

        # Make predictions using the loaded model
        prediction = model.predict(input_vector)
        result = prediction[0]
        return jsonify({'predicted_digit': int(result), 'status': 'success'}), 200

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failure'}), 500

if __name__ == "__main__":
    app.run(debug=True)
