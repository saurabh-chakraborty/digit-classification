from flask import Flask, request, jsonify, session, render_template, make_response
from flask_session import Session
import numpy as np
import json
from joblib import load
import os

model_name = 'svm_train_0.6_dev_0.2_test_0.2_gamma_0.001_C_1.joblib'
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

@app.route("/predict", methods = ['POST'])
def predict():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    model_file_path = os.path.join(models_dir, model_name)

    # Load model
    model = load(model_file_path)

    # Get the input data as a JSON object
    data = request.get_json(force=True)
    input_vector = np.array(data['image']).reshape(1, -1)
    
    # Make predictions using the loaded model
    prediction = model.predict(input_vector)
    result = prediction[0]
    return "Predicted Digit = " + str(result)


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
    data = json.loads(text1)
    output = predict_form_input(data)
    result = {
        "output": output
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)
    