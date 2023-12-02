FROM python:latest

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# Set the FLASK_APP environment variable with the correct path
ENV FLASK_APP=./api/app.py

# CMD ["sh", "-c", "python create_json.py ; python digiclf.py ./json_files/config.json"]

# CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

# Define build arguments
ARG CLASSIFIER_NAME=default_classifier
ARG RANDOM_STATE=default_random_state

# Set environment variables using the build arguments
ENV CLASSIFIER_NAME=$CLASSIFIER_NAME
ENV RANDOM_STATE=$RANDOM_STATE

# Specify the default command
CMD ["python", "digiclf.py", "--classifier", "$CLASSIFIER_NAME", "--random_state", "$RANDOM_STATE"]
# CMD ["python", "digiclf.py", "svm", "42"]

