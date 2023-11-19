# FROM python:latest

# COPY requirements.txt .

# RUN pip install -r requirements.txt

# COPY . .

# # CMD ["python", "./digiclf.py", "./json_files/config.json"]

# # CMD ["sh", "-c", "python create_json.py ; python digiclf.py ./json_files/config.json"]

# # Set the FLASK_APP environment variable with the correct path
# ENV FLASK_APP=./api/app.py

# # CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

# CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
