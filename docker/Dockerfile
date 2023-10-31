FROM python:latest

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "python create_json.py ; python digiclf.py ./json_files/config.json"]
