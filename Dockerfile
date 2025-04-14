FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY test_answer.json .
RUN pip install -r requirements.txt 
RUN ltt install torch
COPY summar/templates templates
COPY summar/app.py .

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]