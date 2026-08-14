FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .

COPY ["data/ML DATASET_Telco-Customer-Churn.csv", "/app/data/ML DATASET_Telco-Customer-Churn.csv"]

COPY mlruns/1/models/m-70925c8db41c42dd967c6478ffa725fb/artifacts /app/model

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
