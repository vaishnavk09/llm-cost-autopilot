FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY config ./config
COPY dashboard ./dashboard
COPY data ./data
COPY models ./models
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1
ENV DATABASE_PATH=/app/data/app.db
ENV CLASSIFIER_PATH=/app/models/complexity_clf.pkl

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
