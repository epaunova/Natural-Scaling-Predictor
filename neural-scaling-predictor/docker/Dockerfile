FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
