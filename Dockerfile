FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt && pip install .[ml]
CMD ["python", "run.py"]
