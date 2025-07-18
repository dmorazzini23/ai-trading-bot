FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install torch==2.2.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
CMD ["python", "run.py"]
