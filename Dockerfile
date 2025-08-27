FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip uninstall -y alpaca-trade-api || true \
    && pip install -r requirements.txt \
    && pip install .[ml]
CMD ["python", "-m", "ai_trading"]
