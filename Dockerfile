FROM python:3.12-slim
WORKDIR /app
ENV AI_TRADING_DATA_DIR=/var/lib/ai-trading-bot \
    AI_TRADING_CACHE_DIR=/var/cache/ai-trading-bot \
    AI_TRADING_LOG_DIR=/var/log/ai-trading-bot
COPY . .
RUN mkdir -p $AI_TRADING_DATA_DIR $AI_TRADING_CACHE_DIR $AI_TRADING_LOG_DIR \
    && chmod 700 $AI_TRADING_DATA_DIR $AI_TRADING_CACHE_DIR $AI_TRADING_LOG_DIR \
    && (pip uninstall -y alpaca-trade-api || true) \
    && pip install -r requirements.txt \
    && pip install .[ml]
VOLUME ["/var/lib/ai-trading-bot", "/var/cache/ai-trading-bot", "/var/log/ai-trading-bot"]
CMD ["python", "-m", "ai_trading"]
