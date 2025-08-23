"""
Setup test environment with required environment variables.
"""
import os
os.environ.setdefault('ALPACA_API_KEY', 'test_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook_secret')
os.environ.setdefault('FLASK_PORT', '5000')