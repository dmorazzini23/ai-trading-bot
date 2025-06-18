import os
import sys

required_vars = [
    "FLASK_PORT",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    # Add any other required env vars here
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
    sys.exit(1)
