#!/bin/bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
echo "✅ Pycache and .pyc files cleaned up."
