# 🚀 Deployment Guide

## Overview

This guide covers deployment strategies, environment setup, CI/CD configuration, and production considerations for the AI Trading Bot.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Deployment Strategies](#deployment-strategies)
- [CI/CD Pipeline](#cicd-pipeline)
- [Production Configuration](#production-configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Considerations](#security-considerations)
- [Backup and Recovery](#backup-and-recovery)
- [Scaling and Performance](#scaling-and-performance)

## Environment Setup

### System Requirements

#### Minimum Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 12+
- **Python**: 3.12.3 (exact version required)
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Disk**: 20GB minimum, 50GB recommended
- **Network**: Stable internet connection with low latency

#### Recommended Production Specs
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.12.3
- **RAM**: 16GB+
- **CPU**: 8 cores+
- **Disk**: 100GB SSD
- **Network**: Redundant connections, <50ms latency to exchanges

### Development Environment

#### Local Setup

```bash
# Clone repository
git clone https://github.com/dmorazzini23/ai-trading-bot.git
cd ai-trading-bot

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Validate environment
make validate-env

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys and configuration
```

#### Docker Development

```bash
# Build development image
docker build -t ai-trading-bot:dev .

# Run with mounted source code
docker run -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/.env:/app/.env \
  -p 5000:5000 \
  ai-trading-bot:dev bash

# Run development server
docker-compose -f docker-compose.dev.yml up
```

### Staging Environment

```bash
# Setup staging environment
cp .env.example .env.staging

# Configure staging-specific variables
export BOT_MODE=staging
export ALPACA_BASE_URL=https://paper-api.alpaca.markets
export LOG_LEVEL=DEBUG

# Deploy to staging
./deploy.sh staging
```

## Deployment Strategies

### 1. Single Server Deployment

#### Basic Production Setup

```bash
# Create production user
sudo useradd -m -s /bin/bash ai-trading
sudo usermod -aG sudo ai-trading

# Setup application directory
sudo mkdir -p /opt/ai-trading-bot
sudo chown ai-trading:ai-trading /opt/ai-trading-bot

# Switch to application user
sudo su - ai-trading

# Clone and setup
cd /opt/ai-trading-bot
git clone https://github.com/dmorazzini23/ai-trading-bot.git .
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with production configuration

# Install systemd service
sudo cp ai-trading-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-scheduler
sudo systemctl start ai-trading-scheduler
```

#### Systemd Service Configuration

```ini
# /etc/systemd/system/ai-trading-scheduler.service
[Unit]
Description=AI Trading Bot Scheduler
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ai-trading
Group=ai-trading
WorkingDirectory=/opt/ai-trading-bot
Environment=PATH=/opt/ai-trading-bot/venv/bin
ExecStart=/opt/ai-trading-bot/venv/bin/python -m ai_trading
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/ai-trading-bot/logs /opt/ai-trading-bot/data

[Install]
WantedBy=multi-user.target
```

### 2. Docker Deployment

#### Production Dockerfile

```dockerfile
FROM python:3.12.3-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r ai-trading && useradd -r -g ai-trading ai-trading

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R ai-trading:ai-trading /app

# Switch to non-root user
USER ai-trading

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python health_check.py || exit 1

# Start application
CMD ["python", "-m", "ai_trading"]
```

#### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  ai-trading-bot:
    build: .
    container_name: ai-trading-bot
    restart: unless-stopped
    environment:
      - BOT_MODE=production
      - PYTHONUNBUFFERED=1
    env_file:
      - .env.production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./config:/app/config:ro
    ports:
      - "5000:5000"
    healthcheck:
      test: ["CMD", "python", "health_check.py"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  monitoring:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards

volumes:
  grafana-storage:
```

### 3. Kubernetes Deployment

#### Deployment Configuration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-trading-bot
  labels:
    app: ai-trading-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-trading-bot
  template:
    metadata:
      labels:
        app: ai-trading-bot
    spec:
      containers:
      - name: ai-trading-bot
        image: ai-trading-bot:latest
        ports:
        - containerPort: 5000
        env:
        - name: BOT_MODE
          value: "production"
        envFrom:
        - secretRef:
            name: ai-trading-secrets
        - configMapRef:
            name: ai-trading-config
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: data
          mountPath: /app/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: ai-trading-logs
      - name: data
        persistentVolumeClaim:
          claimName: ai-trading-data
```

#### Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-trading-service
spec:
  selector:
    app: ai-trading-bot
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-trading-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - trading-bot.yourdomain.com
    secretName: ai-trading-tls
  rules:
  - host: trading-bot.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-trading-service
            port:
              number: 80
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy AI Trading Bot

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.12.3
      uses: actions/setup-python@v4
      with:
        python-version: 3.12.3
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest -n auto --disable-warnings --cov=ai_trading --cov-fail-under=80
    
    - name: Run linting
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        mypy ai_trading --ignore-missing-imports

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ai-trading-bot:${{ github.sha }} .
        docker tag ai-trading-bot:${{ github.sha }} ai-trading-bot:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push ai-trading-bot:${{ github.sha }}
        docker push ai-trading-bot:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        # Add deployment script here
        echo "Deploying to production..."
```

### Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo "Deploying AI Trading Bot to $ENVIRONMENT..."

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    echo "Error: Environment must be 'staging' or 'production'"
    exit 1
fi

# Backup current deployment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Creating backup..."
    docker tag ai-trading-bot:current ai-trading-bot:backup-$(date +%Y%m%d-%H%M%S) || true
fi

# Pull latest image
echo "Pulling image: ai-trading-bot:$VERSION"
docker pull ai-trading-bot:$VERSION

# Update configuration
echo "Updating configuration..."
cp .env.$ENVIRONMENT .env

# Stop current container
echo "Stopping current container..."
docker-compose -f docker-compose.$ENVIRONMENT.yml down || true

# Start new container
echo "Starting new container..."
docker-compose -f docker-compose.$ENVIRONMENT.yml up -d

# Health check
echo "Performing health check..."
sleep 30
if ! curl -f http://localhost:5000/health; then
    echo "Health check failed! Rolling back..."
    docker-compose -f docker-compose.$ENVIRONMENT.yml down
    docker tag ai-trading-bot:backup-$(date +%Y%m%d) ai-trading-bot:current || true
    docker-compose -f docker-compose.$ENVIRONMENT.yml up -d
    exit 1
fi

echo "Deployment successful!"
```

## Production Configuration

### Environment Variables

```bash
# .env.production
# Trading Configuration
BOT_MODE=production
SCHEDULER_SLEEP_SECONDS=60
MAX_POSITION_PCT=0.05
MAX_PORTFOLIO_HEAT=0.15

# API Configuration
ALPACA_BASE_URL=https://api.alpaca.markets
FINNHUB_API_KEY=your_finnhub_key

# Logging
LOG_LEVEL=INFO
BOT_LOG_FILE=/app/logs/scheduler.log
LOG_ROTATION_SIZE=100MB
LOG_RETENTION_DAYS=30

# Security
API_SECRET_KEY=your_secure_random_key
ENABLE_AUDIT_LOGGING=true

# Performance
USE_PARALLEL_PROCESSING=true
CACHE_TIMEOUT=300
CONNECTION_POOL_SIZE=20

# Monitoring
PROMETHEUS_PORT=8000
HEALTH_CHECK_INTERVAL=30
```

### Configuration Validation

```python
# production_validator.py
import os
from typing import List, Tuple

def validate_production_config() -> Tuple[bool, List[str]]:
    """Validate production configuration."""
    errors = []
    
    # Required environment variables
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'API_SECRET_KEY',
        'BOT_MODE'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Validate bot mode
    if os.getenv('BOT_MODE') not in ['production', 'staging']:
        errors.append("BOT_MODE must be 'production' or 'staging'")
    
    # Validate numeric values
    try:
        max_pos = float(os.getenv('MAX_POSITION_PCT', 0.05))
        if max_pos <= 0 or max_pos > 0.5:
            errors.append("MAX_POSITION_PCT must be between 0 and 0.5")
    except ValueError:
        errors.append("MAX_POSITION_PCT must be a valid number")
    
    return len(errors) == 0, errors

if __name__ == "__main__":
    is_valid, errors = validate_production_config()
    if not is_valid:
        for error in errors:
            print(f"ERROR: {error}")
        exit(1)
    print("Production configuration is valid")
```

## Monitoring and Logging

### Log Aggregation

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logging/logstash.conf:/usr/share/logstash/pipeline/logstash.conf:ro
      - ./logs:/app/logs:ro
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch-data:
```

### Prometheus Metrics

```python
# metrics_exporter.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
trades_total = Counter('trades_total', 'Total number of trades', ['symbol', 'side'])
trade_duration = Histogram('trade_execution_seconds', 'Trade execution time')
portfolio_value = Gauge('portfolio_value_usd', 'Current portfolio value')
open_positions = Gauge('open_positions_count', 'Number of open positions')

def export_metrics():
    """Export metrics to Prometheus."""
    start_http_server(8000)
    
    while True:
        # Update metrics here
        time.sleep(60)

if __name__ == "__main__":
    export_metrics()
```

### Alerting Rules

```yaml
# monitoring/alerts.yml
groups:
- name: ai_trading_bot
  rules:
  - alert: TradingBotDown
    expr: up{job="ai-trading-bot"} == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: AI Trading Bot is down
      description: "The AI Trading Bot has been down for more than 5 minutes."

  - alert: HighDrawdown
    expr: portfolio_drawdown > 0.1
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: High portfolio drawdown detected
      description: "Portfolio drawdown is {{ $value }}%, exceeding 10% threshold."

  - alert: ApiErrors
    expr: rate(api_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: High API error rate
      description: "API error rate is {{ $value }} errors/second."
```

## Security Considerations

### API Key Management

```bash
# Use HashiCorp Vault for secrets management
vault kv put secret/ai-trading-bot \
  alpaca_api_key="your_key" \
  alpaca_secret_key="your_secret" \
  api_secret_key="your_api_secret"

# Access in application
vault kv get -field=alpaca_api_key secret/ai-trading-bot
```

### Network Security

```bash
# Firewall configuration
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 5000/tcp  # Application port
sudo ufw allow from 10.0.0.0/8 to any port 9090  # Prometheus (internal only)
```

### SSL/TLS Configuration

```nginx
# /etc/nginx/sites-available/ai-trading-bot
server {
    listen 443 ssl http2;
    server_name trading-bot.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/trading-bot.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/trading-bot.yourdomain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Backup and Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/opt/backups/ai-trading-bot"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="ai-trading-bot-backup-$DATE.tar.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application data
tar -czf $BACKUP_DIR/$BACKUP_FILE \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  /opt/ai-trading-bot

# Backup database (if applicable)
# pg_dump ai_trading_db > $BACKUP_DIR/database-$DATE.sql

# Upload to cloud storage
aws s3 cp $BACKUP_DIR/$BACKUP_FILE s3://your-backup-bucket/

# Clean old local backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
```

### Recovery Procedure

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
sudo systemctl stop ai-trading-scheduler

# Backup current state
mv /opt/ai-trading-bot /opt/ai-trading-bot.old

# Restore from backup
tar -xzf $BACKUP_FILE -C /

# Restore ownership
sudo chown -R ai-trading:ai-trading /opt/ai-trading-bot

# Start application
sudo systemctl start ai-trading-scheduler

echo "Recovery completed from $BACKUP_FILE"
```

## Scaling and Performance

### Horizontal Scaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-trading-bot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-trading-bot
  minReplicas: 1
  maxReplicas: 3
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Performance Optimization

```python
# performance_config.py
import os

# Connection pooling
CONNECTION_POOL_SIZE = int(os.getenv('CONNECTION_POOL_SIZE', 20))
MAX_CONNECTIONS = int(os.getenv('MAX_CONNECTIONS', 100))

# Caching
CACHE_TTL = int(os.getenv('CACHE_TTL', 300))
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Parallel processing
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
USE_MULTIPROCESSING = os.getenv('USE_MULTIPROCESSING', 'true').lower() == 'true'

# Memory optimization
GC_THRESHOLD = int(os.getenv('GC_THRESHOLD', 700))
MAX_MEMORY_MB = int(os.getenv('MAX_MEMORY_MB', 2048))
```

This deployment guide provides comprehensive coverage of production deployment strategies, monitoring, security, and operational considerations for the AI Trading Bot.