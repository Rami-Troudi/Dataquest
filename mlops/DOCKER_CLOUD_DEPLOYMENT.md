# Docker & Cloud Deployment Guide

Complete guide to deploy MLOps Insurance Model API locally with Docker and to the cloud.

---

## Table of Contents

1. [Local Docker Deployment](#local-docker-deployment)
2. [Cloud Platforms](#cloud-platforms)
3. [Environment Variables](#environment-variables)
4. [Monitoring & Logging](#monitoring--logging)
5. [Troubleshooting](#troubleshooting)

---

## Local Docker Deployment

### Prerequisites

- Docker installed: https://docker.com
- Docker Compose installed: https://docs.docker.com/compose/install/
- model.pkl in project root
- test.csv in project root (optional, for testing)

### Quick Start

```bash
# Build the Docker image
docker-compose build

# Start the services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f mlops-api

# Stop services
docker-compose down
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
# or
curl http://localhost:8000/

# Make a prediction (via CSV)
curl -X POST http://localhost:8000/predict_csv \
  -F "file=@test.csv" \
  | head -20

# Make single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "User_ID": "TEST_001",
    "Policy_Cancelled_Post_Purchase": 0,
    "Policy_Start_Year": 2015,
    "Policy_Start_Week": 43,
    "Policy_Start_Day": 20,
    "Grace_Period_Extensions": 1,
    "Previous_Policy_Duration_Months": 5,
    "Adult_Dependents": 2,
    "Child_Dependents": 0.0,
    "Infant_Dependents": 0,
    "Region_Code": "DEU",
    "Existing_Policyholder": 0,
    "Previous_Claims_Filed": 0,
    "Years_Without_Claims": 0,
    "Policy_Amendments_Count": 0,
    "Broker_ID": 16.0,
    "Employer_ID": null,
    "Underwriting_Processing_Days": 0,
    "Vehicles_on_Policy": 0,
    "Custom_Riders_Requested": 0,
    "Broker_Agency_Type": "Urban_Boutique",
    "Deductible_Tier": "Tier_1_High_Ded",
    "Acquisition_Channel": "Local_Broker",
    "Payment_Schedule": "Monthly_EFT",
    "Employment_Status": "Employed_FullTime",
    "Estimated_Annual_Income": 24493.85,
    "Days_Since_Quote": 87,
    "Policy_Start_Month": "February"
  }'
```

### Docker Compose Commands

```bash
# Build images
docker-compose build

# Start services (background)
docker-compose up -d

# Start services (foreground, see logs)
docker-compose up

# View running services
docker-compose ps

# View logs
docker-compose logs -f              # All services
docker-compose logs -f mlops-api    # Specific service

# Execute command in container
docker-compose exec mlops-api bash

# Stop services
docker-compose stop

# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Rebuild and restart
docker-compose down && docker-compose up -d --build
```

### Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Base**: http://localhost:8000

---

## Cloud Platforms

### 1. AWS (Elastic Container Service + App Runner)

#### Option A: AWS App Runner (Easiest)

```bash
# Push to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build -t insurance-mlops:latest -f mlops/Dockerfile .

docker tag insurance-mlops:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/insurance-mlops:latest

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/insurance-mlops:latest

# Then create App Runner service in AWS Console pointing to this ECR image
```

#### Option B: AWS ECS

```yaml
# ecs-task-definition.json
{
  "family": "insurance-mlops",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "mlops-api",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/insurance-mlops:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/insurance-mlops",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 10
      }
    }
  ]
}
```

Deploy:
```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
aws ecs create-service --cluster insurance-cluster --service-name mlops-api --task-definition insurance-mlops
```

### 2. Google Cloud (Cloud Run)

```bash
# Authenticate
gcloud auth login

# Configure project
gcloud config set project YOUR_PROJECT_ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/insurance-mlops:latest

# Deploy to Cloud Run
gcloud run deploy insurance-mlops \
  --image gcr.io/YOUR_PROJECT_ID/insurance-mlops:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 100 \
  --port 8000

# Get service URL
gcloud run services describe insurance-mlops --region us-central1
```

### 3. Azure Container Instances

```bash
# Create resource group
az group create --name insurance-rg --location eastus

# Create container registry
az acr create --resource-group insurance-rg --name insurancemlops --sku Basic

# Build and push image
az acr build --registry insurancemlops --image insurance-mlops:latest .

# Deploy container instance
az container create \
  --resource-group insurance-rg \
  --name insurance-mlops-api \
  --image insurancemlops.azurecr.io/insurance-mlops:latest \
  --cpu 2 \
  --memory 2 \
  --registry-login-server insurancemlops.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --ports 8000 \
  --dns-name-label insurance-mlops-api

# Get URL
az container show --resource-group insurance-rg --name insurance-mlops-api --query ipAddress.fqdn
```

### 4. Heroku (Deprecated but still works with workarounds)

Use Railway or Render instead. Here's **Railway** (recommended for Heroku replacement):

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Create Procfile
echo "web: /app/entrypoint.sh" > Procfile

# Deploy
railway up

# View logs
railway logs

# Get public URL
railway status
```

### 5. DigitalOcean App Platform

1. Go to https://cloud.digitalocean.com/apps
2. Click "Create" → "App"
3. Connect your GitHub repo
4. Configure:
   - **Build Command**: `docker build -f mlops/Dockerfile -t insurance-mlops:latest .`
   - **Run Command**: `/app/entrypoint.sh`
5. Set environment variables
6. Deploy

Or via CLI:

```bash
# Install doctl
brew install doctl  # or: curl https://github.com/digitalocean/doctl/releases/download/v1.94.0/doctl-1.94.0-linux-x64.tar.gz | tar xz

# Authenticate
doctl auth init

# Create app from spec
doctl apps create --spec app.yaml

# View logs
doctl apps logs <app-id>
```

### 6. Docker Hub + Any Linux Server

```bash
# Push to Docker Hub
docker tag insurance-mlops:latest YOUR_USERNAME/insurance-mlops:latest
docker push YOUR_USERNAME/insurance-mlops:latest

# On your Linux server (Ubuntu 20.04+)
ssh your-server

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Run container
docker run -d \
  --name insurance-mlops \
  -p 8000:8000 \
  -v /path/to/model.pkl:/app/model.pkl:ro \
  --restart unless-stopped \
  YOUR_USERNAME/insurance-mlops:latest

# View logs
docker logs -f insurance-mlops
```

---

## Environment Variables

### Set in Docker (docker-compose.yml or CLI)

```bash
docker run -e API_HOST=0.0.0.0 -e API_PORT=8000 ...
```

### Supported Variables

```env
# Application
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_PATH=model.pkl
MODEL_NAME=insurance_rf
MODEL_VERSION=1.0.0

# Logging
LOG_LEVEL=info
LOG_FORMAT=json

# Performance
WORKERS=4
TIMEOUT=300

# Security
API_KEY=your-api-key  # Optional, implement auth if needed
```

### Set in docker-compose.yml

```yaml
services:
  mlops-api:
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - LOG_LEVEL=info
      - WORKERS=4
```

---

## Monitoring & Logging

### Logs

```bash
# Local Docker
docker-compose logs -f mlops-api

# AWS CloudWatch (ECS)
aws logs tail /ecs/insurance-mlops --follow

# Google Cloud Logging
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=insurance-mlops" --limit 50 --format json

# Azure Monitor
az monitor log-analytics query --workspace-id <workspace-id> --analytics-query "ContainerInstanceLog_CL | tail 50"
```

### Health Check

```bash
# Local
curl http://localhost:8000/health

# Cloud URL
curl https://your-cloud-app-url.com/health

# Expected response
{
  "status": "healthy",
  "model_loaded": true,
  "current_model": "insurance_rf:1.0.0",
  "version": "3.0.0"
}
```

### Metrics

The API provides:
- Request count
- Response time
- Model accuracy
- Prediction latency

View in Swagger UI: http://localhost:8000/docs

### Setting Up Monitoring

#### Prometheus + Grafana (On-premise or Local)

```bash
# Install Prometheus client
pip install prometheus-client

# Update app_v3.py to export metrics (already built-in if you add middleware)
# Metrics available at http://localhost:8000/metrics
```

#### Cloud-Native Monitoring

**AWS CloudWatch**:
```bash
# Automatically logs to CloudWatch when using ECS/App Runner
# View in AWS Console → CloudWatch → Log Groups
```

**Google Cloud Monitoring**:
```bash
# Automatic for Cloud Run
# View in Google Cloud Console → Monitoring
```

**Azure Monitor**:
```bash
# Automatic for Container Instances
# View in Azure Portal → Monitor
```

---

## Troubleshooting

### Problem: Container won't start

```bash
# Check logs
docker-compose logs mlops-api

# Common issues:
# - Model file not found
# - Port already in use  
# - Insufficient memory

# Solution: Increase memory
docker-compose down
docker-compose up -d --build

# Or manually with more memory
docker run -m 2g insurance-mlops:latest
```

### Problem: API returns "No model loaded"

```bash
# Ensure model.pkl exists
ls -la model.pkl

# Check volume mount in Docker
docker-compose exec mlops-api ls -la /app/

# Verify model loads
docker-compose exec mlops-api python -c "import joblib; joblib.load('/app/model.pkl')"
```

### Problem: Slow predictions

```bash
# Check container resources
docker stats insurance-mlops-api

# Increase CPU/memory in docker-compose.yml
# For cloud: check resource allocation in cloud console
```

### Problem: High memory usage

```bash
# Model is loaded once at startup
# If memory issues persist, reduce logging:
# Set LOG_LEVEL=error in environment
```

---

## Production Checklist

- [ ] Use HTTPS/SSL certificates
- [ ] Set resource limits (CPU, memory)
- [ ] Configure logging and monitoring
- [ ] Set up auto-scaling
- [ ] Enable health checks
- [ ] Configure restart policies
- [ ] Backup model files
- [ ] Set up alerting
- [ ] Document API endpoints
- [ ] Test failover scenarios

---

## Quick Deploy Script

```bash
#!/bin/bash
# deploy.sh - Deploy to chosen platform

PLATFORM=$1
case $PLATFORM in
  local)
    docker-compose up -d --build
    echo "✓ Running on http://localhost:8000"
    ;;
  gcloud)
    gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/insurance-mlops:latest
    gcloud run deploy insurance-mlops --image gcr.io/YOUR_PROJECT_ID/insurance-mlops:latest
    ;;
  aws)
    aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
    docker build -t insurance-mlops:latest .
    docker tag insurance-mlops:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/insurance-mlops:latest
    docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/insurance-mlops:latest
    ;;
  *)
    echo "Usage: $0 {local|gcloud|aws}"
    ;;
esac
```

---

## Next Steps

1. **Local Testing**: `docker-compose up -d`
2. **Cloud Deployment**: Choose platform above
3. **Load Testing**: Use `mlops/test_api.py` or Apache JMeter
4. **Monitor**: Set up logging and alerting
5. **Scale**: Configure auto-scaling for production

**Status**: Ready for deployment! 🚀
