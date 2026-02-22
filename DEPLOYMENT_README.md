# 🚀 Deployment Guide - Start to Finish

Complete guide to deploy your MLOps Insurance Model API to the cloud (for free).

---

## 📋 Prerequisites

Before deploying, ensure:

- ✅ Docker works locally (`docker-compose up` succeeds)
- ✅ Git repository with your code
- ✅ `model.pkl` exists in the project root
- ✅ API tested locally at http://localhost:8000

---

## ⚡ Quick Deploy Options

Choose your platform:

| Platform | Free Tier | Setup Time | Best For |
|----------|-----------|------------|----------|
| [Render](#render-deployment-recommended) | ✅ 750hrs/month | 5 min | **Easiest** |
| [Railway](#railway-deployment) | ✅ $5 credit | 5 min | GitHub Students |
| [Google Cloud Run](#google-cloud-run) | ✅ $300 credit | 10 min | Scalability |
| [Hugging Face Spaces](#hugging-face-spaces) | ✅ Free | 15 min | ML Projects |
| [Azure](#azure-deployment) | Student pack | 15 min | Enterprise |

---

# 🎯 Render Deployment (Recommended)

**Why Render?**
- ✅ 750 free hours/month
- ✅ Auto-deploy from GitHub
- ✅ Built-in HTTPS
- ✅ No credit card required

## Step 1: Push to GitHub

```powershell
# If not already a git repo
git init
git add .
git commit -m "Initial commit with MLOps setup"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

## Step 2: Sign up on Render

1. Go to https://render.com
2. Sign up with GitHub (easiest)
3. Authorize Render to access your repos

## Step 3: Create Web Service

1. Click **"New +"** → **"Web Service"**
2. Select your GitHub repository
3. Configure:

```yaml
Name: insurance-mlops-api
Region: Oregon (US West)
Branch: main
Root Directory: (leave empty)
Environment: Docker
Dockerfile Path: mlops/Dockerfile
Docker Build Context: .
Instance Type: Free
```

## Step 4: Add Environment Variables

In Render dashboard, add:

```
API_HOST=0.0.0.0
API_PORT=8000
```

## Step 5: Deploy

Click **"Create Web Service"**

Render will:
1. Clone your repo
2. Build Docker image
3. Deploy and give you a URL like: `https://insurance-mlops-api.onrender.com`

⏱️ First deploy takes ~5-10 minutes.

## Step 6: Test your deployment

```powershell
# Replace with your actual Render URL
$URL = "https://insurance-mlops-api.onrender.com"

# Health check
Invoke-RestMethod "$URL/health"

# Make prediction
$body = @{ file = Get-Item .\test.csv }
Invoke-RestMethod -Uri "$URL/predict_csv" -Method Post -Form $body
```

### 📊 Access API Documentation

Visit: `https://your-app.onrender.com/docs` (Swagger UI)

---

# 🚂 Railway Deployment

**Why Railway?**
- ✅ $5 free credit for GitHub Students
- ✅ Simple CLI deployment
- ✅ Auto HTTPS

## Step 1: Install Railway CLI

```powershell
npm install -g @railway/cli
# or
iwr https://railway.app/install.ps1 | iex
```

## Step 2: Login

```powershell
railway login
```

## Step 3: Initialize Project

```powershell
cd "c:\Users\HP\OneDrive\Documents\Nouveau dossier\Dataquest"
railway init
```

Select: **"Create new project"**

## Step 4: Deploy

```powershell
railway up
```

Railway will:
1. Detect your Dockerfile
2. Build and deploy
3. Give you a URL

## Step 5: Get Public URL

```powershell
railway domain
```

## Step 6: Set Environment Variables

```powershell
railway variables set API_HOST=0.0.0.0
railway variables set API_PORT=8000
```

## Step 7: Test

```powershell
$URL = "https://your-app.railway.app"
Invoke-RestMethod "$URL/health"
```

---

# ☁️ Google Cloud Run

**Why Cloud Run?**
- ✅ $300 free credit (new users)
- ✅ Auto-scaling
- ✅ Pay per request

## Step 1: Install Google Cloud SDK

Download from: https://cloud.google.com/sdk/docs/install

Or use PowerShell:

```powershell
Invoke-WebRequest https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe -OutFile GoogleCloudSDKInstaller.exe
.\GoogleCloudSDKInstaller.exe
```

## Step 2: Authenticate

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Step 3: Build and Push

```powershell
cd "c:\Users\HP\OneDrive\Documents\Nouveau dossier\Dataquest"

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/insurance-mlops -f mlops/Dockerfile .
```

## Step 4: Deploy to Cloud Run

```powershell
gcloud run deploy insurance-mlops `
  --image gcr.io/YOUR_PROJECT_ID/insurance-mlops `
  --platform managed `
  --region us-central1 `
  --memory 2Gi `
  --cpu 2 `
  --timeout 300 `
  --port 8000 `
  --allow-unauthenticated
```

## Step 5: Get URL

```powershell
gcloud run services describe insurance-mlops --region us-central1 --format "value(status.url)"
```

## Step 6: Test

```powershell
$URL = "https://insurance-mlops-xxxxx-uc.a.run.app"
Invoke-RestMethod "$URL/health"
```

---

# 🤗 Hugging Face Spaces

**Why Hugging Face?**
- ✅ Completely free
- ✅ Perfect for ML models
- ✅ Built-in Docker support

## Step 1: Create Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Name: `insurance-mlops-api`
4. License: MIT
5. SDK: **Docker**
6. Click **"Create Space"**

## Step 2: Clone Space

```powershell
git clone https://huggingface.co/spaces/YOUR_USERNAME/insurance-mlops-api
cd insurance-mlops-api
```

## Step 3: Copy Files

```powershell
# Copy from your project
Copy-Item -Path "C:\Users\HP\OneDrive\Documents\Nouveau dossier\Dataquest\mlops\*" -Destination . -Recurse
Copy-Item -Path "C:\Users\HP\OneDrive\Documents\Nouveau dossier\Dataquest\model.pkl" -Destination .
```

## Step 4: Create Dockerfile (simplified for HF)

Create `Dockerfile` in the space root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY mlops/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY mlops/app_v3.py mlops/
COPY mlops/model_loader.py mlops/
COPY mlops/__init__.py mlops/
COPY model.pkl .

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "mlops.app_v3:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Step 5: Push to HF

```powershell
git add .
git commit -m "Deploy MLOps API"
git push
```

Your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/insurance-mlops-api
```

---

# ☁️ Azure Deployment

**Prerequisites:**
- Active Azure subscription (Student pack)
- Azure CLI installed

## Step 1: Setup Azure CLI

```powershell
# Add to PATH
$azDir = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
$env:Path = "$azDir;" + $env:Path

# Login
az login
az account list --all -o table
```

## Step 2: Create Resources

```powershell
$RG = "mlops-rg"
$LOC = "eastus"
$ACR = "mlopsacr$(Get-Random -Maximum 9999)"
$APP = "insurance-mlops-api"
$DNS = "mlops$(Get-Random -Maximum 9999)"

# Create resource group
az group create --name $RG --location $LOC

# Create container registry
az acr create --resource-group $RG --name $ACR --sku Basic
az acr login --name $ACR
```

## Step 3: Build and Push

```powershell
cd "c:\Users\HP\OneDrive\Documents\Nouveau dossier\Dataquest"

az acr build --registry $ACR --image insurance-mlops:latest -f mlops/Dockerfile .
```

## Step 4: Deploy to ACI

```powershell
az container create `
  --resource-group $RG `
  --name $APP `
  --image "$ACR.azurecr.io/insurance-mlops:latest" `
  --registry-login-server "$ACR.azurecr.io" `
  --dns-name-label $DNS `
  --ports 8000 `
  --cpu 2 --memory 2 `
  --environment-variables API_HOST=0.0.0.0 API_PORT=8000
```

## Step 5: Get Public URL

```powershell
az container show --resource-group $RG --name $APP --query ipAddress.fqdn -o tsv
```

## Step 6: Test

```powershell
$URL = "http://mlops1234.eastus.azurecontainer.io:8000"
Invoke-RestMethod "$URL/health"
```

---

# 🧪 Testing Your Deployment

Once deployed on any platform, test with these commands:

## 1. Health Check

```powershell
$URL = "https://your-app-url.com"
Invoke-RestMethod "$URL/health"
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "current_model": "insurance_rf:1.0.0",
  "version": "3.0.0"
}
```

## 2. Single Prediction

```powershell
$body = @{
  User_ID = "TEST_001"
  Policy_Cancelled_Post_Purchase = 0
  Policy_Start_Year = 2015
  Policy_Start_Week = 43
  Policy_Start_Day = 20
  Grace_Period_Extensions = 1
  Previous_Policy_Duration_Months = 5
  Adult_Dependents = 2
  Child_Dependents = 0.0
  Infant_Dependents = 0
  Region_Code = "DEU"
  Existing_Policyholder = 0
  Previous_Claims_Filed = 0
  Years_Without_Claims = 0
  Policy_Amendments_Count = 0
  Broker_ID = 16.0
  Employer_ID = $null
  Underwriting_Processing_Days = 0
  Vehicles_on_Policy = 0
  Custom_Riders_Requested = 0
  Broker_Agency_Type = "Urban_Boutique"
  Deductible_Tier = "Tier_1_High_Ded"
  Acquisition_Channel = "Local_Broker"
  Payment_Schedule = "Monthly_EFT"
  Employment_Status = "Employed_FullTime"
  Estimated_Annual_Income = 24493.85
  Days_Since_Quote = 87
  Policy_Start_Month = "February"
} | ConvertTo-Json

Invoke-RestMethod -Uri "$URL/predict" -Method Post -Body $body -ContentType "application/json"
```

## 3. CSV Batch Prediction

```powershell
$form = @{ file = Get-Item .\test.csv }
Invoke-RestMethod -Uri "$URL/predict_csv" -Method Post -Form $form
```

## 4. View API Documentation

Visit in browser:
```
https://your-app-url.com/docs
```

---

# 🔒 Security (Optional)

## Add API Key Authentication

1. Set environment variable:
```
API_KEY=your-secret-key-here
```

2. Update `app_v3.py` to check headers:
```python
from fastapi import Header, HTTPException

async def verify_token(x_api_key: str = Header()):
    if x_api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
```

3. Test with auth:
```powershell
$headers = @{ "X-API-Key" = "your-secret-key-here" }
Invoke-RestMethod -Uri "$URL/health" -Headers $headers
```

---

# 📊 Monitoring

## View Logs

### Render
Dashboard → Logs tab

### Railway
```powershell
railway logs
```

### Google Cloud Run
```powershell
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

### Azure
```powershell
az container logs --resource-group $RG --name $APP
```

---

# 💰 Cost Estimate

| Platform | Free Tier | After Free | Best For |
|----------|-----------|------------|----------|
| Render | 750 hrs/mo | $7/mo | Development |
| Railway | $5 credit | Pay-as-go | Testing |
| GCP Cloud Run | $300 credit | ~$0.00001/req | Production |
| HF Spaces | Unlimited | Free | ML demos |
| Azure ACI | Student credit | ~$0.10/hr | Enterprise |

---

# 🎓 Summary

**Fastest path (5 minutes):**
1. Push code to GitHub
2. Sign up on Render.com
3. Connect repo → Deploy
4. Done! ✅

**Most powerful:**
- Google Cloud Run (auto-scaling)

**Completely free forever:**
- Hugging Face Spaces

**Best for learning:**
- Railway (simple CLI)

---

# ✅ Your Deployment Checklist

Before deploying:

- [ ] `model.pkl` exists in project root
- [ ] Docker builds locally: `docker-compose -f mlops/docker-compose.yml up`
- [ ] Health check works locally: `http://localhost:8000/health`
- [ ] Code pushed to GitHub or Git repository
- [ ] Platform account created (Render/Railway/GCP/Azure)

After deploying:

- [ ] Health endpoint returns `"status": "healthy"`
- [ ] Model is loaded: `"model_loaded": true`
- [ ] Single prediction works
- [ ] CSV batch prediction works
- [ ] API docs accessible at `/docs`

---

# 🆘 Troubleshooting

## "Container fails to start"
- Check logs on platform dashboard
- Verify `model.pkl` exists in repo
- Ensure Dockerfile path is correct: `mlops/Dockerfile`

## "Model not loaded"
- Model file must be in container (`model.pkl` in project root)
- Check startup logs for errors
- Verify memory allocation (need 2GB minimum)

## "Port binding error"
- Ensure environment variable: `API_PORT=8000`
- Some platforms (HF) require port 7860 — adjust Dockerfile CMD

## "Import error: model_loader"
- Verify `mlops/__init__.py` exists
- Check import in `app_v3.py`: `from mlops.model_loader import ...`

---

# 📚 Next Steps

After successful deployment:

1. **Custom Domain** — Point your domain to the deployed URL
2. **CI/CD** — Auto-deploy on git push
3. **Monitoring** — Set up Prometheus/Grafana
4. **Load Testing** — Test with Apache JMeter
5. **API Keys** — Add authentication
6. **Rate Limiting** — Prevent abuse
7. **Caching** — Cache predictions for performance

---

**Need help?** See [mlops/DOCKER_CLOUD_DEPLOYMENT.md](mlops/DOCKER_CLOUD_DEPLOYMENT.md) for detailed platform-specific guides.

**Status**: Ready to deploy! 🚀
**Date**: February 22, 2026
