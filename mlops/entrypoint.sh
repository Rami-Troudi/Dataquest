#!/bin/bash
# Docker entrypoint script
# Starts the MLOps API with model pre-loaded

set -e

echo "🚀 Starting MLOps Insurance Model API..."
echo "📦 Python version: $(python --version)"
echo "📦 Installed packages:"
pip list | grep -E "fastapi|uvicorn|scikit-learn|pandas|joblib" || true

echo "-" "---"
echo "🔧 Starting uvicorn server..."

# Run the FastAPI app with uvicorn
# The model will be loaded on startup via app_v3.on_event("startup")
exec python -m uvicorn mlops.app_v3:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  --access-log
