FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy project files
COPY solution.py .
COPY explainability.py .
COPY model.pkl .
COPY api/ api/
COPY frontend/ frontend/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
