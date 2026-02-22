# Multi-stage build for optimized Docker image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY mlops/requirements-docker.txt .
RUN pip install --user --no-cache-dir -r requirements-docker.txt


# Final production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code and model
COPY mlops/app_v3.py mlops/
COPY mlops/model_loader.py mlops/
COPY mlops/__init__.py mlops/
COPY model.pkl .

# Make entrypoint executable
COPY mlops/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Health check (using curl instead of requests)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["/app/entrypoint.sh"]
