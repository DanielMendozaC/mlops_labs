FROM python:3.8-slim

# Install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python libs
RUN pip install --no-cache-dir \
        pandas==1.3.5 \
        scikit-learn==1.0.2 \
        mlflow==1.26.1 \
        google-cloud-storage \
        google-auth \
        google-cloud-secret-manager \
        metaflow

WORKDIR /app
COPY metaflow_gcp_flow.py .

ENTRYPOINT ["python", "metaflow_gcp_flow.py"]
