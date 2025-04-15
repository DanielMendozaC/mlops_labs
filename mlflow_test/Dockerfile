FROM python:3.9-slim

WORKDIR /app

# Install a specific version of MLflow
RUN pip install mlflow==2.8.1

# Expose the port MLflow will run on
EXPOSE 5000

# Command to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]