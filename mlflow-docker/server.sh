#!/bin/bash
set -e

mlflow db upgrade "$POSTGRESQL_URL"

exec mlflow server \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --backend-store-uri "$POSTGRESQL_URL" \
  --artifacts-destination "$ARTIFACT_URL"
