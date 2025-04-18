FROM python:3.12-slim

WORKDIR /app

# Copy only what we need
COPY requirements.txt server.sh ./

# Install MLflow + drivers
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Make sure server.sh is Unix‑line endings and executable
RUN apt-get update && apt-get install -y dos2unix \
 && dos2unix server.sh \
 && chmod +x server.sh \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Let Cloud Run inject $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080

# And run our launcher script
ENTRYPOINT ["./server.sh"]
