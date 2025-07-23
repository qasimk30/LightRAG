FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY LightRAG/extensions/app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY LightRAG/extensions/app/main.py .
COPY LightRAG/extensions/ ./extensions/
COPY LightRAG/lightrag/ ./lightrag/


# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Expose the port
EXPOSE 8091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8091/health || exit 1

# Run the application
CMD ["python", "main.py"]