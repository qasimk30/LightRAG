# Use Python 3.11 slim image as base
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
COPY LightRAG/extensions/app/private_ingestion.py .
COPY LightRAG/extensions/ ./extensions/
COPY LightRAG/lightrag/ ./lightrag/

# Create necessary directories
RUN mkdir -p /app/private_documents \
    /app/screenshots \
    /app/storage \
    /app/temp

# Set environment variables with defaults
ENV WORKING_DIR=/app/private_documents
ENV SCREENSHOTS_PATH=/app/screenshots
ENV STORAGE_PATH=/app/storage
ENV SERVICE_PORT=8001
ENV MAX_FILE_SIZE_MB=100
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Command to run the application
CMD ["python", "private_ingestion.py"]