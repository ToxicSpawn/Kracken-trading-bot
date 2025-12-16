FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=botuser:botuser . .

# Create directories for models and data
RUN mkdir -p /app/models /app/rag_store /app/logs /app/data && \
    chown -R botuser:botuser /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    BOT_BASE_DIR=/app \
    PYTHONPATH=/app

# Expose ports
# 8000: FastAPI inference server
# 7860: Gradio UI
# 8001: Prometheus metrics
# 8080: Dashboard
EXPOSE 8000 7860 8001 8080

# Switch to non-root user
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "bot.py"]

