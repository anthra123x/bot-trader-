FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Install curl for container healthcheck and basic system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into container
COPY . .

# Install dependencies and the project
RUN pip install --no-cache-dir .

# Expose default HTTP port (overridden dynamically by cloud platforms via $PORT)
EXPOSE 8000

# Healthcheck for orchestration platforms
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run the 24/7 autonomous trading server and web dashboard
CMD ["python", "server.py"]
