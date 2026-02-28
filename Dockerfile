# Use a lightweight official Python image
FROM python:3.11-slim

# Install system dependencies
# Added chromium, chromium-driver, and fonts-liberation for screenshot capability
RUN apt-get update && apt-get install -y \
    gcc \
    chromium \
    chromium-driver \
    fonts-liberation \
    libnss3 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy and set up entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create directories (volumes will be mounted at runtime)
RUN mkdir -p /app/chroma_db /app/static/documents && \
    chmod -R 777 /app/chroma_db /app/static /app

# Expose port
EXPOSE 2000

# Use entrypoint to fix permissions after volume mount, then start app
ENTRYPOINT ["/entrypoint.sh"]
