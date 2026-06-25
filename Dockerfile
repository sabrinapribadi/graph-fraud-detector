# Use Python 3.12 slim image (lightweight)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Copy dependency files first (for better caching)
COPY pyproject.toml poetry.lock ./

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app/src

# Set environment variables
ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "src/ui/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]