FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for LightGBM & XGBoost
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY app ./app
COPY src ./src

# Expose port
EXPOSE 8010

# Start FastAPI server
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]