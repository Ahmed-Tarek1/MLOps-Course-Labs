FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY main.py .

EXPOSE 8000

CMD ["uv", "run", "litestar", "--app", "main:app", "run", "--host", "0.0.0.0"]