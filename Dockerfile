FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .


RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

CMD ["tail", "-f", "/dev/null"]