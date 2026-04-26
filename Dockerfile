FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY . .

EXPOSE 7860

CMD ["uv", "run", "server"]
