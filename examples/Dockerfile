# Reference pipeline image: one container that crawls then classifies, run on a
# schedule (cron, ECS scheduled task, Kubernetes CronJob).
#
# Build from the directory holding both your config package and this file:
#   docker build -f examples/Dockerfile -t research-digest-pipeline .
FROM python:3.12-slim

# curl_cffi (used to get past Cloudflare on some feeds) needs libcurl.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the engine first so this layer caches independently of your taxonomy,
# which will change far more often.
RUN pip install --no-cache-dir \
    'research-digest[gemini,aws] @ git+https://github.com/dmoskov/research-digest@v0.1.1'

# Your configuration package — the module DIGEST_CONFIG points at.
COPY examples/ examples/

ENV DIGEST_CONFIG=examples.digest_config:CONFIG \
    PYTHONPATH=/app \
    DAYS_BACK=30 \
    WORKERS=8

COPY examples/run-pipeline.sh /app/run-pipeline.sh
RUN chmod +x /app/run-pipeline.sh

CMD ["/app/run-pipeline.sh"]
