FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 chrony \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir .
USER 65532:65532
ENTRYPOINT ["timepredict"]
