# Nous server image. The model weights are NOT baked in: mount them at /models (read-only).
#   docker build -f deploy/Dockerfile -t nous:0.1.0 .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /opt/nous
COPY pyproject.toml README.md ./
COPY nous ./nous
COPY packs ./packs
RUN pip install --no-cache-dir "transformers>=4.51,<5" "fastapi>=0.110" "uvicorn[standard]>=0.29" "httpx>=0.27" "pyyaml>=6" \
 && pip install --no-cache-dir --no-deps .
RUN useradd -m -u 10001 nous && mkdir -p /data && chown nous /data
USER nous
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
ENTRYPOINT ["nous"]
CMD ["serve", "--config", "/etc/nous/nous.yaml"]
