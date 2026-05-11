# Multi-stage Dockerfile. Builder installs CPU PyTorch and the package;
# the runtime stage carries only what's needed to run the CLI.

FROM python:3.11-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY README.md ./

RUN python -m pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2 \
 && pip install --no-deps . \
 && pip install click psutil "numpy<2" "onnx>=1.15,<1.17" "onnxruntime>=1.17,<1.19"

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Copy the installed packages and console scripts.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/quant-explorer /usr/local/bin/quant-explorer
COPY artifacts /app/artifacts

ENTRYPOINT ["quant-explorer"]
CMD ["--help"]
