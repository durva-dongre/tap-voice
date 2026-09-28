# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1

FROM base AS models
ARG SVARA_REVISION=main
ARG SNAC_REVISION=main
ARG PREQUANTIZED_REPO=
ARG PREQUANTIZED_REVISION=main
ARG CALIBRATE=0
ENV MODELS_DIR=/opt/models RAW_DIR=/opt/raw HF_HOME=/opt/hf
WORKDIR /work
COPY requirements-build.txt .
RUN pip install -r requirements-build.txt
COPY build/fetch_weights.py build/quantize_fp8.py build/
RUN --mount=type=secret,id=hf \
    export HF_TOKEN="$(cat /run/secrets/hf 2>/dev/null || true)" \
    && python build/fetch_weights.py \
    && python build/quantize_fp8.py \
    && rm -rf /opt/raw /opt/hf

FROM base AS runtime
ARG VLLM_VERSION
ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    MODELS_DIR=/opt/models MODEL_DIR=/opt/models/svara-fp8 SNAC_DIR=/opt/models/snac_24khz \
    HOME=/home/runner
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 10001 -m runner \
    && mkdir -p /opt/hf && chown 10001:10001 /opt/hf
WORKDIR /app
COPY requirements.txt .
RUN pip install "vllm==${VLLM_VERSION}" -r requirements.txt
COPY --from=models /opt/models /opt/models
COPY app/ app/
COPY app/entrypoint.sh entrypoint.sh
COPY build/verify_image.py build/
RUN chmod +x entrypoint.sh
USER 10001
RUN python build/verify_image.py
ENTRYPOINT ["./entrypoint.sh"]