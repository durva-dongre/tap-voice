# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm

ARG KOKORO_REV
ARG INDIC_REV
ARG INDIC_DTYPE=float16
ENV INDIC_DTYPE=${INDIC_DTYPE}
COPY build/fetch_weights.py build/
RUN --mount=type=secret,id=hf HF_HUB_OFFLINE=0 HF_TOKEN="$(cat /run/secrets/hf 2>/dev/null)" \
    KOKORO_REV=$KOKORO_REV INDIC_REV=$INDIC_REV python build/fetch_weights.py

COPY app/ app/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]