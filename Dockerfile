# ============================================================
# Stage 1: Build — 의존성 설치 전용 (캐시 레이어 극대화)
# ============================================================
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# Stage 2: Runtime — 최소한의 런타임만 포함
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="your-email@example.com"
LABEL description="Face Personality Prediction API"

# MediaPipe/OpenCV 런타임 의존성만 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge -y --auto-remove

# builder 스테이지에서 설치된 Python 패키지만 복사
COPY --from=builder /install /usr/local

# 보안: root가 아닌 전용 유저로 실행
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# 앱 코드 복사 (불필요 파일은 .dockerignore로 제외)
COPY app/ ./app/
RUN mkdir -p uploads && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
