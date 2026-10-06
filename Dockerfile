# ใช้ image เดียวกันทุก service (pipeline, api, monitor, mlflow, prefect)
# ไลบรารีเวอร์ชันเดียวกันหมด -> รันซ้ำได้ผลเหมือนเดิม และไม่เกิด skew ระหว่างเทรนกับ serve
FROM python:3.12.3-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app PYTHONHASHSEED=42 \
    PREFECT_SERVER_ANALYTICS_ENABLED=false DO_NOT_TRACK=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# ลง dependency ก่อน copy โค้ด จะได้ใช้ cache ตอน build ใหม่ (แก้โค้ดไม่ต้องลงใหม่)
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

COPY pyproject.toml params.yaml ./
COPY src ./src
RUN pip install --no-deps -e .
COPY api ./api
COPY pipelines ./pipelines
COPY scripts ./scripts
COPY tests ./tests
COPY data/bad_samples ./data/bad_samples

# ใน container ไม่มี .git เลยส่ง commit เข้ามาตอน build แทน
ARG GIT_COMMIT=unknown
ARG IMAGE_TAG=readmit:local
ENV GIT_COMMIT=${GIT_COMMIT} IMAGE_TAG=${IMAGE_TAG}

RUN mkdir -p data/raw data/processed reports logs

EXPOSE 8000
# ใช้ 1 worker เพราะ counter ของ prometheus_client อยู่ใน process ถ้าหลาย worker ตัวเลขจะเพี้ยน
# ถ้าอยากรับโหลดมากขึ้นให้เพิ่ม replica แทน
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
