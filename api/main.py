# Model server (FastAPI)
#   POST /predict        ทำนายทีละราย (real-time ตอนจำหน่ายคนไข้)
#   POST /predict/batch  ทำนายทีละหลายราย (รายชื่อคนไข้กลับบ้านประจำวัน)
#   POST /feedback       รับ label จริงที่มาทีหลัง ใช้ตรวจ concept drift
#   GET  /health         process ยังทำงานอยู่ไหม (liveness)
#   GET  /ready          โหลดโมเดลเสร็จแล้วหรือยัง (readiness)
#   GET  /metrics        ให้ Prometheus มาเก็บค่า
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api.model_store import ModelStore
from api.schemas import BatchRequest, Encounter, Feedback, Prediction
from readmit.config import path
from readmit.features.preprocess import INPUT_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

# ----- metrics ที่ส่งให้ Prometheus -----
REQS = Counter("api_requests_total", "Requests", ["endpoint", "status"])
LAT = Histogram("api_request_latency_seconds", "Latency", ["endpoint"],
                buckets=(.005, .01, .025, .05, .1, .2, .3, .5, 1, 2.5))
INVALID = Counter("api_invalid_input_total", "Rejected requests (schema violations)")
SCORE = Histogram("prediction_risk_score", "Predicted risk", buckets=[i / 20 for i in range(21)])
FOLLOWUP = Counter("prediction_follow_up_total", "Encounters flagged for follow-up")
MODEL_INFO = Gauge("model_version_info", "Loaded model version", ["version"])
LABELS = Counter("feedback_labels_total", "Delayed ground-truth labels received", ["label"])
DRIFT = Gauge("monitoring_value", "Latest monitoring job outputs", ["metric"])

store = ModelStore()
_log_lock = Lock()
PRED_LOG = path(os.getenv("PRED_LOG", "logs/predictions.jsonl"))
LABEL_LOG = path(os.getenv("LABEL_LOG", "logs/labels.jsonl"))


def append_jsonl(f, records: list) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, open(f, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # ตอนเปิด server: โหลดโมเดล แล้วเริ่ม thread คอยเช็ค champion ใหม่
    store.refresh()
    store.start_polling()
    yield


app = FastAPI(title="30-Day Readmission Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    # จับเวลา + นับ request ทุกอัน (ยกเว้น /metrics เอง)
    start = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path
    if endpoint != "/metrics":
        LAT.labels(endpoint).observe(time.perf_counter() - start)
        REQS.labels(endpoint, str(response.status_code)).inc()
    return response


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    # ข้อมูลไม่ผ่าน schema -> นับไว้ (ถ้าพุ่งขึ้นแปลว่าระบบต้นทางมีปัญหา) แล้วตอบ 422
    INVALID.inc()
    log.warning("invalid input on %s: %s", request.url.path, exc.errors()[:3])
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def score(encounters: list) -> list:
    if not store.ready:
        raise HTTPException(503, "model not loaded")

    rows = [e.to_row() for e in encounters]
    X = pd.DataFrame(rows)[INPUT_COLS]

    # ก๊อปปี้ reference ไว้ก่อน กันกรณีโมเดลถูกสลับกลางทาง
    with store.lock:
        model = store.model
        version = store.version
        threshold = store.threshold

    # โมเดลเป็น Pipeline ที่รวม preprocessing แล้ว ส่งข้อมูลดิบเข้าไปได้เลย
    probs = model.predict_proba(X)[:, 1]

    now = datetime.now(timezone.utc).isoformat()
    preds = []
    logs = []
    for row, p in zip(rows, probs, strict=True):
        follow_up = bool(p >= threshold)
        SCORE.observe(float(p))
        if follow_up:
            FOLLOWUP.inc()
        preds.append(Prediction(encounter_id=row["encounter_id"], risk_score=round(float(p), 5),
                                follow_up=follow_up, model_version=version))
        # เก็บ input + output ไว้ให้ monitoring เอาไปตรวจ drift
        logs.append({"ts": now, "request_id": str(uuid.uuid4()), "model_version": version,
                     "score": float(p), "follow_up": follow_up, "features": row})
    append_jsonl(PRED_LOG, logs)
    return preds


@app.post("/predict", response_model=Prediction)
def predict(enc: Encounter):
    return score([enc])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    return score(req.encounters)


@app.post("/feedback")
def feedback(fb: Feedback):
    LABELS.labels(str(int(fb.readmitted_30d))).inc()
    record = {"ts": datetime.now(timezone.utc).isoformat(), **fb.model_dump()}
    append_jsonl(LABEL_LOG, [record])
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not store.ready:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model_version": store.version}


@app.get("/model")
def model_info():
    return {"name": store.name, "alias": store.alias, "version": store.version, "threshold": store.threshold}


@app.post("/admin/reload")
def reload_model():
    # บังคับโหลดใหม่ทันที (ไม่ต้องรอ 30 วิ)
    store.refresh()
    return model_info()


@app.get("/metrics")
def metrics():
    MODEL_INFO.clear()
    if store.version:
        MODEL_INFO.labels(store.version).set(1)

    # ผล drift มาจาก monitor job (คนละ process) อ่านจากไฟล์แล้วส่งต่อให้ Prometheus
    f = path("reports/monitoring.json")
    if f.exists():
        try:
            gauges = json.loads(f.read_text()).get("gauges", {})
            for k, v in gauges.items():
                DRIFT.labels(k).set(float(v))
        except Exception as e:
            log.warning("cannot read monitoring report: %s", e)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
