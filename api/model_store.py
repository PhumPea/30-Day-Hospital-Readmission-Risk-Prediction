# โหลดโมเดล champion จาก MLflow registry
# มี thread คอยเช็คทุก 30 วิ ถ้า champion เปลี่ยน (promote ใหม่ / rollback) จะโหลดตัวใหม่เองเลย
# ไม่ต้อง restart container
from __future__ import annotations

import logging
import os
import threading
import time

import mlflow
from mlflow import MlflowClient

from readmit.config import load_params

log = logging.getLogger("model_store")


class ModelStore:
    def __init__(self):
        cfg = load_params()["registry"]
        self.name = cfg["model_name"]
        self.alias = cfg["champion_alias"]
        self.model = None
        self.version = None
        self.threshold = 0.5
        self.lock = threading.Lock()  # กันไม่ให้สลับโมเดลระหว่างที่ request อื่นใช้อยู่

    def refresh(self) -> bool:
        # คืน True ถ้ามีโมเดลพร้อมใช้
        try:
            mv = MlflowClient().get_model_version_by_alias(self.name, self.alias)
        except Exception as e:
            log.warning("no champion yet: %s", e)
            return False

        if mv.version == self.version:
            return True  # ตัวเดิม ไม่ต้องโหลดใหม่

        model = mlflow.sklearn.load_model(f"models:/{self.name}/{mv.version}")
        with self.lock:
            self.model = model
            self.version = str(mv.version)
            self.threshold = float(mv.tags.get("threshold", 0.5))
        log.info("loaded %s v%s (threshold=%.4f)", self.name, mv.version, self.threshold)
        return True

    def start_polling(self, every: int | None = None) -> None:
        if every is None:
            every = int(os.getenv("MODEL_POLL_SECONDS", "30"))

        def loop():
            while True:
                try:
                    self.refresh()
                except Exception as e:
                    log.error("refresh failed: %s", e)
                time.sleep(every)

        threading.Thread(target=loop, daemon=True).start()

    @property
    def ready(self) -> bool:
        return self.model is not None
