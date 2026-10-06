# service "monitor" ใน docker compose
# วนรัน retrain_pipeline ทุก MONITOR_INTERVAL_SECONDS (default 5 นาที)
# และบังคับเทรนใหม่ทุก RETRAIN_EVERY_DAYS (default 30 วัน)
from __future__ import annotations

import logging
import os
import time

from pipelines.retrain import retrain_pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scheduler")

INTERVAL = int(os.getenv("MONITOR_INTERVAL_SECONDS", "300"))
FORCE_EVERY = int(os.getenv("RETRAIN_EVERY_DAYS", "30")) * 86400

if __name__ == "__main__":
    last_forced = time.time()
    while True:
        force = (time.time() - last_forced) >= FORCE_EVERY
        try:
            result = retrain_pipeline(force=force)
            log.info("cycle result: %s", result)
            if force:
                last_forced = time.time()
        except Exception as e:
            # รอบนี้พังก็ไม่เป็นไร รอบหน้าลองใหม่ (ดู error ได้ใน Prefect UI)
            log.exception("monitor cycle failed: %s", e)
        time.sleep(INTERVAL)
