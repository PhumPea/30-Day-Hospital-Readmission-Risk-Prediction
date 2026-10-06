# ระบบแจ้งเตือนแบบง่าย ๆ
# - เขียนลง logs/alerts.jsonl ทุกครั้ง
# - ถ้าตั้ง ALERT_WEBHOOK_URL ไว้ (Discord/Slack) จะยิงไปด้วย
from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone

from readmit.config import path

log = logging.getLogger("alerts")


def send_alert(title: str, payload: dict | None = None) -> None:
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "payload": payload or {},
    }
    log.error("ALERT: %s", title)

    f = path("logs/alerts.jsonl")
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    url = os.getenv("ALERT_WEBHOOK_URL")
    if not url:
        return
    try:
        detail = json.dumps(payload, default=str)[:1500]  # webhook จำกัดความยาวข้อความ
        body = json.dumps({"content": f"🚨 {title}\n```{detail}```"}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        # ส่งแจ้งเตือนไม่ได้ก็ไม่เป็นไร ห้ามทำให้ pipeline พัง
        log.warning("webhook failed: %s", e)
