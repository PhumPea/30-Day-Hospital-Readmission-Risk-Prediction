# ไฟล์นี้รวมฟังก์ชันที่ใช้ร่วมกันทั้งโปรเจค เช่น อ่าน params.yaml, หา path, ตั้ง seed
from __future__ import annotations

import hashlib
import os
import random
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

# root ของโปรเจค (ใน docker จะตั้ง PROJECT_ROOT=/app ไว้)
ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


@lru_cache
def load_params(path: str | None = None) -> dict:
    # อ่านค่าตั้งค่าทั้งหมดจาก params.yaml (cache ไว้ จะได้ไม่ต้องอ่านไฟล์ซ้ำ)
    p = Path(path or os.getenv("PARAMS_FILE", ROOT / "params.yaml"))
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(rel: str) -> Path:
    # แปลง path แบบ relative ให้อิงจาก root ของโปรเจค
    p = Path(rel)
    if p.is_absolute():
        return p
    return ROOT / p


def set_seed(seed: int) -> None:
    # ล็อค seed ทุกที่ เพื่อให้รันซ้ำแล้วได้ผลเหมือนเดิม
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_sha256(p: Path) -> str:
    # hash ไฟล์ เอาไว้ทำ data version
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    # เอา commit ปัจจุบันไว้บันทึกเป็น code version
    # ใน docker ไม่มีโฟลเดอร์ .git เลยส่งผ่าน env GIT_COMMIT แทน
    if os.getenv("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"
