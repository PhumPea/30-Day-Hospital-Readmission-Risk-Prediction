# Training pipeline (Prefect DAG)
#   ingest -> validate -> split -> train -> gate + register -> promote
#
# รัน:
#   python -m pipelines.flow              ข้อมูลจริงจาก UCI
#   python -m pipelines.flow --synthetic  ข้อมูลจำลอง (ไม่ต้องใช้เน็ต)
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from prefect import flow, get_run_logger, task

from readmit.config import load_params, path
from readmit.data import split as split_mod
from readmit.data.schema import read_raw, validate
from readmit.models import registry, train


@task(retries=2, retry_delay_seconds=10)  # เน็ตหลุดตอนโหลดให้ลองใหม่ 2 ครั้ง
def ingest(synthetic: bool) -> str:
    cmd = [sys.executable, "-m", "readmit.data.ingest"]
    if synthetic:
        cmd.append("--synthetic")
    subprocess.run(cmd, check=True)
    return str(path(load_params()["data"]["raw_file"]))


@task
def validate_data(raw_file: str) -> dict:
    # ถ้าข้อมูลเสีย ตรงนี้จะ raise -> task ที่เหลือไม่ได้รัน + มี alert
    return validate(read_raw(raw_file))


@task
def split_data(raw_file: str, _validated: dict) -> dict:
    # รับผล validate เข้ามาด้วย เพื่อให้ Prefect รู้ว่าต้องรอ validate ก่อน
    return split_mod.run(raw_file)


@task
def train_models(_meta: dict, trigger: str, extra_train=None) -> dict:
    return train.run(extra_train=extra_train, trigger=trigger)


@task
def gate_and_register(summary: dict) -> dict:
    return registry.register_and_gate(summary)


@flow(name="readmission-training-pipeline", log_prints=True)
def training_pipeline(synthetic: bool = False, trigger: str = "manual", extra_train=None) -> dict:
    log = get_run_logger()
    raw = ingest(synthetic)
    report = validate_data(raw)
    meta = split_data(raw, report)
    summary = train_models(meta, trigger, extra_train)
    result = gate_and_register(summary)
    log.info("Pipeline result: %s", json.dumps(result, default=str))

    # ถ้ายังไม่มี champion เลย (โมเดลแรกไม่ผ่าน gate) API จะไม่มีอะไรให้ใช้
    # ให้ pipeline fail ไปเลยพร้อมบอกเหตุผล ดีกว่าปล่อยให้ API ค้าง unhealthy แบบงง ๆ
    if registry.current_version(load_params()["registry"]["champion_alias"]) is None:
        failed = [k for k, ok in result["checks"].items() if not ok]
        raise RuntimeError(f"No champion model: v{result['version']} rejected by gate {failed}. "
                           "See reports/gate.json")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    training_pipeline(synthetic=args.synthetic)
