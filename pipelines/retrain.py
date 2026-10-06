# Retrain pipeline (Prefect DAG)
#   monitor -> ตัดสินใจ -> สร้าง data version ใหม่ (ข้อมูลเดิม + ข้อมูล production ที่มี label แล้ว)
#   -> validate -> train -> gate -> promote
#
# นโยบายการเทรนใหม่
#   1. concept drift (ROC-AUC ตก >= 0.05 และมี label >= 300)   -> เทรนใหม่ทันที
#   2. data drift (ฟีเจอร์ >= 20% มี PSI >= 0.2)               -> เทรนเมื่อได้ label ครบ 300
#   3. ไม่มี drift                                              -> เทรนตามรอบทุก 30 วัน (scheduler.py)
#   ไม่ว่ากรณีไหน โมเดลใหม่ต้องผ่าน gate ก่อนถึงจะขึ้นใช้งาน
from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd
from prefect import flow, get_run_logger, task

from readmit.alerts import send_alert
from readmit.config import file_sha256, load_params, path
from readmit.data.schema import validate
from readmit.data.split import TARGET
from readmit.models import registry, train
from readmit.monitoring import monitor


@task
def check_drift() -> dict:
    return monitor.run()


@task
def build_dataset_version():
    # เอาข้อมูล production ที่มี label แล้ว แบ่ง 70% ไปเพิ่มใน train, 30% เพิ่มใน test
    # (test ต้องมีข้อมูลใหม่ด้วย ไม่งั้น gate จะวัดแต่กับโลกเก่า)
    params = load_params()
    proc = path(params["data"]["processed_dir"])
    _, labeled = monitor.load_production(10**9)  # เอาทั้งหมด
    if len(labeled) < params["monitoring"]["min_labels"]:
        return None  # label ยังมาไม่ครบ รอก่อน

    new = labeled.drop(columns=["score", "model_version"]).rename(columns={"readmitted_30d": TARGET})
    new[TARGET] = new[TARGET].astype(int)

    # ข้อมูลใหม่ต้องผ่าน schema เดียวกับข้อมูลเทรน (แปลงกลับเป็นรูปแบบไฟล์ดิบก่อน)
    check = new.drop(columns=[TARGET])
    check["readmitted"] = new[TARGET].map({1: "<30", 0: "NO"})
    if "patient_nbr" not in check.columns:
        check["patient_nbr"] = check["encounter_id"]
    check["encounter_id"] = check["encounter_id"].astype(int)
    check["patient_nbr"] = check["patient_nbr"].astype(int)
    relaxed = {**params, "data": {**params["data"], "min_rows": 1}}  # ไม่ต้องเช็คจำนวนแถวขั้นต่ำ
    validate(check, relaxed)

    new = new.sample(frac=1, random_state=params["seed"])
    cut = int(len(new) * 0.7)
    meta = json.loads((proc / "meta.json").read_text())

    for name, extra in [("train", new.iloc[:cut]), ("test", new.iloc[cut:])]:
        old = pd.read_parquet(proc / f"{name}.parquet")
        keep_cols = [c for c in old.columns if c in extra.columns]
        both = pd.concat([old, extra[keep_cols]], ignore_index=True)
        both = both.drop_duplicates("encounter_id", keep="last")
        both.to_parquet(proc / f"{name}.parquet", index=False)
        meta["splits"][name] = {
            "rows": len(both),
            "pos_rate": round(float(both[TARGET].mean()), 4),
            "sha256": file_sha256(proc / f"{name}.parquet"),
        }

    # data version ใหม่ (จำตัวเดิมไว้ใน parent_data_version)
    meta["parent_data_version"] = meta["data_version"]
    meta["data_version"] = hashlib.sha256(json.dumps(meta["splits"], sort_keys=True).encode()).hexdigest()[:12]
    (proc / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


@flow(name="readmission-retrain-pipeline", log_prints=True)
def retrain_pipeline(force: bool = False) -> dict:
    log = get_run_logger()
    rep = check_drift()

    if not force and not rep.get("retrain_recommended"):
        log.info("No retrain needed: data_drift=%s concept_drift=%s", rep.get("data_drift"), rep.get("concept_drift"))
        return {"retrained": False, "monitoring": rep.get("status")}

    reason = "forced" if force else rep["reason"]
    meta = build_dataset_version()
    if meta is None:
        send_alert("RETRAIN POSTPONED: waiting for delayed labels", {"reason": reason})
        return {"retrained": False, "reason": "awaiting_labels"}

    summary = task(train.run)(trigger=reason)
    result = task(registry.register_and_gate)(summary)
    log.info("retrain (%s) on data %s -> %s", reason, meta["data_version"], result)
    return {"retrained": True, "reason": reason, **result}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="เทรนใหม่เลยไม่ต้องรอ drift")
    print(retrain_pipeline(force=ap.parse_args().force))
