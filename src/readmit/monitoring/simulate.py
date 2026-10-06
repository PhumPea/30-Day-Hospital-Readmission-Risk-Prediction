# จำลอง traffic จริงยิงเข้า API (ใช้ตอนเดโม)
#
#   normal        : คนไข้จาก test set ตามปกติ
#   data_drift    : กลุ่มคนไข้เปลี่ยน (แก่ขึ้น ป่วยหนักขึ้น เข้าแบบฉุกเฉินเยอะขึ้น)    -> P(X) เปลี่ยน
#   concept_drift : คนไข้หน้าตาเหมือนเดิม แต่กลไกการกลับมาซ้ำเปลี่ยน                -> P(y|X) เปลี่ยน
#                   สมมติว่า รพ. เริ่มโครงการติดตามคนไข้ที่เคยนอนบ่อย (คนกลุ่มนี้เลยกลับมาน้อยลง)
#                   และมีโรคระบบทางเดินหายใจระบาด (กลุ่มนี้กลับมามากขึ้น)
#   bad_data      : ส่ง request เสีย ๆ ต้องโดนปฏิเสธ 422
from __future__ import annotations

import argparse
import json

import httpx
import numpy as np
import pandas as pd

from readmit.config import load_params, path
from readmit.data.split import TARGET
from readmit.features.preprocess import INPUT_COLS, icd9_group

AGES = [f"[{a}-{a + 10})" for a in range(0, 100, 10)]


def make_batch(scenario: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    proc = path(load_params()["data"]["processed_dir"])
    df = pd.read_parquet(proc / "test.parquet").sample(n, replace=True, random_state=seed).reset_index(drop=True)
    df["encounter_id"] = rng.integers(10**9, 2 * 10**9, n)  # ให้เป็น encounter ใหม่ ไม่ซ้ำของเดิม

    if scenario == "data_drift":
        age_idx = df["age"].map(AGES.index)
        df["age"] = [AGES[min(i + 2, 9)] for i in age_idx]  # แก่ขึ้น 20 ปี
        df["num_medications"] = np.clip(df["num_medications"] + rng.integers(5, 15, n), 0, 100)
        df["number_inpatient"] = np.clip(df["number_inpatient"] + rng.poisson(2, n), 0, 30)
        df["admission_type_id"] = rng.choice([1, 2], n, p=[0.9, 0.1])
        df["time_in_hospital"] = np.clip(df["time_in_hospital"] + 3, 1, 14)

    elif scenario == "concept_drift":
        # X เหมือนเดิม แต่สร้าง label ใหม่จากกติกาใหม่
        is_resp = (df["diag_1"].map(icd9_group) == "respiratory") | (df["diag_2"].map(icd9_group) == "respiratory")
        logit = (-2.6
                 - 0.8 * df["number_inpatient"].clip(upper=3)
                 + 1.4 * (df["num_procedures"] >= 3)
                 + 1.2 * is_resp
                 + 1.0 * (df["number_inpatient"] == 0))
        df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return df


def run(api: str, scenario: str, n: int, seed: int, send_labels: bool = True) -> dict:
    client = httpx.Client(base_url=api, timeout=30)

    if scenario == "bad_data":
        bad = json.loads(path("data/bad_samples/bad_requests.json").read_text())
        codes = [client.post("/predict", json=b).status_code for b in bad]
        return {"scenario": scenario, "sent": len(bad), "status_codes": codes}

    df = make_batch(scenario, n, seed)
    cols = INPUT_COLS + ["encounter_id"]
    sent = 0
    for i in range(0, n, 200):  # ส่งทีละ 200 รายผ่าน batch endpoint
        chunk = json.loads(df.iloc[i:i + 200][cols].to_json(orient="records"))
        r = client.post("/predict/batch", json={"encounters": chunk})
        r.raise_for_status()
        sent += len(chunk)

    # ในความจริง label จะมาหลังจากนี้ 30 วัน ตรงนี้จำลองว่ามาถึงแล้ว
    if send_labels:
        for eid, y in zip(df["encounter_id"], df[TARGET], strict=True):
            client.post("/feedback", json={"encounter_id": int(eid), "readmitted_30d": bool(y)})

    return {"scenario": scenario, "sent": sent, "labels": send_labels, "pos_rate": float(df[TARGET].mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--scenario", choices=["normal", "data_drift", "concept_drift", "bad_data"], default="normal")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-labels", action="store_true", help="ไม่ส่ง label (จำลองว่ายังไม่ครบ 30 วัน)")
    args = ap.parse_args()
    print(run(args.api, args.scenario, args.n, args.seed, not args.no_labels))
