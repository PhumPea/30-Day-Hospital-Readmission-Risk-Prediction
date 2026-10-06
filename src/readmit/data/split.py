# สร้าง label และแบ่ง train / val / test
from __future__ import annotations

import hashlib
import json

import pandas as pd

from readmit.config import file_sha256, load_params, path
from readmit.data.schema import read_raw

TARGET = "target"


def make_target(df: pd.DataFrame, exclude_discharge: list[int]) -> pd.DataFrame:
    # ตัดคนที่เสียชีวิต / ไป hospice ออก เพราะยังไงก็กลับมารักษาซ้ำไม่ได้ จะทำให้โมเดลเพี้ยน
    df = df[~df["discharge_disposition_id"].isin(exclude_discharge)].copy()
    # label = 1 ถ้ากลับมาภายใน 30 วัน (">30" กับ "NO" นับเป็น 0)
    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns=["readmitted"])


def _bucket(patient_id, seed: int) -> float:
    # hash เลขผู้ป่วยให้เป็นเลข 0-1
    # คนเดิมได้เลขเดิมเสมอ -> อยู่ชุดเดิมทุกครั้ง ไม่ว่าแถวจะเรียงยังไง
    h = hashlib.md5(f"{seed}-{patient_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    # แบ่งตามผู้ป่วย ไม่ใช่ตามแถว
    # ถ้าแบ่งตามแถว คนไข้คนเดียวกันอาจไปโผล่ทั้ง train และ test = ข้อมูลรั่ว ผลสอบจะดีเกินจริง
    b = df["patient_nbr"].map(lambda pid: _bucket(pid, seed))
    test = df[b < test_size]
    val = df[(b >= test_size) & (b < test_size + val_size)]
    train = df[b >= test_size + val_size]
    return train, val, test


def run(raw_file=None) -> dict:
    params = load_params()
    if raw_file is None:
        raw_file = path(params["data"]["raw_file"])

    df = make_target(read_raw(raw_file), params["data"]["exclude_discharge_ids"])
    train, val, test = split(df, params["split"]["test_size"], params["split"]["val_size"], params["seed"])

    out = path(params["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    meta = {"raw_sha256": file_sha256(raw_file), "splits": {}}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        f = out / f"{name}.parquet"
        part.to_parquet(f, index=False)
        meta["splits"][name] = {
            "rows": len(part),
            "pos_rate": round(float(part[TARGET].mean()), 4),
            "sha256": file_sha256(f),
        }

    # data version = hash(ไฟล์ดิบ + วิธีแบ่ง + seed) ถ้าอะไรเปลี่ยนเลขนี้ก็เปลี่ยน
    key = meta["raw_sha256"] + json.dumps(params["split"]) + str(params["seed"])
    meta["data_version"] = hashlib.sha256(key.encode()).hexdigest()[:12]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
