# ขั้นตอนแรกของ pipeline: เอาข้อมูลดิบมาไว้ที่ data/raw/
# ปกติจะโหลดชุดข้อมูล Diabetes 130-US hospitals จาก UCI
# ถ้าใส่ --synthetic จะสร้างข้อมูลจำลองที่ column เหมือนของจริง (ใช้ตอนไม่มีเน็ต / ใน CI)
from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from readmit.config import load_params, path

log = logging.getLogger(__name__)

# คอลัมน์ยาเบาหวาน 23 ตัวในชุดข้อมูล
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def download(url: str, out_file) -> None:
    # UCI ให้มาเป็น zip เลยต้องแตกเอาเฉพาะ diabetic_data.csv
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=180) as r:
        content = r.read()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = None
        for n in z.namelist():
            if n.endswith("diabetic_data.csv"):
                name = n
                break
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(z.read(name))


def make_synthetic(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    # สร้างข้อมูลปลอมที่ schema เหมือนของจริง และใส่ความสัมพันธ์กับ label ไว้ให้โมเดลเรียนรู้ได้
    # สัดส่วนแต่ละค่าประมาณเอาจากข้อมูลจริง
    rng = np.random.default_rng(seed)
    ages = np.array([f"[{a}-{a + 10})" for a in range(0, 100, 10)])
    age_idx = rng.choice(10, n, p=[.005, .01, .02, .04, .09, .17, .22, .26, .16, .025])
    inpatient = rng.poisson(0.6, n)
    emergency = rng.poisson(0.2, n)
    los = np.clip(rng.poisson(3.4, n) + 1, 1, 14)
    diag_pool = ["428", "250.83", "410", "414", "486", "491", "584", "V57", "780", "996", "715", "?"]

    df = pd.DataFrame({
        "encounter_id": np.arange(1, n + 1) * 7,
        "patient_nbr": rng.integers(1, int(n * 0.7), n),  # บางคนมาหลายครั้ง
        "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"], n,
                           p=[.74, .19, .02, .01, .015, .025]),
        "gender": rng.choice(["Female", "Male"], n, p=[.54, .46]),
        "age": ages[age_idx],
        "weight": "?",
        "admission_type_id": rng.choice([1, 2, 3, 5, 6], n, p=[.53, .18, .19, .05, .05]),
        "discharge_disposition_id": rng.choice([1, 3, 6, 18, 2, 22, 11], n,
                                               p=[.59, .14, .13, .04, .03, .05, .02]),
        "admission_source_id": rng.choice([7, 1, 17, 4, 6], n, p=[.56, .29, .07, .04, .04]),
        "time_in_hospital": los,
        "payer_code": rng.choice(["?", "MC", "HM", "SP"], n, p=[.4, .32, .14, .14]),
        "medical_specialty": rng.choice(
            ["?", "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
             "Surgery-General"], n, p=[.49, .14, .08, .07, .12, .1]),
        "num_lab_procedures": np.clip(rng.normal(43, 19, n).round(), 1, 132).astype(int),
        "num_procedures": rng.integers(0, 7, n),
        "num_medications": np.clip(rng.normal(16, 8, n).round(), 1, 81).astype(int),
        "number_outpatient": rng.poisson(0.37, n),
        "number_emergency": emergency,
        "number_inpatient": inpatient,
        "diag_1": rng.choice(diag_pool, n),
        "diag_2": rng.choice(diag_pool, n),
        "diag_3": rng.choice(diag_pool, n),
        "number_diagnoses": rng.integers(1, 17, n),
        "max_glu_serum": rng.choice(["None", "Norm", ">200", ">300"], n, p=[.95, .025, .015, .01]),
        "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"], n, p=[.83, .05, .04, .08]),
    })
    for c in MED_COLS:
        df[c] = rng.choice(["No", "Steady", "Up", "Down"], n, p=[.8, .15, .03, .02])
    df["change"] = rng.choice(["No", "Ch"], n, p=[.54, .46])
    df["diabetesMed"] = rng.choice(["Yes", "No"], n, p=[.77, .23])

    # ความเสี่ยง: เคยนอน รพ. บ่อย / มา ER บ่อย / นอนนาน / อายุมาก / หัวใจล้มเหลว (428) / ปรับอินซูลินขึ้น
    # / ถูกส่งต่อไป nursing facility -> เสี่ยงกลับมาซ้ำมากขึ้น
    logit = (-3.3 + 0.7 * inpatient + 0.4 * emergency + 0.08 * los + 0.08 * (age_idx - 6)
             + 0.7 * (df["diag_1"] == "428") + 0.6 * (df["insulin"] == "Up")
             + 0.8 * df["discharge_disposition_id"].isin([3, 22]))
    p = 1 / (1 + np.exp(-logit))
    u = rng.random(n)
    df["readmitted"] = np.where(u < p, "<30", np.where(u < p + 0.35, ">30", "NO"))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="ใช้ข้อมูลจำลองแทนการโหลดจาก UCI")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    params = load_params()
    out = path(params["data"]["raw_file"])
    if args.synthetic:
        out.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic(args.n, params["seed"]).to_csv(out, index=False)
    elif not out.exists():
        # ถ้ามีไฟล์อยู่แล้วไม่ต้องโหลดใหม่
        download(params["data"]["url"], out)
    print(f"raw data ready: {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
