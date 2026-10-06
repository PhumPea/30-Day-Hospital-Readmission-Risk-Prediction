# Feature engineering ทั้งหมดอยู่ไฟล์นี้ไฟล์เดียว
# ใช้ทั้งตอนเทรนและตอน serve: clean_raw + ColumnTransformer ถูกรวมเป็น sklearn Pipeline
# แล้วเซฟไปพร้อมโมเดล ฝั่ง API แค่เรียก predict_proba ไม่ได้เขียนการแปลงข้อมูลซ้ำ
# -> ป้องกัน training-serving skew
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ปิด warning เรื่อง downcast ของ pandas ตอน replace("?")
pd.set_option("future.no_silent_downcasting", True)

NUM_COLS = [
    "age_num", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "total_visits", "n_med_changes", "n_meds_active",
]
CAT_COLS = [
    "race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "diag_1_grp", "diag_2_grp", "diag_3_grp", "max_glu_serum", "A1Cresult",
    "insulin", "metformin", "change", "diabetesMed",
]
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
# คอลัมน์ดิบที่โมเดลต้องใช้ = สิ่งที่ API ต้องรับเข้ามา
INPUT_COLS = sorted(set(
    ["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id",
     "time_in_hospital", "medical_specialty", "num_lab_procedures", "num_procedures",
     "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
     "diag_1", "diag_2", "diag_3", "number_diagnoses", "max_glu_serum", "A1Cresult",
     "change", "diabetesMed"] + MED_COLS))

# แผนกที่เจอบ่อย ที่เหลือรวบเป็น "Other" (มีเป็นร้อยแผนก ถ้า one-hot หมดจะกระจายเกินไป)
TOP_SPECIALTIES = {"InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
                   "Surgery-General", "Nephrology", "Orthopedics", "Radiologist", "Pulmonology"}


def icd9_group(code) -> str:
    # รหัสโรค ICD-9 มีเป็นพันตัว เลยจัดกลุ่มตามระบบของร่างกาย (อ้างอิง Strack et al., 2014)
    if code is None or code in ("?", ""):
        return "missing"
    if isinstance(code, float) and np.isnan(code):
        return "missing"
    s = str(code)
    if s[0] in "VE":  # V = ปัจจัยอื่น, E = สาเหตุภายนอก
        return "other"
    try:
        v = float(s)
    except ValueError:
        return "other"

    if 390 <= v <= 459 or v == 785:
        return "circulatory"
    if 460 <= v <= 519 or v == 786:
        return "respiratory"
    if 520 <= v <= 579 or v == 787:
        return "digestive"
    if int(v) == 250:
        return "diabetes"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or v == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    # แปลงข้อมูลดิบ -> ฟีเจอร์ ฟังก์ชันนี้ไม่มี state ที่ต้อง fit
    d = df.copy()
    d = d.replace("?", np.nan)

    # ถ้า request ไม่มีบางคอลัมน์ ให้เติมเป็นค่าหาย (เดี๋ยว imputer จัดการต่อ)
    for c in INPUT_COLS:
        if c not in d.columns:
            d[c] = np.nan

    # "[70-80)" -> 75
    d["age_num"] = d["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(float) + 5

    for i in (1, 2, 3):
        d[f"diag_{i}_grp"] = d[f"diag_{i}"].map(icd9_group)

    d["medical_specialty"] = d["medical_specialty"].where(d["medical_specialty"].isin(TOP_SPECIALTIES), "Other")

    # ฟีเจอร์ที่สร้างเพิ่ม
    d["total_visits"] = d[["number_outpatient", "number_emergency", "number_inpatient"]].sum(axis=1)
    meds = d[MED_COLS]
    d["n_med_changes"] = meds.isin(["Up", "Down"]).sum(axis=1)  # ปรับยากี่ตัว
    d["n_meds_active"] = meds.isin(["Steady", "Up", "Down"]).sum(axis=1)  # ใช้ยากี่ตัว

    # id พวกนี้เป็นรหัสหมวดหมู่ ไม่ใช่ตัวเลขที่มีลำดับ เลยแปลงเป็น string
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        d[c] = d[c].astype("Int64").astype(str).replace("<NA>", np.nan)

    for c in CAT_COLS:
        d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)

    return d[NUM_COLS + CAT_COLS]


class QuantileClipper(BaseEstimator, TransformerMixin):
    # ตัดค่าสุดโต่ง (outlier) ให้อยู่ในช่วง quantile 0.5% - 99.5%
    # ขอบเขตเรียนจาก train อย่างเดียว ตอน serve ใช้ค่าเดิม

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def build_preprocessor() -> Pipeline:
    # ตัวเลข: เติมค่าหายด้วย median -> ตัด outlier -> scale
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper()),
        ("scale", StandardScaler()),
    ])
    # หมวดหมู่: ค่าหาย = "missing" -> one-hot
    # หมวดที่ไม่เคยเห็นตอนเทรน / หมวดที่เจอน้อยกว่า 20 ครั้ง จะไปรวมเป็น infrequent ไม่ error
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
    ])
    ct = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])
    return Pipeline([("clean", FunctionTransformer(clean_raw)), ("prep", ct)])
