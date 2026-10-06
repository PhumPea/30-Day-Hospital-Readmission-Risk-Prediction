# คำนวณ PSI (Population Stability Index) เขียนเอง
# PSI ใช้วัดว่าการกระจายของข้อมูลชุดใหม่ต่างจากชุดอ้างอิงแค่ไหน
#   < 0.1  แทบไม่เปลี่ยน
#   0.1-0.2 เปลี่ยนนิดหน่อย
#   >= 0.2 เปลี่ยนชัดเจน (ใช้เป็นเกณฑ์ drift)
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4  # กันหารศูนย์ / log(0) ตอนบางช่องไม่มีข้อมูล


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = ref.dropna().astype(float)
    cur = cur.dropna().astype(float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # แบ่งช่องตาม quantile ของข้อมูลอ้างอิง (แต่ละช่องมีข้อมูลเท่า ๆ กัน)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        # ข้อมูลอ้างอิงมีค่าเดียว
        same = cur.nunique() <= 1 and (cur == ref.iloc[0]).all()
        return 0.0 if same else 1.0
    edges[0], edges[-1] = -np.inf, np.inf

    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, EPS, None)
    c = np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref = ref.fillna("missing").astype(str)
    cur = cur.fillna("missing").astype(str)
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r = np.clip(r, EPS, None)
    c = np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, num_cols, cat_cols) -> dict:
    result = {}
    for c in num_cols:
        result[c] = round(psi_numeric(ref[c], cur[c]), 4)
    for c in cat_cols:
        result[c] = round(psi_categorical(ref[c], cur[c]), 4)
    return result
