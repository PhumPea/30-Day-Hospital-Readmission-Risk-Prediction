# Baseline: LACE index (van Walraven, 2010)
# เป็นสูตรคะแนนที่ รพ. ใช้กันอยู่แล้ว ไม่ได้เรียนรู้อะไร เอาไว้พิสูจน์ว่า ML ดีกว่ากฎจริงไหม
#   L = จำนวนวันนอน รพ.
#   A = เข้าแบบฉุกเฉิน (acute)
#   C = Charlson comorbidity (โรคร่วม)
#   E = จำนวนครั้งที่มา ER ก่อนหน้า
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

# (ICD-9 ต่ำสุด, สูงสุด, น้ำหนัก Charlson) — ทำแบบย่อ
CHARLSON = [
    (410, 412.99, 1),   # กล้ามเนื้อหัวใจตาย
    (428, 428.99, 1),   # หัวใจล้มเหลว
    (440, 443.99, 1),   # หลอดเลือดส่วนปลาย
    (430, 438.99, 1),   # หลอดเลือดสมอง
    (290, 290.99, 1),   # สมองเสื่อม
    (490, 496.99, 1),   # ปอดเรื้อรัง
    (710, 714.99, 1),   # โรคข้อ/รูมาตอยด์
    (531, 534.99, 1),   # แผลในกระเพาะ
    (571, 571.99, 1),   # ตับ
    (250, 250.39, 1),   # เบาหวานไม่มีภาวะแทรกซ้อน
    (250.4, 250.99, 2), # เบาหวานมีภาวะแทรกซ้อน
    (342, 344.99, 2),   # อัมพาต
    (582, 586.99, 2),   # ไต
    (140, 195.99, 2),   # มะเร็ง
    (200, 208.99, 2),   # มะเร็งเม็ดเลือด
    (196, 199.99, 6),   # มะเร็งแพร่กระจาย
    (42, 44.99, 6),     # HIV
]


def _charlson(codes) -> int:
    score = 0
    for c in codes:
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue  # รหัส V/E หรือค่าหาย ข้ามไป
        best = 0
        for lo, hi, w in CHARLSON:
            if lo <= v <= hi:
                best = max(best, w)
        score += best
    return score


def lace_score(df: pd.DataFrame) -> np.ndarray:
    los = df["time_in_hospital"].astype(float)
    L = np.select([los < 1, los == 1, los == 2, los == 3, los <= 6, los <= 13], [0, 1, 2, 3, 4, 5], 7)
    A = np.where(pd.to_numeric(df["admission_type_id"], errors="coerce") == 1, 3, 0)  # 1 = Emergency
    ch = df[["diag_1", "diag_2", "diag_3"]].apply(_charlson, axis=1).to_numpy()
    C = np.where(ch >= 4, 5, ch)
    E = np.minimum(df["number_emergency"].astype(float), 4)
    return (L + A + C + E).astype(float)  # เต็ม 19


class LaceModel(BaseEstimator, ClassifierMixin):
    # ห่อ LACE ให้มี interface เหมือนโมเดล sklearn จะได้ประเมินด้วยโค้ดเดียวกันได้

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(lace_score(X) / 19.0, 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        # LACE >= 10 ถือว่าเสี่ยงสูง (ตามงานวิจัยต้นฉบับ)
        return (self.predict_proba(X)[:, 1] >= 10 / 19).astype(int)
