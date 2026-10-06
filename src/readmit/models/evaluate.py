# ฟังก์ชันวัดผลโมเดล
# - ML metrics: PR-AUC (ตัวหลัก), ROC-AUC, Brier
# - ธุรกิจ: recall / precision ที่ top 20% (พยาบาลโทรตามได้แค่ประมาณ 20% ของคนไข้ที่กลับบ้าน)
# - ความเป็นธรรม: recall ต่างกันระหว่างกลุ่มเพศ/เชื้อชาติแค่ไหน
# - ด้าน ops: latency และขนาดโมเดล
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def recall_precision_at_k(y, p, k_frac: float):
    # เรียงคนไข้จากเสี่ยงมากไปน้อย เอา k% แรก แล้วดูว่าจับคนที่กลับมาจริงได้กี่คน
    y = np.asarray(y)
    k = max(1, int(len(p) * k_frac))
    top_idx = np.argsort(-np.asarray(p), kind="stable")[:k]
    tp = y[top_idx].sum()
    recall = tp / max(y.sum(), 1)
    precision = tp / k
    return float(recall), float(precision)


# กลุ่มที่ไม่ใช่กลุ่มประชากรจริง ไม่เอามาเทียบความเป็นธรรม ("missing" = ไม่ได้กรอกเชื้อชาติ)
NOT_A_GROUP = {"missing", "Unknown/Invalid"}
# ต้องมีคนที่กลับมาจริงในกลุ่มอย่างน้อยเท่านี้ ไม่งั้น recall แกว่งเพราะสุ่ม ไม่ได้แปลว่าลำเอียง
# (เช่น 30 คน แค่จับได้ต่างกัน 3 คน recall ก็ต่างไป 10% แล้ว)
MIN_POSITIVES = 100


def fairness_gap(df: pd.DataFrame, y, p, k_frac: float, detail: bool = False) -> dict:
    # ดูว่า recall@k ของแต่ละกลุ่มต่างกันมากไหม (ค่ามาก = โมเดลลำเอียง)
    # gate ใช้แค่ gender กับ race
    # อายุไม่เอามาตัดสิน เพราะอายุเป็นปัจจัยเสี่ยงทางการแพทย์อยู่แล้ว ต่างกันได้ แค่รายงานไว้
    y = np.asarray(y)
    p = np.asarray(p)
    threshold = np.quantile(p, 1 - k_frac)
    flagged = p >= threshold

    age_start = df["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(float)
    groups = {
        "gender": df["gender"].astype(str).to_numpy(),
        "race": df["race"].astype(str).replace({"?": "missing", "nan": "missing"}).to_numpy(),
        "age60": np.where(age_start >= 60, ">=60", "<60"),
    }

    out = {}
    groups_detail = {}
    for name, values in groups.items():
        recalls = {}
        groups_detail[name] = {}
        for v in np.unique(values):
            mask = (values == v) & (y == 1)
            n_pos = int(mask.sum())
            r = float(flagged[mask].mean()) if n_pos > 0 else None
            used = v not in NOT_A_GROUP and n_pos >= MIN_POSITIVES
            groups_detail[name][str(v)] = {"n_positive": n_pos, "recall_at_k": r, "used_in_gate": used}
            if used:
                recalls[v] = r
        if len(recalls) > 1:
            out[name] = float(max(recalls.values()) - min(recalls.values()))
        else:
            out[name] = 0.0
    out["max"] = max(out["gender"], out["race"])
    if detail:
        out["groups"] = groups_detail  # เอาไว้ใส่รายงาน ว่าแต่ละกลุ่ม recall เท่าไหร่
    return out


def latency_p95_ms(model, X: pd.DataFrame, n: int = 200) -> float:
    # จับเวลาทำนายทีละ 1 แถว (เหมือนตอนเรียก API จริง)
    times = []
    for i in range(n):
        row = X.iloc[[i % len(X)]]
        start = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 95))


def model_size_mb(model) -> float:
    return len(pickle.dumps(model)) / 1e6


def evaluate(model, X: pd.DataFrame, y, k_frac: float, prefix: str = "", with_ops: bool = False) -> dict:
    p = model.predict_proba(X)[:, 1]
    recall_k, precision_k = recall_precision_at_k(y, p, k_frac)
    fg = fairness_gap(X, y, p, k_frac)

    m = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "recall_at_k": recall_k,
        "precision_at_k": precision_k,
        "fairness_gap": fg["max"],
        "age_recall_gap": fg["age60"],
    }
    if with_ops:
        m["p95_latency_ms"] = latency_p95_ms(model, X)
        m["model_size_mb"] = model_size_mb(model)

    return {prefix + k: round(v, 5) for k, v in m.items()}
