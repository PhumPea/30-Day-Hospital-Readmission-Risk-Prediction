# งานเฝ้าระวัง (รันเป็นรอบ ๆ จาก scheduler)
#
# Data drift       = P(X) เปลี่ยน   -> PSI ของแต่ละฟีเจอร์ (production เทียบกับ train)
# Prediction drift = คะแนนความเสี่ยงที่ทำนายออกมามีการกระจายเปลี่ยน
# Concept drift    = P(y|X) เปลี่ยน -> ต้องรอ label จริง (ราว 30 วันหลังกลับบ้าน)
#                    แล้วดูว่า ROC-AUC ตกจากตอนทดสอบไหม
# Label shift      = อัตราการกลับมารักษาซ้ำเปลี่ยน
#
# ทำไมใช้ ROC-AUC ไม่ใช้ PR-AUC: PR-AUC ขึ้นกับสัดส่วน label บวก ถ้าอัตรากลับมาซ้ำเปลี่ยน
# (label shift) ค่า PR-AUC ก็เปลี่ยนตาม ทั้งที่โมเดลยังจัดลำดับได้ดีเหมือนเดิม จะทำให้สับสนกับ concept drift
#
# ระหว่างรอ label, data drift กับ prediction drift ทำหน้าที่เตือนล่วงหน้า
from __future__ import annotations

import json
from datetime import datetime, timezone

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from readmit.alerts import send_alert
from readmit.config import load_params, path
from readmit.data.split import TARGET
from readmit.features.preprocess import CAT_COLS, INPUT_COLS, NUM_COLS, clean_raw
from readmit.monitoring.drift import feature_drift, psi_numeric


def read_jsonl(f) -> pd.DataFrame:
    f = path(f)
    if not f.exists() or f.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(f, lines=True)


def load_production(window: int):
    # อ่าน log การทำนายจาก API (เอาแค่ window ล่าสุด) แล้วจับคู่กับ label ที่ส่งมาทีหลัง
    preds = read_jsonl("logs/predictions.jsonl")
    if preds.empty:
        return pd.DataFrame(), pd.DataFrame()
    preds = preds.tail(window).reset_index(drop=True)

    feats = pd.json_normalize(preds["features"].tolist())
    feats["score"] = preds["score"].to_numpy()
    feats["model_version"] = preds["model_version"].astype(str).to_numpy()

    labels = read_jsonl("logs/labels.jsonl")
    if labels.empty:
        return feats, pd.DataFrame()
    labels = labels.drop_duplicates("encounter_id", keep="last")
    labeled = feats.merge(labels[["encounter_id", "readmitted_30d"]], on="encounter_id")
    return feats, labeled


def run() -> dict:
    params = load_params()
    mcfg = params["monitoring"]
    reg = params["registry"]
    proc = path(params["data"]["processed_dir"])

    train_raw = pd.read_parquet(proc / "train.parquet")  # อ้างอิงการกระจายของฟีเจอร์
    ref = pd.read_parquet(proc / "test.parquet")          # อ้างอิงประสิทธิภาพที่คาดไว้
    cur_raw, labeled = load_production(mcfg["window_size"])

    report = {"time": datetime.now(timezone.utc).isoformat(), "n_production": len(cur_raw)}
    if len(cur_raw) < 100:
        report["status"] = "insufficient_data"  # ข้อมูลน้อยไป สถิติไม่น่าเชื่อถือ
        _write(report)
        return report

    # ---------- 1) data drift ----------
    # ใช้ clean_raw ตัวเดียวกับโมเดล เทียบกันบนฟีเจอร์ที่โมเดลเห็นจริง
    ref_f = clean_raw(train_raw[INPUT_COLS])
    cur_f = clean_raw(cur_raw[INPUT_COLS])
    psi = feature_drift(ref_f, cur_f, NUM_COLS, CAT_COLS)
    drifted = {k: v for k, v in psi.items() if v >= mcfg["psi_threshold"]}
    drift_share = len(drifted) / len(psi)
    data_drift = bool(drift_share >= mcfg["drift_share_threshold"])

    # ---------- 2) prediction drift ----------
    champ = MlflowClient().get_model_version_by_alias(reg["model_name"], reg["champion_alias"])
    model = mlflow.sklearn.load_model(f"models:/{reg['model_name']}/{champ.version}")
    ref_scores = model.predict_proba(ref[INPUT_COLS])[:, 1]
    ref_pr = average_precision_score(ref[TARGET], ref_scores)
    ref_auc = roc_auc_score(ref[TARGET], ref_scores)
    pred_psi = psi_numeric(pd.Series(ref_scores), cur_raw["score"])
    pred_drift = bool(pred_psi >= mcfg["pred_psi_threshold"])

    # ---------- 3) concept drift + label shift (ต้องมี label พอ) ----------
    live_pr = None
    live_auc = None
    live_pos = None
    concept_drift = False
    label_shift = False
    if len(labeled) >= mcfg["min_labels"] and labeled["readmitted_30d"].nunique() == 2:
        y = labeled["readmitted_30d"].astype(int)
        live_pr = average_precision_score(y, labeled["score"])
        live_auc = roc_auc_score(y, labeled["score"])
        live_pos = float(y.mean())
        concept_drift = bool((ref_auc - live_auc) >= mcfg["perf_drop_threshold"])
        label_shift = bool(abs(live_pos - ref[TARGET].mean()) >= mcfg["label_shift_threshold"])

    reason = None
    if concept_drift:
        reason = "concept_drift"
    elif data_drift:
        reason = "data_drift"

    report.update({
        "status": "ok",
        "champion_version": champ.version,
        "data_drift": data_drift,
        "drift_share": round(drift_share, 3),
        "drifted_features": drifted,
        "prediction_drift": pred_drift,
        "prediction_psi": round(pred_psi, 4),
        "concept_drift": concept_drift,
        "label_shift": label_shift,
        "n_labeled": len(labeled),
        "ref_roc_auc": round(ref_auc, 4),
        "live_roc_auc": None if live_auc is None else round(live_auc, 4),
        "ref_pr_auc": round(ref_pr, 4),
        "live_pr_auc": None if live_pr is None else round(live_pr, 4),
        "ref_pos_rate": round(float(ref[TARGET].mean()), 4),
        "live_pos_rate": live_pos,
        "feature_psi": psi,
        "retrain_recommended": concept_drift or data_drift,
        "reason": reason,
    })

    # ค่าที่ API จะ expose ให้ Prometheus (ไปขึ้นกราฟ Grafana + alert rules)
    report["gauges"] = {
        "data_drift": int(data_drift),
        "drift_share": drift_share,
        "prediction_psi": pred_psi,
        "prediction_drift": int(pred_drift),
        "concept_drift": int(concept_drift),
        "label_shift": int(label_shift),
        "live_pr_auc": live_pr if live_pr is not None else -1,
        "ref_pr_auc": ref_pr,
        "live_roc_auc": live_auc if live_auc is not None else -1,
        "ref_roc_auc": ref_auc,
        "n_labeled": len(labeled),
        "max_feature_psi": max(psi.values()),
    }

    if data_drift:
        send_alert("DATA DRIFT DETECTED", {"drift_share": drift_share, "features": drifted})
    if pred_drift:
        send_alert("PREDICTION DRIFT DETECTED", {"psi": pred_psi})
    if concept_drift:
        send_alert("CONCEPT DRIFT DETECTED", {"ref_roc_auc": ref_auc, "live_roc_auc": live_auc})
    if label_shift:
        send_alert("LABEL SHIFT (readmission rate changed)", {"ref": float(ref[TARGET].mean()), "live": live_pos})

    _write(report)
    return report


def _write(report: dict) -> None:
    f = path("reports/monitoring.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(report, indent=2, default=str))
    # เก็บประวัติไว้ดูย้อนหลัง (ไม่เก็บ psi รายฟีเจอร์ ไฟล์จะใหญ่)
    history = {k: v for k, v in report.items() if k != "feature_psi"}
    with open(path("reports/monitoring_history.jsonl"), "a") as fh:
        fh.write(json.dumps(history, default=str) + "\n")


if __name__ == "__main__":
    r = run()
    r.pop("feature_psi", None)
    print(json.dumps(r, indent=2, default=str))
