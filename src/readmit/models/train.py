# เทรน baseline + การทดลองทั้งหมดใน params.yaml แล้วบันทึกลง MLflow
# บันทึกครบ 6 อย่างตามโจทย์:
#   1. code version  2. data version  3. hyperparameters  4. metrics  5. artifacts  6. environment
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from readmit.config import git_commit, load_params, path, set_seed
from readmit.data.split import TARGET
from readmit.features.preprocess import INPUT_COLS, build_preprocessor
from readmit.models.evaluate import evaluate, fairness_gap
from readmit.models.lace import LaceModel

# MLflow 3 เซฟโมเดลด้วย skops ซึ่งจะไม่ยอมโหลด class ที่ไม่รู้จัก (กันไฟล์อันตราย)
# เลยต้องระบุว่าเราไว้ใจ class ไหนบ้าง (เป็นโค้ดของเราเอง + class ของ sklearn)
TRUSTED_TYPES = [
    "numpy.dtype",
    "readmit.features.preprocess.QuantileClipper",
    "readmit.features.preprocess.clean_raw",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.calibration._CalibratedClassifier",
]


def load_splits(extra_train: pd.DataFrame | None = None):
    d = path(load_params()["data"]["processed_dir"])
    tr = pd.read_parquet(d / "train.parquet")
    va = pd.read_parquet(d / "val.parquet")
    te = pd.read_parquet(d / "test.parquet")
    if extra_train is not None and len(extra_train) > 0:
        tr = pd.concat([tr, extra_train], ignore_index=True)
    meta = json.loads((d / "meta.json").read_text())
    return tr, va, te, meta


def make_estimator(kind: str, params: dict, seed: int):
    params = dict(params)
    calibrate = params.pop("calibrate", False)
    if kind == "logreg":
        est = LogisticRegression(random_state=seed, **params)
    elif kind == "hgb":
        est = HistGradientBoostingClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"unknown model: {kind}")
    if calibrate:
        # ปรับให้ความน่าจะเป็นตรงกับความจริงมากขึ้น (Brier ดีขึ้น)
        est = CalibratedClassifierCV(est, method="isotonic", cv=3)
    return est


def log_environment() -> None:
    # (6) environment: pip freeze + python + os + docker image
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "requirements.lock.txt"
        f.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode())
        mlflow.log_artifact(str(f), "environment")
    mlflow.set_tags({
        "env.python": platform.python_version(),
        "env.platform": platform.platform(),
        "env.docker_image": os.getenv("IMAGE_TAG", "local"),
    })


def log_common(meta: dict, run_kind: str) -> None:
    commit = git_commit()
    mlflow.set_tags({
        "code_version": commit,                  # (1)
        "mlflow.source.git.commit": commit,
        "data_version": meta["data_version"],    # (2)
        "data.raw_sha256": meta["raw_sha256"],
        "run_kind": run_kind,
    })
    mlflow.log_dict(meta, "data/meta.json")
    log_environment()


def run(extra_train: pd.DataFrame | None = None, trigger: str = "manual") -> dict:
    params = load_params()
    set_seed(params["seed"])
    cfg = params["train"]
    k = cfg["top_k_frac"]

    mlflow.set_experiment(cfg["experiment_name"])
    tr, va, te, meta = load_splits(extra_train)
    X_train, y_train = tr[INPUT_COLS], tr[TARGET]
    X_val, y_val = va[INPUT_COLS], va[TARGET]
    X_test, y_test = te[INPUT_COLS], te[TARGET]

    results = []
    # 1 run แม่ ต่อ 1 รอบการเทรน แล้วแต่ละการทดลองเป็น run ลูก (เปิดดูเทียบกันใน UI ได้ง่าย)
    with mlflow.start_run(run_name=f"train-session-{trigger}") as parent:
        mlflow.set_tag("trigger", trigger)

        # ---------- baseline: LACE ----------
        with mlflow.start_run(run_name="baseline_lace", nested=True):
            log_common(meta, "baseline")
            lace = LaceModel().fit(X_train)
            mlflow.log_param("model", "LACE index (rule)")
            lace_metrics = evaluate(lace, X_val, y_val, k, "val_")
            lace_metrics.update(evaluate(lace, X_test, y_test, k, "test_"))
            mlflow.log_metrics(lace_metrics)

        # ---------- การทดลอง ML ----------
        for exp in cfg["experiments"]:
            with mlflow.start_run(run_name=exp["name"], nested=True) as r:
                log_common(meta, "experiment")
                mlflow.log_params({  # (3)
                    "model": exp["model"], **exp["params"],
                    "seed": params["seed"], "train_rows": len(tr),
                })

                pipe = Pipeline([
                    ("features", build_preprocessor()),
                    ("model", make_estimator(exp["model"], exp["params"], params["seed"])),
                ])
                pipe.fit(X_train, y_train)

                val_metrics = evaluate(pipe, X_val, y_val, k, "val_")
                mlflow.log_metrics(val_metrics)  # (4)

                sig = infer_signature(X_val.head(50), pipe.predict_proba(X_val.head(50)))
                info = mlflow.sklearn.log_model(  # (5) ตัวโมเดล (รวม preprocessing)
                    pipe, name="model", signature=sig, input_example=X_val.head(3),
                    serialization_format="skops", skops_trusted_types=TRUSTED_TYPES,
                )
                mlflow.log_dict(val_metrics, "metrics/val_metrics.json")
                results.append({"name": exp["name"], "run_id": r.info.run_id,
                                "model_uri": info.model_uri, "pipe": pipe, "val": val_metrics})

        # เลือกตัวที่ val PR-AUC สูงสุด แล้วค่อยวัดบน test ครั้งเดียว (ไม่เอา test มาเลือกโมเดล)
        best = results[0]
        for res in results:
            if res["val"]["val_pr_auc"] > best["val"]["val_pr_auc"]:
                best = res
        test_metrics = evaluate(best["pipe"], X_test, y_test, k, "test_", with_ops=True)
        # รายละเอียด recall แยกตามกลุ่ม (เพศ / เชื้อชาติ / อายุ) เก็บไว้ทำรายงานเรื่อง fairness
        test_scores = best["pipe"].predict_proba(X_test)[:, 1]
        fairness = fairness_gap(X_test, y_test, test_scores, k, detail=True)

        # threshold ตัดสินว่าต้องโทรตามไหม = คะแนนที่ทำให้ได้ top 20% บน validation
        val_scores = best["pipe"].predict_proba(X_val)[:, 1]
        threshold = float(pd.Series(val_scores).quantile(1 - k))

        with mlflow.start_run(run_id=best["run_id"], nested=True):
            mlflow.log_metrics(test_metrics)
            mlflow.set_tag("selected", "true")
            mlflow.log_metric("decision_threshold", threshold)
            mlflow.log_dict(fairness, "metrics/fairness_test.json")

        # ตารางเทียบทุกการทดลอง
        rows = [{"name": "baseline_lace", **lace_metrics}]
        for res in results:
            rows.append({"name": res["name"], **res["val"]})
        comparison = pd.DataFrame(rows)
        mlflow.log_table(comparison, "comparison.json")
        out = path("reports/experiments.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out, index=False)
        mlflow.log_param("selected_model", best["name"])

        summary = {
            "parent_run_id": parent.info.run_id,
            "best": best["name"],
            "run_id": best["run_id"],
            "model_uri": best["model_uri"],
            "test_metrics": test_metrics,
            "threshold": round(threshold, 6),
            "fairness_detail": fairness,
            "lace_test_metrics": {key: v for key, v in lace_metrics.items() if key.startswith("test_")},
            "data_version": meta["data_version"],
            "trigger": trigger,
        }

    path("reports/train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
