# จัดการ Model Registry (MLflow)
# ขั้นตอน: register โมเดลใหม่ -> ผ่านด่าน gate ไหม -> ถ้าผ่านย้าย alias "champion" มาที่ตัวใหม่
# ตัว champion เดิมจะได้ alias "previous" ไว้ rollback
#
# สถานะ (เก็บใน tag "status"):
#   approved -> production -> archived / rolled_back      หรือ   rejected (ไม่ผ่าน gate)
#
# ใช้จาก command line:
#   python -m readmit.models.registry list
#   python -m readmit.models.registry rollback [--version N]
#   python -m readmit.models.registry promote N
from __future__ import annotations

import argparse
import json

import mlflow
import pandas as pd
from mlflow import MlflowClient

from readmit.alerts import send_alert
from readmit.config import load_params, path
from readmit.data.split import TARGET
from readmit.features.preprocess import INPUT_COLS
from readmit.models.evaluate import evaluate


def _cfg():
    return load_params()["registry"]


def current_version(alias: str):
    # คืน version ที่ alias นั้นชี้อยู่ ถ้ายังไม่มีคืน None
    try:
        return MlflowClient().get_model_version_by_alias(_cfg()["model_name"], alias)
    except Exception:
        return None


def run_gate(summary: dict) -> dict:
    # ด่านตรวจก่อนอนุมัติ ต้องผ่านทุกข้อ
    params = load_params()
    g = params["gate"]
    k = params["train"]["top_k_frac"]
    m = summary["test_metrics"]
    lace = summary["lace_test_metrics"]

    checks = {
        "pr_auc_min": m["test_pr_auc"] >= g["min_pr_auc"],
        "beats_lace": m["test_pr_auc"] >= lace["test_pr_auc"] + g["min_gain_over_lace"],
        "brier_max": m["test_brier"] <= g["max_brier"],
        "size_max": m["test_model_size_mb"] <= g["max_model_size_mb"],
        "latency_max": m["test_p95_latency_ms"] <= g["max_p95_latency_ms"],
        "fairness_gap_max": m["test_fairness_gap"] <= g["max_fairness_gap"],
    }

    # ถ้ามี champion อยู่แล้ว ต้องไม่แย่กว่าตัวเดิม
    # เอา champion มาทำนาย test ชุดเดียวกันใหม่ จะได้เทียบกันแฟร์ ๆ (เผื่อ test เปลี่ยนตอน retrain)
    champ = current_version(_cfg()["champion_alias"])
    champ_pr = None
    if champ is not None:
        te = pd.read_parquet(path(params["data"]["processed_dir"]) / "test.parquet")
        champ_model = mlflow.sklearn.load_model(f"models:/{_cfg()['model_name']}@{_cfg()['champion_alias']}")
        champ_pr = evaluate(champ_model, te[INPUT_COLS], te[TARGET], k)["pr_auc"]
        checks["not_worse_than_champion"] = m["test_pr_auc"] >= champ_pr - g["max_regression_vs_champion"]

    return {
        "checks": checks,
        "passed": all(checks.values()),
        "champion_pr_auc": champ_pr,
        "champion_version": champ.version if champ else None,
    }


def register_and_gate(summary: dict) -> dict:
    cfg = _cfg()
    client = MlflowClient()
    name = cfg["model_name"]

    mv = mlflow.register_model(summary["model_uri"], name)

    # เก็บ metric และ data version ไว้ใน tag ของ version นี้ (เปิดดูใน UI ได้เลย)
    tags = dict(summary["test_metrics"])
    tags["data_version"] = summary["data_version"]
    tags["threshold"] = summary["threshold"]
    tags["run_id"] = summary["run_id"]
    tags["trigger"] = summary["trigger"]
    for key, val in tags.items():
        client.set_model_version_tag(name, mv.version, key, str(val))

    gate = run_gate(summary)
    client.set_model_version_tag(name, mv.version, "gate_checks", json.dumps(gate["checks"]))

    if gate["passed"]:
        client.set_model_version_tag(name, mv.version, "status", "approved")
        promote(mv.version)  # ข้างในจะเปลี่ยน status เป็น production
        status = "approved"
    else:
        status = "rejected"
        client.set_model_version_tag(name, mv.version, "status", status)
        send_alert(f"MODEL v{mv.version} REJECTED BY GATE", gate)

    result = {"version": mv.version, "status": status, **gate,
              "test_metrics": summary["test_metrics"],  # ใส่ตัวเลขจริงไว้ด้วย ดูง่ายว่าตกข้อไหนเพราะอะไร
              "fairness_detail": summary.get("fairness_detail")}
    path("reports/gate.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def promote(version) -> None:
    cfg = _cfg()
    client = MlflowClient()
    name = cfg["model_name"]
    old = current_version(cfg["champion_alias"])

    # champion เดิม -> previous (เก็บไว้ rollback)
    if old is not None and str(old.version) != str(version):
        client.set_registered_model_alias(name, cfg["previous_alias"], old.version)
        client.set_model_version_tag(name, old.version, "status", "archived")

    client.set_registered_model_alias(name, cfg["champion_alias"], version)
    client.set_model_version_tag(name, version, "status", "production")

    msg = f"champion -> v{version}"
    if old:
        msg += f" (previous: v{old.version})"
    print(msg)


def rollback(to_version=None) -> None:
    # ถอยกลับไปเวอร์ชันก่อนหน้า แค่ย้าย alias ไม่ต้อง deploy ใหม่
    # API จะ poll registry ทุก 30 วิ แล้วโหลดตัวใหม่เอง
    cfg = _cfg()
    client = MlflowClient()
    name = cfg["model_name"]

    target = to_version
    if target is None:
        prev = current_version(cfg["previous_alias"])
        if prev is None:
            raise SystemExit("no previous version to roll back to")
        target = prev.version

    cur = current_version(cfg["champion_alias"])
    client.set_registered_model_alias(name, cfg["champion_alias"], target)
    client.set_model_version_tag(name, target, "status", "production")

    # ตอนนี้ previous กับ champion ชี้ตัวเดียวกันแล้ว ลบ previous ทิ้ง
    prev = current_version(cfg["previous_alias"])
    if prev is not None and str(prev.version) == str(target):
        client.delete_registered_model_alias(name, cfg["previous_alias"])

    if cur:
        client.set_model_version_tag(name, cur.version, "status", "rolled_back")
    send_alert(f"ROLLBACK champion v{cur.version if cur else '?'} -> v{target}")
    print(f"rolled back: champion -> v{target}")


def list_versions() -> None:
    cfg = _cfg()
    client = MlflowClient()
    rm = client.get_registered_model(cfg["model_name"])

    # กลับด้าน alias -> version ให้เป็น version -> [alias]
    aliases = {}
    for a, v in (rm.aliases or {}).items():
        aliases.setdefault(str(v), []).append(a)

    versions = client.search_model_versions(f"name='{cfg['model_name']}'")
    for mv in sorted(versions, key=lambda x: int(x.version)):
        t = mv.tags
        print(f"v{mv.version:<3} status={t.get('status', '-'):<12} pr_auc={t.get('test_pr_auc', '-'):<8} "
              f"data={t.get('data_version', '-')} aliases={aliases.get(str(mv.version), [])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    rb = sub.add_parser("rollback")
    rb.add_argument("--version")
    pr = sub.add_parser("promote")
    pr.add_argument("version")
    args = ap.parse_args()

    if args.cmd == "list":
        list_versions()
    elif args.cmd == "rollback":
        rollback(args.version)
    elif args.cmd == "promote":
        promote(args.version)
