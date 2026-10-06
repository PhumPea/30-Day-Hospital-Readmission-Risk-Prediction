# เทสต์ API โดยใช้โมเดลเล็ก ๆ แทน (ไม่ต้องต่อ MLflow)
import json

import pytest
from fastapi.testclient import TestClient

import api.main as main


@pytest.fixture()
def client(fitted_pipe, monkeypatch, tmp_path):
    # ปิดการเชื่อม MLflow แล้วยัดโมเดลที่เทรนใน fixture เข้าไปแทน
    monkeypatch.setattr(main.store, "refresh", lambda: True)
    monkeypatch.setattr(main.store, "start_polling", lambda *a, **k: None)
    main.store.model, main.store.version, main.store.threshold = fitted_pipe[0], "test", 0.5
    monkeypatch.setattr(main, "PRED_LOG", tmp_path / "p.jsonl")
    monkeypatch.setattr(main, "LABEL_LOG", tmp_path / "l.jsonl")
    with TestClient(main.app) as c:
        yield c


GOOD = json.load(open("data/bad_samples/good_request.json"))


def test_health_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["model_version"] == "test"


def test_predict(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1 and body["model_version"] == "test"


def test_batch(client):
    r = client.post("/predict/batch", json={"encounters": [GOOD, dict(GOOD, encounter_id=2)]})
    assert r.status_code == 200 and len(r.json()) == 2


@pytest.mark.parametrize("bad", json.load(open("data/bad_samples/bad_requests.json")))
def test_bad_input_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_feedback_and_metrics(client):
    assert client.post("/feedback", json={"encounter_id": 1, "readmitted_30d": True}).status_code == 200
    m = client.get("/metrics").text
    assert "api_requests_total" in m and "feedback_labels_total" in m


def test_not_ready_returns_503(client, monkeypatch):
    monkeypatch.setattr(main.store, "model", None)
    assert client.post("/predict", json=GOOD).status_code == 503
