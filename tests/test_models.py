# เทสต์ฟังก์ชันวัดผล, LACE baseline และ PSI
import numpy as np

from readmit.data.split import TARGET
from readmit.features.preprocess import INPUT_COLS
from readmit.models.evaluate import evaluate, fairness_gap, recall_precision_at_k
from readmit.models.lace import LaceModel, lace_score
from readmit.monitoring.drift import psi_categorical, psi_numeric


def test_lace_range(synth):
    s = lace_score(synth)
    assert s.min() >= 0 and s.max() <= 19


def test_recall_at_k():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .1])
    r, prec = recall_precision_at_k(y, p, 0.2)
    assert r == 0.5 and prec == 0.5


def test_model_beats_lace(fitted_pipe):
    pipe, df = fitted_pipe
    m = evaluate(pipe, df[INPUT_COLS], df[TARGET], 0.2)
    b = evaluate(LaceModel().fit(df), df[INPUT_COLS], df[TARGET], 0.2)
    assert m["pr_auc"] > b["pr_auc"]


def test_psi():
    rng = np.random.default_rng(0)
    import pandas as pd
    a = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi_numeric(a, pd.Series(rng.normal(1.5, 1, 5000))) > 0.2
    c = pd.Series(["a"] * 500 + ["b"] * 500)
    assert psi_categorical(c, c) < 1e-6
    assert psi_categorical(c, pd.Series(["a"] * 900 + ["b"] * 100)) > 0.2


def test_fairness_ignores_missing_and_small_groups():
    import pandas as pd
    # กลุ่ม "?" กับกลุ่มเล็ก (positive < 100) ห้ามทำให้ gap สูง
    n = 4000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "gender": rng.choice(["Female", "Male"], n),
        "race": ["Caucasian"] * 3000 + ["?"] * 900 + ["Asian"] * 100,
        "age": "[70-80)",
    })
    y = rng.random(n) < 0.3
    p = rng.random(n)
    p[3000:] = 0  # กลุ่ม missing / Asian ไม่ถูก flag เลย
    fg = fairness_gap(df, y, p, 0.2, detail=True)
    assert fg["race"] == 0.0
    assert fg["groups"]["race"]["missing"]["used_in_gate"] is False
    assert fg["groups"]["race"]["Asian"]["used_in_gate"] is False
