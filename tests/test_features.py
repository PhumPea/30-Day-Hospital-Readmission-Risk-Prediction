# เทสต์ feature engineering และ training-serving skew
import json

import numpy as np
import pandas as pd

from api.schemas import Encounter
from readmit.features.preprocess import CAT_COLS, INPUT_COLS, NUM_COLS, clean_raw, icd9_group


def test_icd9_groups():
    assert icd9_group("428") == "circulatory"
    assert icd9_group("250.83") == "diabetes"
    assert icd9_group("V57") == "other"
    assert icd9_group("?") == "missing"
    assert icd9_group(np.nan) == "missing"


def test_clean_raw_output_columns_and_no_question_marks(synth):
    out = clean_raw(synth[INPUT_COLS])
    assert list(out.columns) == NUM_COLS + CAT_COLS
    assert not (out == "?").any().any()
    assert out["age_num"].between(5, 95).all()


def test_clean_raw_tolerates_missing_fields():
    out = clean_raw(pd.DataFrame([{"age": "[70-80)", "time_in_hospital": 3}]))
    assert len(out) == 1


def test_no_training_serving_skew(fitted_pipe):
    # ข้อมูลชุดเดียวกัน ส่งผ่านทางตอนเทรน (DataFrame) กับทางตอน serve (JSON -> Pydantic -> DataFrame)
    # ต้องได้ผลทำนายเท่ากันเป๊ะ ถ้าไม่เท่าแปลว่ามี training-serving skew
    pipe, df = fitted_pipe
    sample = df.head(50)
    p_train = pipe.predict_proba(sample[INPUT_COLS])[:, 1]
    rows = json.loads(sample[INPUT_COLS + ["encounter_id"]].to_json(orient="records"))
    served = pd.DataFrame([Encounter(**r).to_row() for r in rows])[INPUT_COLS]
    p_serve = pipe.predict_proba(served)[:, 1]
    np.testing.assert_allclose(p_train, p_serve, rtol=1e-9)
