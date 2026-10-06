# เทสต์ฝั่งข้อมูล: schema จับข้อมูลเสียได้ไหม และการแบ่งข้อมูลถูกต้อง/ทำซ้ำได้ไหม
import pandas as pd
import pytest

from readmit.data.schema import DataValidationError, read_raw, validate
from readmit.data.split import TARGET, make_target, split


def test_valid_data_passes(synth):
    assert validate(synth)["passed"]


def test_bad_data_stops_pipeline():
    bad = read_raw("data/bad_samples/bad_raw.csv")
    with pytest.raises(DataValidationError):
        validate(bad)


def test_bad_data_reports_each_problem():
    rep = validate(read_raw("data/bad_samples/bad_raw.csv"), raise_on_error=False)
    cols = {i.get("column") for i in rep["issues"]}
    assert {"time_in_hospital", "gender", "encounter_id", "age", "num_medications"} <= cols


def test_missing_column_detected(synth):
    rep = validate(synth.drop(columns=["time_in_hospital"]), raise_on_error=False)
    assert not rep["passed"]


def test_target_excludes_expired(synth):
    df = make_target(synth, [11])
    assert 11 not in df["discharge_disposition_id"].unique()
    assert set(df[TARGET].unique()) <= {0, 1}


def test_split_patient_level_and_deterministic(synth):
    df = make_target(synth, [11])
    a = split(df, 0.15, 0.15, 42)
    b = split(df.sample(frac=1, random_state=1), 0.15, 0.15, 42)  # สลับลำดับแถวแล้วต้องได้เหมือนเดิม
    # ผู้ป่วยคนเดียวกันต้องไม่อยู่หลายชุด
    ids = [set(p["patient_nbr"]) for p in a]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    for x, y in zip(a, b, strict=True):
        assert set(x["encounter_id"]) == set(y["encounter_id"])
    assert sum(len(p) for p in a) == len(df)
    assert isinstance(a[0], pd.DataFrame)
