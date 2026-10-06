# ตรวจคุณภาพข้อมูลดิบก่อนเอาไปเทรน
# มี 2 ชั้น
#   1) schema ของ pandera: ชนิดข้อมูล ช่วงค่า ค่าที่อนุญาต id ห้ามซ้ำ
#   2) เช็คทั้งชุดข้อมูล: จำนวนแถว อัตราค่าหาย อัตรา label
# ถ้าไม่ผ่าน -> ส่ง alert แล้ว raise error ให้ pipeline หยุด
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from readmit.alerts import send_alert
from readmit.config import load_params, path
from readmit.data.ingest import MED_COLS

AGE_BINS = [f"[{a}-{a + 10})" for a in range(0, 100, 10)]
MED_VALUES = ["No", "Steady", "Up", "Down"]

# ช่วงค่าต่าง ๆ ตั้งจากค่าที่เป็นไปได้จริงในชุดข้อมูล (เผื่อไว้นิดหน่อย)
columns = {
    "encounter_id": pa.Column(int, pa.Check.gt(0), unique=True),
    "patient_nbr": pa.Column(int, pa.Check.gt(0)),
    "race": pa.Column(str, nullable=True),
    "gender": pa.Column(str, pa.Check.isin(["Female", "Male", "Unknown/Invalid"])),
    "age": pa.Column(str, pa.Check.isin(AGE_BINS)),
    "admission_type_id": pa.Column(int, pa.Check.in_range(1, 8)),
    "discharge_disposition_id": pa.Column(int, pa.Check.in_range(1, 30)),
    "admission_source_id": pa.Column(int, pa.Check.in_range(1, 26)),
    "time_in_hospital": pa.Column(int, pa.Check.in_range(1, 14)),  # ข้อมูลนี้นอนได้ 1-14 วัน
    "medical_specialty": pa.Column(str, nullable=True),
    "num_lab_procedures": pa.Column(int, pa.Check.in_range(0, 200)),
    "num_procedures": pa.Column(int, pa.Check.in_range(0, 10)),
    "num_medications": pa.Column(int, pa.Check.in_range(0, 100)),
    "number_outpatient": pa.Column(int, pa.Check.in_range(0, 60)),
    "number_emergency": pa.Column(int, pa.Check.in_range(0, 100)),
    "number_inpatient": pa.Column(int, pa.Check.in_range(0, 30)),
    "diag_1": pa.Column(str, nullable=True),
    "diag_2": pa.Column(str, nullable=True),
    "diag_3": pa.Column(str, nullable=True),
    "number_diagnoses": pa.Column(int, pa.Check.in_range(1, 20)),
    "max_glu_serum": pa.Column(str, pa.Check.isin(["None", "Norm", ">200", ">300"]), nullable=True),
    "A1Cresult": pa.Column(str, pa.Check.isin(["None", "Norm", ">7", ">8"]), nullable=True),
    "change": pa.Column(str, pa.Check.isin(["No", "Ch"])),
    "diabetesMed": pa.Column(str, pa.Check.isin(["Yes", "No"])),
    "readmitted": pa.Column(str, pa.Check.isin(["<30", ">30", "NO"])),
}
for med in MED_COLS:
    columns[med] = pa.Column(str, pa.Check.isin(MED_VALUES))

# strict=False = มีคอลัมน์เกินได้ (เช่น weight, payer_code ที่เราไม่ใช้)
RAW_SCHEMA = pa.DataFrameSchema(columns, strict=False, coerce=False)


class DataValidationError(RuntimeError):
    pass


def read_raw(file) -> pd.DataFrame:
    # ต้องอ่านแบบนี้เพราะ
    # - "None" ในคอลัมน์ผลแล็บแปลว่า "ไม่ได้ตรวจ" ไม่ใช่ค่าหาย ห้ามให้ pandas แปลงเป็น NaN
    # - รหัส ICD-9 เป็น string (มี "V57", "250.83") ถ้าไม่บังคับ dtype อาจโดนแปลงเป็นตัวเลข
    df = pd.read_csv(file, keep_default_na=False, na_values=[""], low_memory=False,
                     dtype={"diag_1": str, "diag_2": str, "diag_3": str})
    # UCI เวอร์ชันใหม่ใช้ช่องว่างแทน "None" เลยเติมกลับให้เหมือนกัน
    for c in ["max_glu_serum", "A1Cresult"]:
        if c in df.columns:
            df[c] = df[c].fillna("None")
    return df


def validate(df: pd.DataFrame, params: dict | None = None, raise_on_error: bool = True) -> dict:
    if params is None:
        params = load_params()
    issues = []

    # ชั้นที่ 1: pandera (lazy=True = เก็บ error ทุกอันก่อนแล้วค่อยรายงาน ไม่หยุดที่อันแรก)
    try:
        RAW_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as e:
        fc = e.failure_cases
        for (col, check), grp in fc.groupby(["column", "check"], dropna=False):
            issues.append({
                "type": "schema",
                "column": str(col),
                "check": str(check),
                "n_failures": int(len(grp)),
                "examples": [str(x) for x in grp["failure_case"].head(5)],
            })

    # ชั้นที่ 2: เช็คทั้งชุด
    if len(df) < params["data"]["min_rows"]:
        issues.append({"type": "dataset", "check": "min_rows", "value": len(df)})

    cols_to_check = [c for c in RAW_SCHEMA.columns if c in df.columns]
    missing_rate = df[cols_to_check].replace("?", pd.NA).isna().mean()  # ในชุดนี้ "?" คือค่าหาย
    for col, rate in missing_rate.items():
        if rate > params["data"]["max_missing_rate"]:
            issues.append({"type": "dataset", "check": "missing_rate", "column": col, "value": round(float(rate), 3)})

    if "readmitted" in df.columns:
        # ปกติอัตรากลับมาภายใน 30 วันอยู่ราว 11% ถ้าหลุดช่วงนี้แปลว่าข้อมูลน่าจะผิด
        pos = (df["readmitted"] == "<30").mean()
        if pos < 0.02 or pos > 0.5:
            issues.append({"type": "dataset", "check": "label_rate", "value": round(float(pos), 4)})

    report = {"n_rows": int(len(df)), "n_issues": len(issues), "passed": len(issues) == 0, "issues": issues}
    out = path("reports/validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if issues:
        send_alert("DATA VALIDATION FAILED", report)
        if raise_on_error:
            raise DataValidationError(f"{len(issues)} data issue(s): see reports/validation.json")
    return report


def main() -> None:
    # ใช้: python -m readmit.data.schema [ไฟล์.csv]
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="ไฟล์ csv ที่จะตรวจ (ไม่ใส่ = ไฟล์ raw หลัก)")
    args = ap.parse_args()
    f = args.file or path(load_params()["data"]["raw_file"])
    try:
        rep = validate(read_raw(f))
        print(f"VALIDATION PASSED ({rep['n_rows']} rows)")
    except DataValidationError as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
