# กำหนดรูปแบบ request / response ของ API ด้วย Pydantic
# ข้อมูลผิดรูปแบบ (เช่น นอน รพ. 99 วัน, อายุเป็นคำว่า "old") จะโดนปฏิเสธ 422 ตั้งแต่ตรงนี้
# ไม่หลุดไปถึงโมเดล
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgeBin = Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)",
                 "[70-80)", "[80-90)", "[90-100)"]
Med = Literal["No", "Steady", "Up", "Down"]  # สถานะยา: ไม่ใช้ / คงเดิม / เพิ่ม / ลด


class Encounter(BaseModel):
    # ข้อมูลคนไข้ 1 ครั้งที่มานอน รพ. ณ วันที่จำหน่ายกลับบ้าน
    # populate_by_name: รับได้ทั้งชื่อ field และ alias (ชื่อยาที่มีขีด เช่น glyburide-metformin)
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    encounter_id: int = Field(gt=0)
    race: str | None = None
    gender: Literal["Female", "Male", "Unknown/Invalid"]
    age: AgeBin
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(ge=1, le=30)
    admission_source_id: int = Field(ge=1, le=26)
    time_in_hospital: int = Field(ge=1, le=14)
    medical_specialty: str | None = None
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=10)
    num_medications: int = Field(ge=0, le=100)
    number_outpatient: int = Field(ge=0, le=60)
    number_emergency: int = Field(ge=0, le=100)
    number_inpatient: int = Field(ge=0, le=30)
    diag_1: str | None = None
    diag_2: str | None = None
    diag_3: str | None = None
    number_diagnoses: int = Field(ge=1, le=20)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] | None = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] | None = "None"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["Yes", "No"] = "No"

    # ยา: ไม่ส่งมา = "No" (ไม่ได้ใช้)
    metformin: Med = "No"
    repaglinide: Med = "No"
    nateglinide: Med = "No"
    chlorpropamide: Med = "No"
    glimepiride: Med = "No"
    acetohexamide: Med = "No"
    glipizide: Med = "No"
    glyburide: Med = "No"
    tolbutamide: Med = "No"
    pioglitazone: Med = "No"
    rosiglitazone: Med = "No"
    acarbose: Med = "No"
    miglitol: Med = "No"
    troglitazone: Med = "No"
    tolazamide: Med = "No"
    examide: Med = "No"
    citoglipton: Med = "No"
    insulin: Med = "No"
    glyburide_metformin: Med = Field("No", alias="glyburide-metformin")
    glipizide_metformin: Med = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: Med = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Med = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: Med = Field("No", alias="metformin-pioglitazone")

    def to_row(self) -> dict:
        # แปลงกลับเป็นชื่อคอลัมน์แบบเดียวกับข้อมูลเทรน (ใช้ alias ที่มีขีด)
        return self.model_dump(by_alias=True)


class Prediction(BaseModel):
    encounter_id: int
    risk_score: float     # ความน่าจะเป็นที่จะกลับมาภายใน 30 วัน
    follow_up: bool       # True = ควรโทรติดตาม (อยู่ใน top 20%)
    model_version: str


class BatchRequest(BaseModel):
    encounters: list[Encounter] = Field(min_length=1, max_length=1000)


class Feedback(BaseModel):
    # label จริงที่ส่งกลับมาภายหลัง (ครบ 30 วันแล้วรู้ผล)
    encounter_id: int = Field(gt=0)
    readmitted_30d: bool
