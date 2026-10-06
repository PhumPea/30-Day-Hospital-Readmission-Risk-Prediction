# ระบบพยากรณ์ความเสี่ยงการกลับมารักษาซ้ำภายใน 30 วัน (CP413008)

ระบบทำนายว่าผู้ป่วยเบาหวานที่กำลังจะกลับบ้าน มีโอกาสกลับมานอนโรงพยาบาลอีกภายใน 30 วันแค่ไหน
พยาบาลจะได้เลือกโทรติดตามคนที่เสี่ยงที่สุดประมาณ 20% ก่อน

ข้อมูลที่ใช้คือ Diabetes 130-US Hospitals 1999–2008 จาก UCI (101,766 ครั้งการนอน รพ.)

## วิธีรัน (เครื่องเปล่า มีแค่ Docker)

```bash
git clone <repo> && cd readmission-mlops
docker compose up --build
```

ถ้าไม่มีเน็ตหรือโหลดข้อมูลจาก UCI ไม่ได้ ให้ใช้ข้อมูลจำลองแทน

```bash
PIPELINE_ARGS=--synthetic docker compose up --build
```

คำสั่งเดียวนี้รันครบทั้งวงจร: ข้อมูลดิบ → ตรวจคุณภาพ → แบ่งข้อมูล → เทรน (LACE baseline + 4 การทดลอง) → gate → registry → เปิด API → เริ่ม monitoring

ครั้งแรกใช้เวลาประมาณ 5–10 นาที (build image + เทรน)

| Service | URL |
|---|---|
| API (Swagger ลองยิงได้เลย) | http://localhost:8000/docs |
| MLflow (ผลการทดลอง + registry) | http://localhost:5000 |
| Prefect (ดู DAG) | http://localhost:4200 |
| Prometheus (alerts) | http://localhost:9090/alerts |
| Grafana (dashboard) | http://localhost:3000 |

ลองยิง API:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @data/bad_samples/good_request.json
```

## ขั้นตอนเดโมวันนำเสนอ

| สิ่งที่จะโชว์ | คำสั่ง | ผลที่ควรเห็น |
|---|---|---|
| ข้อมูลเสีย → pipeline หยุด | `make validate-bad` | `VALIDATION FAILED` + มีบรรทัดใหม่ใน `logs/alerts.jsonl` |
| request เสีย → ถูกปฏิเสธ | `make demo-bad-api` | ได้ 422 ทุกอัน |
| วัด latency / throughput เทียบ SLO | `make benchmark` | `reports/benchmark.json` |
| traffic ปกติ | `make demo-normal && make monitor` | ไม่มี drift |
| data drift | `make demo-data-drift && make monitor` | `data_drift: true` และรอ label ก่อนเทรนใหม่ |
| concept drift → เทรนใหม่ | `make demo-concept-drift && make retrain` | เจอ concept drift → ได้โมเดลเวอร์ชันใหม่ → ผ่าน gate → ขึ้นเป็น champion |
| ดู registry | `make versions` | เวอร์ชัน สถานะ alias |
| rollback | `make rollback` | champion ถอยไปตัวก่อน API โหลดใหม่เองใน 30 วิ (ดูที่ `/model`) |

ถ้าเครื่องไม่มี `make` ให้เปิด `Makefile` แล้วก๊อปคำสั่งไปรันตรง ๆ

## โครงสร้างโปรเจค

```
src/readmit/data/        ingest.py (โหลดข้อมูล), schema.py (Pandera), split.py (แบ่งตามผู้ป่วย)
src/readmit/features/    preprocess.py  <- feature ชุดเดียวใช้ทั้งเทรนและ serve
src/readmit/models/      lace.py (baseline), train.py (MLflow), evaluate.py, registry.py (gate/promote/rollback)
src/readmit/monitoring/  drift.py (PSI), monitor.py, simulate.py (จำลอง traffic)
pipelines/               flow.py (DAG เทรน), retrain.py (DAG ตรวจ drift + เทรนใหม่), scheduler.py
api/                     FastAPI, schemas.py (รูปแบบ request), model_store.py (โหลดโมเดลอัตโนมัติ)
infra/                   Prometheus (scrape + alert rules), Grafana (dashboard)
tests/                   pytest
.github/workflows/ci.yml CI: โค้ด + ข้อมูล + โมเดล + docker
params.yaml              ค่าตั้งค่า/เกณฑ์ทั้งหมดอยู่ที่นี่
docs/                    canvas, สถาปัตยกรรม, รายงาน
```

## เครื่องมือที่ใช้

| หน้าที่ | เครื่องมือ |
|---|---|
| Version Control | Git + GitHub (branch + PR) |
| Containerization | Docker + docker compose |
| Data Validation | Pandera + เช็คระดับชุดข้อมูลที่เขียนเอง |
| Experiment Tracking / Model Registry | MLflow (alias `champion` / `previous`) |
| Pipeline Orchestration | Prefect 3 |
| Model Serving | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana + drift (PSI) เขียนเอง |
| CI/CD | GitHub Actions (+ push image ไป GHCR) |

## การรันซ้ำได้ผลเดิม

- เวอร์ชันไลบรารีล็อคไว้หมดใน `requirements.lock.txt` และทุก service ใช้ Docker image เดียวกัน
- `seed: 42` และแบ่งข้อมูลด้วย hash ของเลขผู้ป่วย
- data version = hash ของไฟล์ดิบ + วิธีแบ่ง บันทึกไว้กับทุก run

## รันบนเครื่องตัวเอง (ไม่ใช้ Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt && pip install --no-deps -e .
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
#$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
python -m pipelines.flow --synthetic
uvicorn api.main:app --port 8000
pytest -q && ruff check .
```

## วิธีทำงานกับ Git

- ห้าม push เข้า `main` ตรง ๆ ให้แตก branch `feature/<ชื่อ>` → เปิด PR → CI ต้องผ่าน → ให้เพื่อนรีวิว → merge
- ถ้าอยากได้หลักฐาน CI ไม่ผ่าน: เปิด PR ที่แก้ `gate.min_pr_auc: 0.9` ใน `params.yaml` แล้ว model gate จะ fail
