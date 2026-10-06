# ทางลัดสำหรับเดโม (ไม่มี make ก็ก๊อปคำสั่งข้างในไปรันเองได้)
.PHONY: up up-synthetic down reset test lint validate-bad demo-bad-api demo-normal demo-data-drift \
        demo-concept-drift monitor retrain retrain-force rollback versions benchmark logs-reset

up:                  ## ทั้งระบบ ข้อมูลจริง
	docker compose up --build -d && docker compose logs -f pipeline
up-synthetic:        ## ทั้งระบบ ข้อมูลจำลอง
	PIPELINE_ARGS=--synthetic docker compose up --build -d && docker compose logs -f pipeline
down:
	docker compose down
reset:               ## ล้างทุกอย่าง (registry, ข้อมูล, log)
	docker compose down -v && rm -rf data/raw/*.csv data/processed/* reports/* logs/*.jsonl

test:
	docker compose run --rm --no-deps api pytest -q
lint:
	docker compose run --rm --no-deps api ruff check .

validate-bad:        ## ไฟล์ข้อมูลเสีย -> หยุด + แจ้งเตือน
	docker compose run --rm --no-deps api python -m readmit.data.schema data/bad_samples/bad_raw.csv
demo-bad-api:        ## request เสีย -> 422
	docker compose exec api python -m readmit.monitoring.simulate --api http://localhost:8000 --scenario bad_data
logs-reset:
	rm -f logs/predictions.jsonl logs/labels.jsonl
demo-normal: logs-reset
	docker compose exec api python -m readmit.monitoring.simulate --api http://localhost:8000 --scenario normal
demo-data-drift: logs-reset
	docker compose exec api python -m readmit.monitoring.simulate --api http://localhost:8000 --scenario data_drift --no-labels
demo-concept-drift: logs-reset
	docker compose exec api python -m readmit.monitoring.simulate --api http://localhost:8000 --scenario concept_drift
monitor:
	docker compose exec monitor python -m readmit.monitoring.monitor
retrain:             ## ตรวจ drift แล้วเทรนใหม่ถ้าจำเป็น
	docker compose exec monitor python -m pipelines.retrain
retrain-force:
	docker compose exec monitor python -m pipelines.retrain --force
rollback:
	docker compose exec monitor python -m readmit.models.registry rollback
versions:
	docker compose exec monitor python -m readmit.models.registry list
benchmark:
	docker compose exec api python scripts/benchmark.py --api http://localhost:8000 -n 2000 -c 16
