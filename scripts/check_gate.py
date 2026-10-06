# ใช้ใน CI: ถ้าโมเดลไม่ผ่าน gate ให้ CI fail
import json
import sys

from readmit.config import path

g = json.loads(path("reports/gate.json").read_text())
print(json.dumps(g, indent=2))

if not g["passed"]:
    failed = [k for k, ok in g["checks"].items() if not ok]
    print(f"::error::Model quality gate FAILED: {failed}")  # format นี้ GitHub Actions จะขึ้นเป็นกล่องแดง
    sys.exit(1)

print("Model quality gate PASSED")
