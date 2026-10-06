# Load test: วัด latency p50/p95 และ throughput ของ API จริง แล้วเทียบกับ SLO ใน params.yaml
# ใช้: python scripts/benchmark.py --api http://localhost:8000 -n 2000 -c 16
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import numpy as np

from readmit.config import load_params, path


async def worker(client, payload, n, latencies, errors):
    for _ in range(n):
        start = time.perf_counter()
        try:
            r = await client.post("/predict", json=payload)
            if r.status_code != 200:
                errors.append(r.status_code)
        except Exception:
            errors.append("exception")
        latencies.append((time.perf_counter() - start) * 1000)


async def main(api: str, total: int, concurrency: int) -> dict:
    payload = json.loads(path("data/bad_samples/good_request.json").read_text())
    latencies = []
    errors = []

    async with httpx.AsyncClient(base_url=api, timeout=10) as client:
        await client.post("/predict", json=payload)  # warm-up ครั้งแรกมักช้า ไม่นับ
        t0 = time.perf_counter()
        per_worker = total // concurrency
        # ยิงพร้อมกัน concurrency คน
        await asyncio.gather(*[worker(client, payload, per_worker, latencies, errors) for _ in range(concurrency)])
        duration = time.perf_counter() - t0

    slo = load_params()["slo"]
    res = {
        "requests": len(latencies),
        "concurrency": concurrency,
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "throughput_rps": round(len(latencies) / duration, 1),
        "error_rate": round(len(errors) / len(latencies), 4),
        "slo": slo,
    }
    res["slo_met"] = {
        "p50": res["p50_ms"] <= slo["p50_latency_ms"],
        "p95": res["p95_ms"] <= slo["p95_latency_ms"],
        "throughput": res["throughput_rps"] >= slo["min_throughput_rps"],
        "errors": res["error_rate"] <= slo["error_rate"],
    }
    res["all_slo_met"] = all(res["slo_met"].values())
    path("reports/benchmark.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("-n", "--total", type=int, default=2000)
    ap.add_argument("-c", "--concurrency", type=int, default=16)
    args = ap.parse_args()
    print(json.dumps(asyncio.run(main(args.api, args.total, args.concurrency)), indent=2))
