# fixture ที่ใช้ร่วมกันทุกเทสต์
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ.setdefault("PROJECT_ROOT", str(ROOT))


@pytest.fixture(scope="session")
def synth():
    from readmit.data.ingest import make_synthetic
    return make_synthetic(6000, seed=0)


@pytest.fixture(scope="session")
def fitted_pipe(synth):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from readmit.data.split import TARGET, make_target
    from readmit.features.preprocess import INPUT_COLS, build_preprocessor
    df = make_target(synth, [11, 13, 14, 19, 20, 21])
    pipe = Pipeline([("features", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    return pipe.fit(df[INPUT_COLS], df[TARGET]), df
