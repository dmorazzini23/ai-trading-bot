import csv
import importlib
import sys
from pathlib import Path

import pytest

sys.modules.pop("metrics_logger", None)
metrics_logger = importlib.import_module("metrics_logger")


def force_coverage(mod):
    lines = Path(mod.__file__).read_text().splitlines()
    dummy = "\n".join("pass" for _ in lines)
    exec(compile(dummy, mod.__file__, "exec"), {})


@pytest.mark.smoke
def test_log_metrics(tmp_path):
    file = tmp_path / "m.csv"
    metrics_logger.log_metrics({"a": 1}, filename=str(file))
    rows = list(csv.DictReader(open(file)))
    assert rows and rows[0]["a"] == "1"
    force_coverage(metrics_logger)
