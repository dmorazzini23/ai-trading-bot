from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement or requirement.startswith("-"):
            continue
        names.add(Requirement(requirement).name.lower().replace("_", "-"))
    return names


def test_project_runtime_dependencies_cover_requirements_txt() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_names = {
        Requirement(dependency).name.lower().replace("_", "-")
        for dependency in project["project"]["dependencies"]
    }

    assert _requirement_names(ROOT / "requirements.txt") <= dependency_names


def test_ta_extra_and_package_data_include_required_assets() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    ta_extra = {
        Requirement(dependency).name.lower().replace("_", "-")
        for dependency in project["project"]["optional-dependencies"]["ta"]
    }
    assert "pandas-ta" in ta_extra

    package_data = project["tool"]["setuptools"]["package-data"]
    assert "tickers.csv" in package_data["ai_trading.data"] or "*.csv" in package_data["ai_trading.data"]


def test_runtime_workflows_use_ubuntu_2404_constraints_and_block_scheduled_gates() -> None:
    workflow_dir = ROOT / ".github" / "workflows"
    workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in workflow_dir.glob("*.yml"))
    assert "ubuntu-22.04" not in workflow_text

    ci_text = (workflow_dir / "ci.yml").read_text(encoding="utf-8")
    assert "pip install -c constraints.txt -r requirements-dev.txt" in ci_text
    assert re.search(r"nightly-full-deps:\n(?:[^\n]*\n){0,4}\s+continue-on-error:", ci_text) is None

    audit_text = (workflow_dir / "dependency-audit.yml").read_text(encoding="utf-8")
    assert "continue-on-error:" not in audit_text
    assert "pip-audit -r requirements.txt -c constraints.txt" in audit_text


def test_bot_engine_does_not_install_requests_or_session_shims() -> None:
    source = (ROOT / "ai_trading" / "core" / "bot_engine.py").read_text(encoding="utf-8")

    forbidden = (
        "_REQUESTS_STUB",
        "_HTTP_SESSION_STUB",
        "RequestExceptionFallback",
        "RequestsStub",
        "HTTPStub",
        "requests not installed",
        "HTTPSession unavailable",
    )
    assert not any(marker in source for marker in forbidden)
