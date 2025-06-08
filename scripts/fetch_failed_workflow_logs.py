#!/usr/bin/env python3
"""Download logs for recent failed GitHub Actions runs and report failing steps."""
import argparse, io, os, zipfile
from typing import Iterable, Set
import requests

REPO = "dmorazzini23/ai-trading-bot"
API = "https://api.github.com"

def _get_failed_steps(zip_bytes: bytes) -> Set[str]:
    failed = set()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            with zf.open(name) as fp:
                text = fp.read().decode("utf-8", errors="ignore")
                if "##[error]" in text or "Process completed with exit code" in text:
                    step = os.path.splitext(os.path.basename(name))[0]
                    failed.add(step)
    return failed

def list_failed_runs(token: str) -> Iterable[dict]:
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    params = {"status": "failure", "per_page": 10}
    resp = requests.get(f"{API}/repos/{REPO}/actions/runs", headers=headers, params=params)
    resp.raise_for_status()
    return resp.json().get("workflow_runs", [])

def download_logs(run_id: int, token: str) -> bytes:
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    resp = requests.get(f"{API}/repos/{REPO}/actions/runs/{run_id}/logs", headers=headers, allow_redirects=True)
    resp.raise_for_status()
    return resp.content

def fetch_failed_steps(run_id: int, token: str) -> Set[str]:
    steps = _get_failed_steps(download_logs(run_id, token))
    if steps:
        return steps
    # fallback to jobs API
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    resp = requests.get(f"{API}/repos/{REPO}/actions/runs/{run_id}/jobs", headers=headers)
    if resp.ok:
        for job in resp.json().get("jobs", []):
            for step in job.get("steps", []):
                if step.get("conclusion") == "failure":
                    steps.add(step["name"])
    return steps

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token", help="GitHub token with repo scope")
    args = p.parse_args()
    token = args.token or os.getenv("GITHUB_TOKEN")
    if not token:
        p.error("You must supply --token or set GITHUB_TOKEN")
    for run in list_failed_runs(token):
        print(f"\nRun {run['id']} – {run.get('name')}\nURL: {run.get('html_url')}")
        steps = fetch_failed_steps(run["id"], token)
        if steps:
            print("  Failed steps:")
            for s in sorted(steps):
                print(f"   • {s}")
        else:
            print("  [No failed steps detected]")

if __name__ == "__main__":
    main()

