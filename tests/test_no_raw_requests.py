import pathlib, re


def test_no_raw_requests_in_src():
    root = pathlib.Path(__file__).resolve().parents[1] / "ai_trading"
    banned = []
    for p in root.rglob("*.py"):
        if "utils/http.py" in str(p):
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\brequests\.(get|post|put|delete|patch|head|options)\b", txt):
            banned.append(str(p))
    assert not banned, f"Raw requests.* found in: {banned}"
