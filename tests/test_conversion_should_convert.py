import os
import tempfile

from ai_trading.meta_learning.conversion import should_convert


def _write(path: str, content: str) -> str:
    with open(path, 'w') as f:
        f.write(content)
    return path


def test_should_convert_pure_meta():
    header = "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n"
    rows = (
        "TEST,2025-08-05T23:17:35Z,100,2025-08-05T23:18:35Z,105,10,buy,strat,test,tag,0.8,5\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        _write(f.name, header + rows)
        path = f.name
    try:
        assert should_convert(path) is True
    finally:
        os.unlink(path)


def test_should_convert_pure_audit():
    header = "order_id,timestamp,symbol,side,qty,price,mode,status\n"
    rows = (
        "1,2025-08-05T23:17:35Z,TEST,buy,10,100,live,filled\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        _write(f.name, header + rows)
        path = f.name
    try:
        assert should_convert(path) is True
    finally:
        os.unlink(path)


def test_should_convert_mixed():
    header = "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n"
    meta_row = "TEST,2025-08-05T23:17:35Z,100,2025-08-05T23:18:35Z,105,10,buy,strat,test,tag,0.8,5\n"
    audit_row = "1,2025-08-05T23:19:35Z,TEST,buy,10,100,live,filled\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        _write(f.name, header + meta_row + audit_row)
        path = f.name
    try:
        assert should_convert(path) is True
    finally:
        os.unlink(path)


def test_should_convert_problem_exact():
    header = "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n"
    rows = [
        "TEST,2025-08-05T23:17:35Z,100,2025-08-05T23:18:35Z,105,10,buy,strat,test,tag1+tag2,0.8,5\n",
        "AAPL,2025-08-05T23:19:35Z,150,2025-08-05T23:20:35Z,155,5,buy,strat,test,tag3,0.7,25\n",
        "MSFT,2025-08-05T23:21:35Z,300,2025-08-05T23:22:35Z,295,2,sell,strat,test,tag4,0.6,-10\n",
        "GOOGL,2025-08-05T23:23:35Z,2500,2025-08-05T23:24:35Z,2505,1,buy,strat,test,tag5,0.9,5\n",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        _write(f.name, header + "".join(rows))
        path = f.name
    try:
        assert should_convert(path) is True
    finally:
        os.unlink(path)
