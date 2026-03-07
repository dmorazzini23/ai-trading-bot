import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_service_exits_cleanly():
    script = textwrap.dedent(
        '''
        import os, sys
        from ai_trading import main as m

        m.run_cycle = lambda: None
        m.start_api_with_signal = lambda ready, err: ready.set()
        m.time.sleep = lambda s: None
        os.environ["IMPORT_PREFLIGHT_DISABLED"] = "1"
        sys.exit(m.main(['--iterations','1','--interval','0']) or 0)
        '''
    )
    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        project_root if not existing_pythonpath else f"{project_root}{os.pathsep}{existing_pythonpath}"
    )
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
