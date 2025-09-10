import os
import subprocess
import sys
import textwrap


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
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
