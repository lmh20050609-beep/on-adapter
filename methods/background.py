"""Run the original queue/observer with explicit logs under windowless Python.

This entry is launched by Windows WMI, independently of a transient terminal.
Training modules remain unchanged; their original command tokens are retained
in the process command line so the existing observer can identify them.
"""
import argparse
import ctypes
import os
from pathlib import Path
import runpy
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('-m', dest='module', choices=['method_comparison_periodic_v3.queue'])
    target.add_argument('--script', choices=[])
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / 'scripts')]
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1',
                      TOKENIZERS_PARALLELISM='false')
    if args.script:
        os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('a', encoding='utf-8', buffering=1) as stream:
        sys.stdout = stream
        sys.stderr = stream
        ctypes.windll.kernel32.GetConsoleWindow.restype = ctypes.c_void_p
        print('BACKGROUND_ENTRY', os.getpid(), 'executable', sys.executable,
              'console_window', ctypes.windll.kernel32.GetConsoleWindow() or 0, flush=True)
        try:
            if args.module:
                sys.argv = [args.module] + remaining
                runpy.run_module(args.module, run_name='__main__')
            else:
                path = root / args.script
                sys.argv = [str(path)] + remaining
                runpy.run_path(str(path), run_name='__main__')
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
