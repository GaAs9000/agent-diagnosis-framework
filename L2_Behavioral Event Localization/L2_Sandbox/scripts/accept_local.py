#!/usr/bin/env python3
"""Compatibility wrapper for the current l2 run command (dry-run by default)."""
from l2_sandbox.cli import main
if __name__ == '__main__':
    import sys
    raise SystemExit(main(['run', *sys.argv[1:]]))
