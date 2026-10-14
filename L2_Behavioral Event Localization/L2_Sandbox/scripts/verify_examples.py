#!/usr/bin/env python3
"""List historical examples; v0.1 verification is unavailable in v0.3."""
from pathlib import Path
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('examples'))
    args = parser.parse_args()
    print(json.dumps({'status': 'historical_examples_not_verified',
                      'reason': 'v0.1 package verifier is not part of lean v0.3',
                      'directory': str(args.directory)}, indent=2))
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
