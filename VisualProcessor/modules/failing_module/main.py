#!/usr/bin/env python3
"""
failing_module

Utility module for PR-4 evidence:
- always fails (exit code 2)
- used to demonstrate optional component failure without stopping the run
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    _ = argparse.ArgumentParser()
    # Accept the standard module CLI args so VisualProcessor can call it.
    _.add_argument("--frames-dir", required=True)
    _.add_argument("--rs-path", required=True)
    _.parse_args()

    print("failing_module: intentional failure (PR-4 optional failure evidence)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


