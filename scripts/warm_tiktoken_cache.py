#!/usr/bin/env python3
"""Populate skynet/cache/tiktoken_cache/ with the encoding files tiktoken
would otherwise fetch on first use.

Run this on a machine with outbound access to
openaipublic.blob.core.windows.net, then commit the resulting files so the
firewalled Docker build and runtime can find them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENCODINGS = ('o200k_base', 'cl100k_base', 'p50k_base', 'r50k_base')


def main() -> int:
    target = Path(__file__).resolve().parent.parent / 'skynet' / 'cache' / 'tiktoken_cache'
    target.mkdir(parents=True, exist_ok=True)
    os.environ['TIKTOKEN_CACHE_DIR'] = str(target)

    import tiktoken  # imported after TIKTOKEN_CACHE_DIR is set

    for name in ENCODINGS:
        print(f'warming {name} into {target}')
        tiktoken.get_encoding(name)

    written = sorted(p.name for p in target.iterdir() if p.is_file() and p.name != 'README.md')
    print(f'cache now contains {len(written)} file(s): {written}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
