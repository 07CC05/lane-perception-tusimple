# -*- coding: utf-8 -*-
"""Download the TuSimple sample used in this experiment.

Images are fetched one by one from a public mirror (the official S3 links are
dead).  Nothing is re-distributed in this repository.

    python scripts/fetch_sample.py
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import time
import urllib.request

BASE = "https://hf-mirror.com/datasets/dotblue/TuSimple/resolve/main/"
LABELS = ["label_data_0313.json", "label_data_0531.json", "label_data_0601.json"]
FRAMES = [1, 5, 10, 15, 20]


def get(url: str, path: str, ctx) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=90) as r, \
                    open(path, "wb") as fh:
                fh.write(r.read())
            return
        except Exception as exc:                     # noqa: BLE001
            if attempt == 2:
                raise SystemExit(f"failed: {url} ({exc})")
            time.sleep(1)


def main() -> None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE       # the local proxy re-signs TLS

    for name in LABELS:
        get(BASE + name, os.path.join("data", "labels", name), ctx)
        print("label", name)

    selected = json.load(open("data/selected.json", encoding="utf-8"))
    for item in selected:
        seq = item["seq"]
        for i in FRAMES:
            get(f"{BASE}{seq}/{i}.jpg",
                os.path.join("data", "raw", seq, f"{i}.jpg"), ctx)
        print("clip ", seq)


if __name__ == "__main__":
    sys.exit(main())
