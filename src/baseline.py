# -*- coding: utf-8 -*-
"""Stage 1: one frame, one pipeline.

Usage:
    python src/baseline.py --image data/raw/clips/.../20.jpg --labels data/labels/label_data_0313.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_lib as L  # noqa: E402


def process_frame(img: np.ndarray, ys: np.ndarray, tol: float = 20.0,
                  verbose: bool = False) -> dict:
    """Run the baseline chain on one BGR frame. Pure function, used by eval too."""
    t0 = time.perf_counter()
    gray, edges, edges_roi = L.preprocess(img)
    segments = L.hough_segments(edges_roi)
    left_seg, right_seg = L.split_segments(segments, img.shape[1], img.shape[0])
    left_seg = L.select_ego_cluster(left_seg, "left", img.shape[1], img.shape[0])
    right_seg = L.select_ego_cluster(right_seg, "right", img.shape[1], img.shape[0])
    left_fit = L.fit_line(left_seg)
    right_fit = L.fit_line(right_seg)
    pred_l = L.predict(left_fit, ys, img.shape[1])
    pred_r = L.predict(right_fit, ys, img.shape[1])
    dt = (time.perf_counter() - t0) * 1000.0

    info = {
        "gray": gray.shape,
        "canny_edges": int((edges > 0).sum()),
        "roi_edges": int((edges_roi > 0).sum()),
        "hough_segments": len(segments),
        "left_segments": len(left_seg),
        "right_segments": len(right_seg),
        "left_fit": None if left_fit is None else [round(left_fit.a, 5), round(left_fit.b, 2)],
        "right_fit": None if right_fit is None else [round(right_fit.a, 5), round(right_fit.b, 2)],
        "left_missed": pred_l is None,
        "right_missed": pred_r is None,
        "ms": dt,
    }
    if verbose:
        print(f"[1] gray {info['gray']}  canny edges={info['canny_edges']}")
        print(f"[2] roi edges={info['roi_edges']}  hough segments={info['hough_segments']}")
        print(f"[3] split: left={info['left_segments']}  right={info['right_segments']}")
        print(f"[4] fit  left x=a*y+b: {info['left_fit']}   right: {info['right_fit']}")
        print(f"[5] missed left={info['left_missed']} right={info['right_missed']}"
              f"  {info['ms']:.1f} ms")
    return {"pred_left": pred_l, "pred_right": pred_r,
            "left_fit": left_fit, "right_fit": right_fit,
            "info": info,
            "exports": {"gray": gray, "edges": edges, "roi": edges_roi}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", nargs="*", default=["data/labels/label_data_0313.json"])
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--tol", type=float, default=20.0)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    labels = L.load_labels(args.labels)
    key = args.image.replace("\\", "/").split("data/raw/")[-1]
    sample = labels.get(key)
    if sample is None:
        raise SystemExit(f"no label for {key}; pass the right --labels file")

    ys = L.eval_ys(sample)
    gt_l, gt_r, _ = L.ego_lanes(sample)

    print(f"frame: {key}  size={img.shape[1]}x{img.shape[0]}  "
          f"gt points: left={len(gt_l)} right={len(gt_r)}")
    res = process_frame(img, ys, args.tol, verbose=True)

    out_dir = args.out
    L.ensure_dir(os.path.join(out_dir, "overlays"))
    L.ensure_dir(os.path.join(out_dir, "preds"))

    overlay = L.draw_baseline(img, res["exports"]["roi"], res["left_fit"],
                              res["right_fit"], gt_l, gt_r, ys,
                              title=f"baseline  {key}")
    stem = os.path.splitext(os.path.basename(args.image))[0]
    seq = os.path.basename(os.path.dirname(args.image))
    tag = f"{seq}_{stem}"
    cv2.imwrite(os.path.join(out_dir, "overlays", f"{tag}.jpg"), overlay)

    payload = {
        "image": key,
        "size": [img.shape[1], img.shape[0]],
        "ys": [int(y) for y in ys],
        "pred_left": None if res["pred_left"] is None else [round(float(v), 2) for v in res["pred_left"]],
        "pred_right": None if res["pred_right"] is None else [round(float(v), 2) for v in res["pred_right"]],
        "steps": res["info"],
    }
    with open(os.path.join(out_dir, "preds", f"{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"saved overlay -> {out_dir}/overlays/{tag}.jpg")
    print(f"saved coords  -> {out_dir}/preds/{tag}.json")


if __name__ == "__main__":
    main()
