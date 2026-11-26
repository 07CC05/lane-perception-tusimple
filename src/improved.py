# -*- coding: utf-8 -*-
"""Stage 3: one change only — replace the straight-line fit with a sliding
window search + quadratic fit.  Edge extraction (gray / gaussian / canny /
trapezoid ROI) is byte-for-byte identical to the baseline.

Usage:
    python src/improved.py --image data/raw/clips/.../20.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_lib as L  # noqa: E402


def process_frame_improved(img: np.ndarray, ys: np.ndarray,
                           verbose: bool = False) -> dict:
    t0 = time.perf_counter()
    gray, edges, edges_roi = L.preprocess(img)
    # Rough straight-line estimate, used only to initialise the search windows.
    segments = L.hough_segments(edges_roi)
    l_seg, r_seg = L.split_segments(segments, img.shape[1], img.shape[0])
    l_seg = L.select_ego_cluster(l_seg, "left", img.shape[1], img.shape[0])
    r_seg = L.select_ego_cluster(r_seg, "right", img.shape[1], img.shape[0])
    lf = L.fit_line(l_seg)
    rf = L.fit_line(r_seg)
    guide_l = None if lf is None else (lf.a, lf.b)
    guide_r = None if rf is None else (rf.a, rf.b)
    left_fit = L.sliding_window_fit(edges_roi, "left", guide_l)
    right_fit = L.sliding_window_fit(edges_roi, "right", guide_r)
    pred_l = L.predict_quad(left_fit, ys, img.shape[1])
    pred_r = L.predict_quad(right_fit, ys, img.shape[1])

    # Reliability guard: the quadratic is allowed to disagree with the rough
    # straight fit, but not to run away.  If more than 30% of the sampled
    # points differ by >70 px, fall back to the straight line for that side.
    line_l = L.predict(lf, ys, img.shape[1])
    line_r = L.predict(rf, ys, img.shape[1])

    def guarded(pred_q, pred_line):
        if pred_q is None:
            return pred_line
        if pred_line is None:
            return pred_q
        far = np.abs(pred_q - pred_line) > 70.0
        return pred_line if far.mean() > 0.30 else pred_q

    fell_back = {"left": pred_l is not None and line_l is not None
                 and guarded(pred_l, line_l) is line_l,
                 "right": pred_r is not None and line_r is not None
                 and guarded(pred_r, line_r) is line_r}
    pred_l = guarded(pred_l, line_l)
    pred_r = guarded(pred_r, line_r)
    dt = (time.perf_counter() - t0) * 1000.0

    info = {
        "roi_edges": int((edges_roi > 0).sum()),
        "guide_left": None if guide_l is None else [round(guide_l[0], 3), round(guide_l[1], 1)],
        "guide_right": None if guide_r is None else [round(guide_r[0], 3), round(guide_r[1], 1)],
        "left_points": 0 if left_fit is None else left_fit.n_points,
        "right_points": 0 if right_fit is None else right_fit.n_points,
        "left_coeffs": None if left_fit is None else [round(float(c), 6) for c in left_fit.coeffs],
        "right_coeffs": None if right_fit is None else [round(float(c), 6) for c in right_fit.coeffs],
        "fallback_left": bool(fell_back["left"]),
        "fallback_right": bool(fell_back["right"]),
        "left_missed": pred_l is None,
        "right_missed": pred_r is None,
        "ms": dt,
    }
    if verbose:
        print(f"[1] roi edges={info['roi_edges']} (same preprocessing as baseline)")
        print(f"[2] sliding window: left points={info['left_points']} "
              f"right points={info['right_points']}")
        print(f"[3] quadratic x=a*y^2+b*y+c  left={info['left_coeffs']}")
        print(f"                             right={info['right_coeffs']}")
        print(f"[4] missed left={info['left_missed']} right={info['right_missed']}"
              f"  {info['ms']:.1f} ms")
    return {"pred_left": pred_l, "pred_right": pred_r,
            "left_fit": left_fit, "right_fit": right_fit,
            "info": info, "exports": {"gray": gray, "edges": edges,
                                      "roi": edges_roi}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", nargs="*", default=["data/labels/label_data_0313.json"])
    ap.add_argument("--out", default="results/improved")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    labels = L.load_labels(args.labels)
    key = args.image.replace("\\", "/").split("data/raw/")[-1]
    sample = labels.get(key)
    if sample is None:
        raise SystemExit(f"no label for {key}")
    ys = L.eval_ys(sample)
    gt_l, gt_r, _ = L.ego_lanes(sample)

    print(f"frame: {key}  size={img.shape[1]}x{img.shape[0]}")
    res = process_frame_improved(img, ys, verbose=True)

    L.ensure_dir(os.path.join(args.out, "overlays"))
    L.ensure_dir(os.path.join(args.out, "preds"))
    overlay = L.draw_improved(img, res["left_fit"], res["right_fit"],
                              gt_l, gt_r, ys, title=f"improved  {key}")
    stem = os.path.splitext(os.path.basename(args.image))[0]
    seq = os.path.basename(os.path.dirname(args.image))
    tag = f"{seq}_{stem}"
    cv2.imwrite(os.path.join(args.out, "overlays", f"{tag}.jpg"), overlay)
    payload = {
        "image": key, "size": [img.shape[1], img.shape[0]],
        "ys": [int(y) for y in ys],
        "pred_left": None if res["pred_left"] is None else [round(float(v), 2) for v in res["pred_left"]],
        "pred_right": None if res["pred_right"] is None else [round(float(v), 2) for v in res["pred_right"]],
        "steps": res["info"],
    }
    with open(os.path.join(args.out, "preds", f"{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"saved -> {args.out}/overlays/{tag}.jpg")


if __name__ == "__main__":
    main()
