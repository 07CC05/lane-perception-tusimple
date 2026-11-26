# -*- coding: utf-8 -*-
"""Stage 2: evaluate baseline and improved on the same fixed frame set.

Usage:
    python src/evaluate.py --selected data/selected.json --out results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_lib as L  # noqa: E402
from baseline import process_frame as run_baseline  # noqa: E402
from improved import process_frame_improved as run_improved  # noqa: E402

METHODS = {
    "baseline": (run_baseline, "baseline"),
    "improved": (run_improved, "improved"),
}


def scene_class(curv: float) -> str:
    if curv < 40:
        return "straight"
    if curv >= 150:
        return "curve"
    return "mixed"


def timeit(fn, *args, repeats=3, **kwargs):
    fn(*args, **kwargs)   # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default="data/selected.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tol", type=float, default=20.0)
    ap.add_argument("--frames", default="20", help="last frame id used for GT")
    args = ap.parse_args()

    sel = json.load(open(args.selected, encoding="utf-8"))
    labels = L.load_labels([
        "data/labels/label_data_0313.json",
        "data/labels/label_data_0531.json",
        "data/labels/label_data_0601.json",
    ])

    rows = []
    for s in sel:
        key = s["file"]
        img_path = os.path.join("data/raw", key)
        img = cv2.imread(img_path)
        if img is None:
            print("missing", img_path)
            continue
        sample = labels[key]
        ys = L.eval_ys(sample)
        gt_l, gt_r, n_lanes = L.ego_lanes(sample)
        curv = float(s["curv"])
        scene = scene_class(curv)

        for name, (fn, _label) in METHODS.items():
            res = fn(img, ys)
            m = L.frame_metrics(res["pred_left"], res["pred_right"],
                                gt_l, gt_r, ys, args.tol)
            ms = timeit(fn, img, ys, repeats=3)
            out_dir = os.path.join(args.out, name)
            L.ensure_dir(os.path.join(out_dir, "overlays"))
            L.ensure_dir(os.path.join(out_dir, "preds"))
            if name == "baseline":
                overlay = L.draw_baseline(img, res["exports"]["roi"],
                                          res["left_fit"], res["right_fit"],
                                          gt_l, gt_r, ys,
                                          title=f"baseline {s['seq'].split('/')[1]}/{s['seq'].split('/')[2]}")
            else:
                overlay = L.draw_improved(img, res["left_fit"], res["right_fit"],
                                          gt_l, gt_r, ys,
                                          title=f"improved {s['seq'].split('/')[1]}/{s['seq'].split('/')[2]}")
            seq_tag = f"{s['seq'].split('/')[1]}_{s['seq'].split('/')[2]}"
            cv2.imwrite(os.path.join(out_dir, "overlays", f"{seq_tag}.jpg"), overlay)
            payload = {
                "image": key, "role": s["role"], "scene": scene,
                "curvature": round(curv, 1), "gt_lanes": n_lanes,
                "ys": [int(y) for y in ys],
                "pred_left": None if res["pred_left"] is None else
                             [round(float(v), 2) for v in res["pred_left"]],
                "pred_right": None if res["pred_right"] is None else
                              [round(float(v), 2) for v in res["pred_right"]],
            }
            with open(os.path.join(out_dir, "preds", f"{seq_tag}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)

            rows.append({
                "seq": s["seq"], "frame": key.split("/")[-1], "role": s["role"],
                "scene": scene, "curvature": round(curv, 1),
                "method": name, "n_points": m["n_points"], "hits": m["hits"],
                "hit_rate": round(m["hit_rate"], 4),
                "hit_left": round(m["left_hit_rate"], 4) if not np.isnan(m["left_hit_rate"]) else "",
                "hit_right": round(m["right_hit_rate"], 4) if not np.isnan(m["right_hit_rate"]) else "",
                "mae_px": round(m["mae"], 2) if not np.isnan(m["mae"]) else "",
                "left_missed": int(m["left_missed"]),
                "right_missed": int(m["right_missed"]),
                "ms_median": round(ms, 2),
            })

    L.ensure_dir(args.out)
    csv_path = os.path.join(args.out, "per_frame.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path, len(rows), "rows")

    # ---------------- summary ----------------
    def pool(rs):
        n = sum(r["n_points"] for r in rs)
        h = sum(r["hits"] for r in rs)
        return (h / n if n else float("nan")), n, len(rs)

    lines = []
    header = ["method", "split", "scene", "frames", "points", "pooled_hit_rate",
              "mean_hit_rate", "mean_mae_px", "mean_ms"]
    for name in METHODS:
        for split in ("test", "tune"):
            for scene in ("all", "straight", "curve", "mixed"):
                rs = [r for r in rows if r["method"] == name and r["role"] == split]
                if scene != "all":
                    rs = [r for r in rs if r["scene"] == scene]
                if not rs:
                    continue
                pr, n, k = pool(rs)
                maes = [r["mae_px"] for r in rs if r["mae_px"] != ""]
                lines.append([name, split, scene, k, n, round(pr, 4),
                              round(float(np.mean([r["hit_rate"] for r in rs])), 4),
                              round(float(np.mean(maes)), 2) if maes else "",
                              round(float(np.mean([r["ms_median"] for r in rs])), 2)])
    sum_path = os.path.join(args.out, "summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(lines)
    print("wrote", sum_path)

    env = {
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tol_px": args.tol,
        "timing": "median of 3 runs per frame, warm-up included",
    }
    with open(os.path.join(args.out, "env.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, ensure_ascii=False, indent=1)
    print(json.dumps(env, ensure_ascii=False, indent=1))

    for line in [header] + lines:
        print(" | ".join(str(x) for x in line))


if __name__ == "__main__":
    main()
