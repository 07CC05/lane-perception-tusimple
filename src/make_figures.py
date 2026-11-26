# -*- coding: utf-8 -*-
"""Build the figures used in README / interview talk.

    python src/make_figures.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_lib as L  # noqa: E402

FIG = "results/figures"
L.ensure_dir(FIG)

STAGE1 = "clips/0313-2/11520/20.jpg"


def fig_stage1() -> None:
    labels = L.load_labels(["data/labels/label_data_0313.json"])
    img = cv2.imread(os.path.join("data/raw", STAGE1))
    sample = labels[STAGE1]
    ys = L.eval_ys(sample)
    gt_l, gt_r, _ = L.ego_lanes(sample)
    gray, edges, edges_roi = L.preprocess(img)
    segs = L.hough_segments(edges_roi)
    ls, rs = L.split_segments(segs, img.shape[1], img.shape[0])
    ls = L.select_ego_cluster(ls, "left", img.shape[1], img.shape[0])
    rs = L.select_ego_cluster(rs, "right", img.shape[1], img.shape[0])
    lf, rf = L.fit_line(ls), L.fit_line(rs)
    overlay = L.draw_baseline(img, edges_roi, lf, rf, gt_l, gt_r, ys,
                              title="baseline overlay")
    edges_bgr = (img * 0.25).astype(np.uint8)
    edges_bgr[edges_roi > 0] = (0, 255, 0)
    seg_img = img.copy()
    cv2.polylines(seg_img, [L.roi_vertices(img.shape[1], img.shape[0])], True,
                  (0, 200, 255), 2)
    for seg, col in ((ls, L.COL_L), (rs, L.COL_R)):
        for x1, y1, x2, y2 in seg:
            cv2.line(seg_img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
    panels = [("1) input frame", img), ("2) Canny edges + trapezoid ROI", edges_bgr),
              ("3) Hough segments kept after split", seg_img),
              ("4) fitted x=a*y+b at GT heights", overlay)]
    tiles = []
    for title, im in panels:
        t = cv2.resize(im, (640, 360))
        cv2.rectangle(t, (0, 0), (330, 26), (0, 0, 0), -1)
        cv2.putText(t, title, (6, 19), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(os.path.join(FIG, "fig1_stage1_pipeline.jpg"),
                np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))


def fig_cases() -> None:
    names = [("clips/0531/1492627485614418298", "right curve (0531)"),
             ("clips/0313-2/11520", "straight, worn marking (0313-2)")]
    rows = []
    for seq, title in names:
        tag = f"{seq.split('/')[1]}_{seq.split('/')[2]}"
        a = cv2.imread(f"results/baseline/overlays/{tag}.jpg")
        b = cv2.imread(f"results/improved/overlays/{tag}.jpg")
        pair = []
        for label, im in (("baseline", a), ("improved", b)):
            t = cv2.resize(im, (640, 360))
            cv2.rectangle(t, (0, 0), (300, 26), (0, 0, 0), -1)
            cv2.putText(t, f"{label} | {title}", (6, 19), 0, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            pair.append(t)
        rows.append(np.hstack(pair))
    cv2.imwrite(os.path.join(FIG, "fig3_cases.jpg"), np.vstack(rows))


def fig_bars() -> None:
    rows = list(csv.DictReader(open("results/summary.csv", encoding="utf-8-sig")))
    data = {}
    for r in rows:
        if r["split"] != "test":
            continue
        data[(r["method"], r["scene"])] = float(r["pooled_hit_rate"])

    W, H = 900, 520
    canvas = np.full((H, W, 3), 255, np.uint8)
    left, right, top, bottom = 90, W - 40, 60, H - 90
    cv2.line(canvas, (left, bottom), (right, bottom), (0, 0, 0), 2)
    cv2.line(canvas, (left, top), (left, bottom), (0, 0, 0), 2)
    for i in range(6):
        y = bottom - int((bottom - top) * i / 5)
        cv2.line(canvas, (left, y), (right, y), (230, 230, 230), 1)
        cv2.putText(canvas, f"{i*20}%", (left - 46, y + 5), 0, 0.45,
                    (80, 80, 80), 1, cv2.LINE_AA)

    groups = ["straight", "curve", "all"]
    bw = 90
    for gi, scene in enumerate(groups):
        x0 = left + 60 + gi * 250
        for mi, method in enumerate(("baseline", "improved")):
            v = data.get((method, scene), 0.0)
            h = int((bottom - top) * v)
            x = x0 + mi * (bw + 30)
            col = (180, 120, 40) if method == "baseline" else (40, 120, 200)
            cv2.rectangle(canvas, (x, bottom - h), (x + bw, bottom), col, -1)
            cv2.putText(canvas, f"{v*100:.1f}%", (x + 8, bottom - h - 8),
                        0, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas, method, (x + 6, bottom + 22), 0, 0.45,
                        (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, scene, (x0 + 40, bottom + 52), 0, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Point hit rate @20px, test split (6 clips, 304 points)",
                (left, 35), 0, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(FIG, "fig2_hit_rate.png"), canvas)


if __name__ == "__main__":
    fig_stage1()
    fig_cases()
    fig_bars()
    print("figures ->", FIG, os.listdir(FIG))
