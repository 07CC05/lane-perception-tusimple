# -*- coding: utf-8 -*-
"""Core utilities for the TuSimple lane-detection experiment.

Everything works in image coordinates: x = f(y). No perspective transform is
used, so the numbers in the report stay comparable between methods.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- labels ----

IMG_W, IMG_H = 1280, 720


def load_labels(label_files: Sequence[str]) -> Dict[str, dict]:
    """Return {raw_file: sample} for every TuSimple JSON line."""
    out: Dict[str, dict] = {}
    for path in label_files:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                out[sample["raw_file"]] = sample
    return out


def ego_lanes(sample: dict) -> Tuple[List[Tuple[float, float]],
                                     List[Tuple[float, float]], int]:
    """Pick the two lanes that bracket the image centre.

    TuSimple stores up to 4 lanes per frame in left-to-right order.  The ego
    lanes are the pair around x = W/2; -2 means "not annotated at this y".
    Returns (left_gt, right_gt, n_gt_lanes) where each lane is [(x, y), ...].
    """
    ys = np.asarray(sample["h_samples"], dtype=float)
    lanes = []
    for lane in sample["lanes"]:
        xs = np.asarray(lane, dtype=float)
        valid = xs > -2
        if valid.sum() < 5:
            continue
        lanes.append((xs, ys, valid))

    if len(lanes) < 2:
        return [], [], len(lanes)

    # bottom-most y where a lane has a valid point -> use it to order lanes
    bottom_y = ys.max()
    means = []
    for xs, yy, valid in lanes:
        idx = np.where(valid)[0]
        # x at the largest valid y of this lane
        x_bottom = xs[idx[-1]]
        means.append((x_bottom, xs, yy, valid))
    means.sort(key=lambda t: t[0])

    left = None
    right = None
    for i in range(len(means) - 1):
        xl = means[i][0]
        xr = means[i + 1][0]
        if xl <= IMG_W / 2 <= xr:
            left, right = means[i], means[i + 1]
            break
    if left is None:  # centre outside all lanes: take the two closest
        means.sort(key=lambda t: abs(t[0] - IMG_W / 2))
        pair = sorted(means[:2], key=lambda t: t[0])
        left, right = pair[0], pair[1]

    def to_points(item):
        _, xs, yy, valid = item
        return [(float(x), float(y)) for x, y in zip(xs[valid], yy[valid])]

    return to_points(left), to_points(right), len(lanes)


def eval_ys(sample: dict, y_min: int = int(0.60 * IMG_H)) -> np.ndarray:
    """Annotated y values inside the ROI. Above the ROI the model has no
    evidence, so extrapolated points are excluded from the metrics."""
    ys = np.asarray(sample["h_samples"], dtype=float)
    return ys[ys >= y_min]


def curvature_score(left_gt: List[Tuple[float, float]],
                    right_gt: List[Tuple[float, float]]) -> float:
    """Scale-free curvature proxy: |a| * span^2 of a quadratic x(y) fit."""
    best = 0.0
    for pts in (left_gt, right_gt):
        if len(pts) < 8:
            continue
        arr = np.asarray(pts, dtype=float)
        a, _, _ = np.polyfit(arr[:, 1], arr[:, 0], 2)
        span = arr[:, 1].max() - arr[:, 1].min()
        best = max(best, abs(a) * span * span)
    return best


# ----------------------------------------------------------- preprocessing --

def roi_vertices(w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    """Trapezoid covering the ego lane and its two neighbours."""
    return np.array(
        [[int(0.02 * w), h],
         [int(0.33 * w), int(0.60 * h)],
         [int(0.67 * w), int(0.60 * h)],
         [int(0.98 * w), h]], dtype=np.int32)


def preprocess(img: np.ndarray,
               canny_low: int = 50,
               canny_high: int = 150,
               blur_ksize: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gray -> gaussian -> canny -> trapezoid mask. Returns (gray, edges, roi)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    roi = np.zeros_like(edges)
    cv2.fillPoly(roi, [roi_vertices(img.shape[1], img.shape[0])], 255)
    edges_roi = cv2.bitwise_and(edges, roi)
    return gray, edges, edges_roi


# --------------------------------------------------------------- baseline ---

@dataclass
class LineFit:
    a: float          # x = a*y + b
    b: float
    n_segments: int
    total_len: float
    segments: List[Tuple[int, int, int, int]]

    def x_at(self, y: np.ndarray | float):
        return self.a * np.asarray(y, dtype=float) + self.b


def hough_segments(edges_roi: np.ndarray,
                   threshold: int = 30,
                   min_len: int = 25,
                   max_gap: int = 100) -> List[Tuple[int, int, int, int]]:
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold,
                            minLineLength=min_len, maxLineGap=max_gap)
    if lines is None:
        return []
    arr = np.asarray(lines).reshape(-1, 4)   # OpenCV 4: (N,1,4); 5.x: (N,4)
    return [tuple(int(v) for v in row) for row in arr]


def split_segments(segments: List[Tuple[int, int, int, int]],
                   w: int = IMG_W, h: int = IMG_H
                   ) -> Tuple[List[Tuple[int, int, int, int]],
                              List[Tuple[int, int, int, int]]]:
    """Split by the sign of dx/dy and by the segment midpoint."""
    left, right = [], []
    for x1, y1, x2, y2 in segments:
        if y1 == y2:
            continue
        dxdy = (x2 - x1) / (y2 - y1)
        if abs(dxdy) > 1.6:          # nearly horizontal: kerb / shadow edge
            continue
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        if mid_y < 0.55 * h:         # ignore far-field noise near the horizon
            continue
        if dxdy < -0.08 and mid_x < 0.62 * w:
            left.append((x1, y1, x2, y2))
        elif dxdy > 0.08 and mid_x > 0.38 * w:
            right.append((x1, y1, x2, y2))
    return left, right


def select_ego_cluster(segments: List[Tuple[int, int, int, int]],
                       side: str, w: int = IMG_W, h: int = IMG_H,
                       gap: float = 55.0) -> List[Tuple[int, int, int, int]]:
    """Keep the lane cluster closest to the image centre.

    The trapezoid is deliberately wide (wide-angle camera), so each side may
    contain 2-3 parallel lane boundaries.  Cluster the segments by their
    extrapolated x at the bottom row and keep the cluster nearest to W/2.
    """
    items = []
    for x1, y1, x2, y2 in segments:
        if y2 == y1:
            continue
        a = (x2 - x1) / (y2 - y1)
        b = x1 - a * y1
        x_bottom = a * h + b
        if side == "left" and not (0.02 * w <= x_bottom <= 0.56 * w):
            continue
        if side == "right" and not (0.44 * w <= x_bottom <= 0.98 * w):
            continue
        items.append((x_bottom, (x1, y1, x2, y2)))
    if not items:
        return []
    items.sort(key=lambda t: t[0])
    clusters = [[items[0]]]
    for item in items[1:]:
        if item[0] - clusters[-1][-1][0] <= gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    best = min(clusters, key=lambda c: abs(
        float(np.mean([i[0] for i in c])) - w / 2))
    return [i[1] for i in best]


def fit_line(segments: List[Tuple[int, int, int, int]],
             min_segments: int = 2,
             min_length: float = 60.0) -> Optional[LineFit]:
    """Least-squares x = a*y + b on segment endpoints, length-weighted."""
    if len(segments) < min_segments:
        return None
    ys, xs, weights = [], [], []
    for x1, y1, x2, y2 in segments:
        length = math.hypot(x2 - x1, y2 - y1)
        for x, y in ((x1, y1), (x2, y2)):
            xs.append(x)
            ys.append(y)
            weights.append(length)
    if sum(weights) / 2 < min_length:
        return None
    a, b = np.polyfit(np.asarray(ys, float), np.asarray(xs, float), 1,
                      w=np.asarray(weights, float))
    return LineFit(float(a), float(b), len(segments), sum(weights) / 2, segments)


def predict(fit: Optional[LineFit], ys: np.ndarray,
            w: int = IMG_W) -> Optional[np.ndarray]:
    if fit is None:
        return None
    xs = fit.x_at(ys)
    if not np.all(np.isfinite(xs)):
        return None
    if xs.min() < -0.15 * w or xs.max() > 1.15 * w:
        return None
    return xs


# --------------------------------------------------------------- improved ---

@dataclass
class QuadFit:
    coeffs: np.ndarray   # x = a*y^2 + b*y + c
    n_points: int
    windows: List[Tuple[int, int, int, int]]   # (x0,y0,x1,y1) for drawing
    points: np.ndarray   # (N,2) x,y

    def x_at(self, y):
        a, b, c = self.coeffs
        y = np.asarray(y, dtype=float)
        return a * y * y + b * y + c


# Bird's-eye view for the sliding-window fit.  TuSimple uses one fixed
# camera, so a single homography is enough; the trapezoid brackets both ego
# lanes between y = 455 and the bottom row.
BEV_W, BEV_H = 480, 480
BEV_SRC = np.float32([[60, 720], [430, 455], [1060, 455], [1240, 720]])
BEV_DST = np.float32([[0, BEV_H], [0, 0], [BEV_W, 0], [BEV_W, BEV_H]])
BEV_M = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)
BEV_MINV = np.linalg.inv(BEV_M)


@dataclass
class BevFit:
    coeffs: np.ndarray          # x_bev = a*y^2 + b*y + c
    n_points: int
    windows_img: List[np.ndarray]
    points_img: np.ndarray

    def x_at(self, ys):
        ys = np.asarray(ys, dtype=float)
        y_bev = np.linspace(BEV_H + 30, -30, 260)
        a, b, c = self.coeffs
        x_bev = a * y_bev ** 2 + b * y_bev + c
        pts = np.stack([x_bev, y_bev], axis=1).astype(np.float32).reshape(-1, 1, 2)
        img = cv2.perspectiveTransform(pts, BEV_MINV).reshape(-1, 2)
        order = np.argsort(img[:, 1])
        yy, xx = img[order, 1], img[order, 0]
        return np.interp(ys, yy, xx)


def bev_warp(edges_roi: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(edges_roi, BEV_M, (BEV_W, BEV_H),
                               flags=cv2.INTER_NEAREST)


def sliding_window_fit(edges_roi: np.ndarray,
                       side: str,
                       guide: Optional[Tuple[float, float]] = None,
                       n_windows: int = 9,
                       half_width: int = 45,
                       min_points: int = 120,
                       min_pix: int = 8) -> Optional[BevFit]:
    """Sliding-window search in bird's-eye view + quadratic fit.

    In BEV the lane boundaries are close to vertical and parallel, so a fixed
    window width is safe for both straight and curved roads.  ``guide`` (the
    baseline straight line) is only used to pick the seed column.
    """
    warped = bev_warp(edges_roi)
    binary = (warped > 0).astype(np.uint8)
    nz_y, nz_x = binary.nonzero()
    if nz_x.size < min_points:
        return None

    y_bottom = int(0.98 * BEV_H)
    y_top = int(0.02 * BEV_H)
    band = max(1, (y_bottom - y_top) // n_windows)

    base = nz_x[(nz_y >= y_bottom - band)]
    if base.size < 8:
        return None
    hist = np.bincount(base, minlength=BEV_W).astype(float)
    smooth = np.convolve(hist, np.ones(7) / 7.0, mode="same")
    half_sel = (0, BEV_W // 2) if side == "left" else (BEV_W // 2, BEV_W)
    peaks = [x for x in range(half_sel[0] + 1, half_sel[1] - 1)
             if smooth[x] >= smooth[x - 1] and smooth[x] >= smooth[x + 1]
             and smooth[x] >= 3]
    if peaks:
        center = float(min(peaks, key=lambda x: abs(x - BEV_W / 2)))
    else:
        seg = smooth[half_sel[0]:half_sel[1]]
        if seg.size == 0 or seg.max() < 3:
            return None
        center = float(np.argmax(seg) + half_sel[0])

    windows_bev, pts, accepted = [], [], []
    for i in range(n_windows):
        y_hi = y_bottom - i * band
        y_lo = y_hi - band
        y_mid = 0.5 * (y_lo + y_hi)
        if len(accepted) >= 3:
            yy = np.array([p[0] for p in accepted], float)
            xx = np.array([p[1] for p in accepted], float)
            a_i, b_i, c_i = np.polyfit(yy, xx, 2)
            cx = float(a_i * y_mid ** 2 + b_i * y_mid + c_i)
        elif len(accepted) == 2:
            yy = np.array([p[0] for p in accepted], float)
            xx = np.array([p[1] for p in accepted], float)
            a_i, b_i = np.polyfit(yy, xx, 1)
            cx = float(a_i * y_mid + b_i)
        elif accepted:
            cx = accepted[-1][1]
        else:
            cx = center
        x_lo = int(max(0, cx - half_width))
        x_hi = int(min(BEV_W, cx + half_width))
        windows_bev.append((x_lo, y_lo, x_hi, y_hi))
        sel = (nz_y >= y_lo) & (nz_y < y_hi) & (nz_x >= x_lo) & (nz_x < x_hi)
        if sel.sum() < min_pix:
            continue
        med = float(np.median(nz_x[sel]))
        pts.append(np.stack([nz_x[sel], nz_y[sel]], axis=1))
        accepted.append((y_mid, med))

    if not pts:
        return None
    points_bev = np.concatenate(pts, axis=0).astype(float)
    if len(points_bev) < min_points:
        return None

    coeffs = np.polyfit(points_bev[:, 1], points_bev[:, 0], 2)
    for _ in range(2):
        a, b, c = coeffs
        res = np.abs(a * points_bev[:, 1] ** 2 + b * points_bev[:, 1] + c
                     - points_bev[:, 0])
        keep = res <= max(12.0, float(np.percentile(res, 90)))
        if keep.sum() < 0.6 * len(points_bev) or keep.all():
            break
        points_bev = points_bev[keep]
        coeffs = np.polyfit(points_bev[:, 1], points_bev[:, 0], 2)

    # Second pass: a lane marking is a band, not a single edge.  Re-centre
    # each band on the midpoint between the 5th and 95th percentile of the
    # edge pixels around the current curve, then refit.
    a, b, c = coeffs
    centres = []
    for i in range(n_windows):
        y_hi = y_bottom - i * band
        y_lo = y_hi - band
        y_mid = 0.5 * (y_lo + y_hi)
        cx = a * y_mid ** 2 + b * y_mid + c
        sel = (nz_y >= y_lo) & (nz_y < y_hi) & (np.abs(nz_x - cx) < 38)
        if sel.sum() >= 10:
            lo, hi = np.percentile(nz_x[sel], [5, 95])
            centres.append((y_mid, float((lo + hi) / 2), float(sel.sum())))
    if len(centres) >= 3:
        yy = np.array([t[0] for t in centres], float)
        xx = np.array([t[1] for t in centres], float)
        ww = np.array([t[2] for t in centres], float)
        coeffs = np.polyfit(yy, xx, 2, w=ww)

    # back-project sampled points and window corners for drawing
    sample = np.linspace(BEV_H, 0, 120)
    a, b, c = coeffs
    pts_bev = np.stack([a * sample ** 2 + b * sample + c, sample], axis=1)
    pts_img = cv2.perspectiveTransform(
        pts_bev.astype(np.float32).reshape(-1, 1, 2), BEV_MINV).reshape(-1, 2)

    windows_img = []
    for x_lo, y_lo, x_hi, y_hi in windows_bev:
        corners = np.float32([[x_lo, y_lo], [x_hi, y_lo],
                              [x_hi, y_hi], [x_lo, y_hi]]).reshape(-1, 1, 2)
        windows_img.append(cv2.perspectiveTransform(corners, BEV_MINV).reshape(-1, 2))

    return BevFit(coeffs, len(points_bev), windows_img, pts_img)


def predict_quad(fit, ys: np.ndarray,
                 w: int = IMG_W) -> Optional[np.ndarray]:
    if fit is None:
        return None
    xs = fit.x_at(ys)
    if not np.all(np.isfinite(xs)):
        return None
    if xs.min() < -0.15 * w or xs.max() > 1.15 * w:
        return None
    return xs


# ---------------------------------------------------------------- metrics ---

def point_metrics(pred: Optional[np.ndarray],
                  gt: List[Tuple[float, float]],
                  ys: np.ndarray,
                  tol: float = 20.0) -> dict:
    """Hit rate over the annotated y grid; a missed side counts all its points."""
    gt_map = {float(y): float(x) for x, y in gt}
    valid = [float(y) for y in ys if float(y) in gt_map]
    if not valid:
        return {"n": 0, "hits": 0, "hit_rate": float("nan"),
                "mae": float("nan"), "missed": pred is None}
    if pred is None:
        return {"n": len(valid), "hits": 0, "hit_rate": 0.0,
                "mae": float("nan"), "missed": True}
    errs = []
    hits = 0
    for y, px in zip(ys, pred):
        gy = float(y)
        if gy not in gt_map:
            continue
        e = abs(float(px) - gt_map[gy])
        errs.append(e)
        if e <= tol:
            hits += 1
    return {"n": len(valid), "hits": hits,
            "hit_rate": hits / len(valid),
            "mae": float(np.mean(errs)) if errs else float("nan"),
            "missed": False}


def frame_metrics(pred_l, pred_r, gt_l, gt_r, ys, tol=20.0) -> dict:
    ml = point_metrics(pred_l, gt_l, ys, tol)
    mr = point_metrics(pred_r, gt_r, ys, tol)
    n = ml["n"] + mr["n"]
    hits = ml["hits"] + mr["hits"]
    errs = [v for v in (ml["mae"], mr["mae"]) if not math.isnan(v)]
    return {
        "n_points": n,
        "hits": hits,
        "hit_rate": hits / n if n else float("nan"),
        "left_hit_rate": ml["hit_rate"],
        "right_hit_rate": mr["hit_rate"],
        "left_missed": ml["missed"],
        "right_missed": mr["missed"],
        "mae": float(np.mean(errs)) if errs else float("nan"),
    }


# ---------------------------------------------------------------- drawing ---

COL_L = (255, 120, 0)     # BGR blue-ish for left
COL_R = (0, 0, 255)       # red for right
COL_GT = (0, 255, 0)


def draw_baseline(img, edges_roi, left: Optional[LineFit], right: Optional[LineFit],
                  gt_l, gt_r, ys, title="") -> np.ndarray:
    out = img.copy()
    poly = roi_vertices(img.shape[1], img.shape[0])
    cv2.polylines(out, [poly], True, (0, 200, 255), 2)
    for fit, col in ((left, COL_L), (right, COL_R)):
        if fit is None:
            continue
        for x1, y1, x2, y2 in fit.segments:
            cv2.line(out, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
    for fit, col in ((left, COL_L), (right, COL_R)):
        xs = predict(fit, ys)
        if xs is None:
            continue
        pts = [(int(x), int(y)) for x, y in zip(xs, ys)
               if 0 <= x < img.shape[1]]
        for p in pts:
            cv2.circle(out, p, 2, col, -1, cv2.LINE_AA)
    for gt, col in ((gt_l, COL_GT), (gt_r, COL_GT)):
        for x, y in gt:
            cv2.drawMarker(out, (int(x), int(y)), col,
                           cv2.MARKER_TILTED_CROSS, 7, 1)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_improved(img, left, right,
                  gt_l, gt_r, ys, title="") -> np.ndarray:
    out = img.copy()
    cv2.polylines(out, [roi_vertices(img.shape[1], img.shape[0])], True,
                  (0, 200, 255), 2)
    for fit, col in ((left, COL_L), (right, COL_R)):
        if fit is None:
            continue
        for poly in fit.windows_img:
            cv2.polylines(out, [poly.astype(np.int32)], True, col, 1)
        for x, y in fit.points_img[::2]:
            if 0 <= int(y) < out.shape[0] and 0 <= int(x) < out.shape[1]:
                out[int(y), int(x)] = col
        xs = predict_quad(fit, ys)
        if xs is None:
            continue
        pts2 = [(int(x), int(y)) for x, y in zip(xs, ys) if 0 <= x < img.shape[1]]
        for p in pts2:
            cv2.circle(out, p, 2, col, -1, cv2.LINE_AA)
    for gt, col in ((gt_l, COL_GT), (gt_r, COL_GT)):
        for x, y in gt:
            cv2.drawMarker(out, (int(x), int(y)), col,
                           cv2.MARKER_TILTED_CROSS, 7, 1)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
