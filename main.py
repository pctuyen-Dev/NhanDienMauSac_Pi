#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HE THONG NHAN DIEN VA PHAN LOAI MAU SAC & HINH HOC (OpenCV + NumPy) -- phien ban v7
10 mau (du 8 nhom mau de bai + 2 mau gan giong):
DO (red), CAM (orange), VANG (yellow), XANH LA MA (lime #85B94A),
XANH LA (green), XANH MINT (mint), XANH DUONG (blue), TIM (purple),
TRANG (white), DEN (black)

Chay:
    python python.py                      # tu dong mo rubik.jpg / nap_chai.jpg / lego.jpg / webcam
    python python.py --source rubik.jpg
    python python.py --source nap_chai.jpg
    python python.py --source lego.jpg
    python python.py --source 0           # webcam so 0

Phim tat:
    o       : mo hop thoai chon anh tu may tinh de dua vao
    g       : mo giao dien do hoa truc quan (Tkinter GUI - giaoDien.py)
    n / TAB : chuyen doi nguon anh dau vao (rubik.jpg <-> nap_chai.jpg <-> lego.jpg <-> webcam 0)
    v       : chuyen kieu hien thi:
              - Kieu 0 (Mac dinh): Tach rieng vat the voi nen den, phan mau + vien Contour xanh la + chu trang "color shape" giong anh mau muc 3
              - Kieu 1: Hien thi tren anh goc voi day du Bounding Box + Tam + Dien tich + Nhan Tieng Viet + HUD
    h       : bat/tat bang thong ke HUD tren goc anh
    c       : bat/tat cua so hieu chinh (Calibration, co Trackbar)
    m       : bat/tat cua so mask cua mau dang chon
    r       : reset nguong cua mau dang chon (khi o che do hieu chinh)
    0..4    : mo phong dieu kien anh sang de danh gia:
              0 binh thuong | 1 anh sang manh | 2 anh sang yeu
              3 anh sang vang | 4 bong do len vat the
    b       : chup NEN / xoa nen
    w       : bat/tat can bang trang (gray-world)
    s       : luu anh ket qua vao thu muc output/
    q / ESC : thoat
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# CAU HINH CHUNG
# --------------------------------------------------------------------------- #
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MIN_CONTOUR_AREA = 3200        # Loc bo bong do va hat nhieu nho (vat the that thuong > 3200 px)
MAX_AREA_RATIO = 0.60          # Bo vung mau co sac > 60% khung hinh
MAX_AREA_RATIO_ACHRO = 0.15    # TRANG/DEN: bo vung > 15% khung hinh (thuong la tuong/nen)
BORDER_MARGIN = 4              # TRANG/DEN: vung cham canh khung hinh -> la nen, bo
BG_THRESH = 25                 # Nguong khac biet so voi anh nen (che do phim 'b')
MIN_SIDE = 30                  # Canh ngan nhat cua khung (px)
MIN_EXTENT = 0.30              # Dien tich contour / dien tich bounding box
OVERLAP_IOU = 0.50             # Hai khung khac mau chong len nhau > 50% -> giu 1
BLUR_KERNEL = (11, 11)         # GaussianBlur (kernel le)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

K_CHROMATIC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_ACHRO_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
K_ACHRO_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
K_SMOOTH_CAP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

MAIN_WINDOW = "Image"
CALIB_WINDOW = "Calibration"
MASK_WINDOW = "Mask"

LIGHTING_MODES = ["BINH THUONG", "ANH SANG MANH", "ANH SANG YEU",
                  "ANH SANG VANG", "BONG DO"]

# Mau phan doan phang tuoi sang tren nen den (BGR) giong anh mau muc 3
SEGMENT_FILL_COLORS = {
    "red":    (95, 65, 240),
    "orange": (15, 135, 255),
    "yellow": (20, 230, 255),
    "lime":   (74, 185, 133),
    "green":  (110, 210, 45),
    "mint":   (210, 225, 45),
    "blue":   (240, 145, 0),
    "purple": (210, 65, 165),
    "white":  (235, 235, 235),
    "black":  (50, 50, 50),
}


# --------------------------------------------------------------------------- #
# HO SO MAU
# --------------------------------------------------------------------------- #
@dataclass
class ColorProfile:
    """Mot mau can nhan dien.

    params = [H_min, H_max, S_min, S_max, V_min, V_max]   (thang OpenCV: H 0-180)
    wraps_hue  : mau DO - dai Hue thu 2 = [180 - H_max, 180 - H_min]
    achromatic : mau vo sac (TRANG/DEN) -> hinh thai hoc manh hon
    """
    label: str
    en_label: str
    params: List[int]
    box_color: Tuple[int, int, int]  # BGR
    wraps_hue: bool = False
    achromatic: bool = False
    exclude: List[str] = field(default_factory=list)
    defaults: List[int] = field(init=False)

    def __post_init__(self):
        self.defaults = list(self.params)

    def reset(self):
        self.params = list(self.defaults)

    def hsv_ranges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        h_min, h_max, s_min, s_max, v_min, v_max = self.params
        ranges = [(np.array([h_min, s_min, v_min], np.uint8),
                   np.array([h_max, s_max, v_max], np.uint8))]
        if self.wraps_hue:
            lo2 = max(0, 180 - max(h_max, 10))
            hi2 = min(180, 180 - h_min)
            ranges.append((np.array([lo2, s_min, v_min], np.uint8),
                           np.array([hi2, s_max, v_max], np.uint8)))
        return ranges


def build_profiles() -> List[ColorProfile]:
    return [
        # 1. DO: H 0-5 va 170-180 (hai dau dai Hue)
        ColorProfile("DO", "red", [0, 5, 120, 255, 70, 255],
                     box_color=(0, 0, 255), wraps_hue=True),
        # 2. CAM: H 5-16
        ColorProfile("CAM", "orange", [5, 16, 150, 255, 100, 255],
                     box_color=(0, 140, 255)),
        # 3. VANG: H 17-32
        ColorProfile("VANG", "yellow", [17, 32, 120, 255, 80, 255],
                     box_color=(0, 230, 255)),
        # 4a. XANH LA MA: H 33-63, S >= 100 (loai bo bong do nhat tren ban mau)
        ColorProfile("XANH LA MA", "lime", [33, 63, 100, 255, 90, 255],
                     box_color=(74, 185, 133)),
        # 4b. XANH LA: H 64-76, S >= 115 (phan biet voi XANH LA MA & nen)
        ColorProfile("XANH LA", "green", [64, 76, 115, 255, 65, 255],
                     box_color=(0, 200, 0)),
        # 5a. XANH MINT (giua xanh la va xanh duong): H 77-103, S >= 85, V >= 70 (khu bong do nen)
        ColorProfile("XANH MINT", "mint", [77, 103, 85, 255, 70, 255],
                     box_color=(255, 230, 100)),
        # 5b. XANH DUONG: H 104-128
        ColorProfile("XANH DUONG", "blue", [104, 128, 110, 255, 60, 255],
                     box_color=(255, 100, 0)),
        # 6. TIM: H 129-168, S >= 85 (tranh bong toi tim tren nen sang)
        ColorProfile("TIM", "purple", [129, 168, 85, 255, 60, 255],
                     box_color=(220, 60, 180)),
        # 7. TRANG: S thap (<= 26), V cao (>= 165)
        ColorProfile("TRANG", "white", [0, 180, 0, 26, 165, 255],
                     box_color=(220, 220, 220), achromatic=True),
        # 8. DEN: V thap (<= 48), S thap-trung binh (<= 80, tranh bong do co mau)
        ColorProfile("DEN", "black", [0, 180, 0, 80, 0, 48],
                     box_color=(0, 165, 255), achromatic=True),
    ]


@dataclass
class Detection:
    profile: ColorProfile
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: float
    purity: float        # ty le pixel trong contour thuc su thuoc mau nay
    shape: str = "object"
    contour: Optional[np.ndarray] = None


def classify_shape(cnt: np.ndarray) -> str:
    """Nhan dien hinh dang hinh hoc (triangle, square, rectangle, pentagon, circle)."""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    if peri <= 0:
        return "circle"
    circularity = 4.0 * np.pi * area / (peri * peri)
    (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
    ar = max(rw, rh) / float(max(1.0, min(rw, rh)))
    approx20 = cv2.approxPolyDP(hull, 0.020 * peri, True)
    approx30 = cv2.approxPolyDP(hull, 0.030 * peri, True)
    v = len(approx30)
    if v == 3:
        return "triangle"
    if len(approx20) >= 7 and circularity >= 0.88:
        return "circle"
    if v == 4:
        return "square" if ar <= 1.22 else "rectangle"
    if v == 5:
        return "rectangle" if ar > 1.25 else "pentagon"
    return "square" if ar <= 1.22 else "rectangle"


# --------------------------------------------------------------------------- #
# MO PHONG ANH SANG (phuc vu danh gia do chinh xac - Nhiem vu 5)
# --------------------------------------------------------------------------- #
def simulate_lighting(frame: np.ndarray, mode: int) -> np.ndarray:
    if mode == 1:      # anh sang manh
        return cv2.convertScaleAbs(frame, alpha=1.04, beta=16)
    if mode == 2:      # anh sang yeu
        return cv2.convertScaleAbs(frame, alpha=0.45, beta=0)
    if mode == 3:      # anh sang vang: giam kenh B, tang nhe kenh R
        f = frame.astype(np.float32)
        f[..., 0] *= 0.55
        f[..., 1] *= 0.95
        f[..., 2] *= 1.10
        return np.clip(f, 0, 255).astype(np.uint8)
    if mode == 4:      # bong do: nua trai toi, chuyen mem sang nua phai
        w = frame.shape[1]
        gain = np.interp(np.arange(w), [0, w * 0.35, w * 0.65], [0.35, 0.35, 1.0])
        return np.clip(frame.astype(np.float32) * gain[None, :, None],
                       0, 255).astype(np.uint8)
    return frame


# --------------------------------------------------------------------------- #
# XU LY ANH & BU ANH SANG (Nhiem vu 2)
# --------------------------------------------------------------------------- #
def gray_world_balance(frame: np.ndarray) -> np.ndarray:
    """Can bang trang gray-world: dua trung binh 3 kenh B,G,R ve cung muc."""
    means = frame.reshape(-1, 3).mean(axis=0) + 1e-6
    gains = means.mean() / means
    return np.clip(frame.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def compensate_lighting(frame: np.ndarray) -> np.ndarray:
    """Tu dong bu anh sang yeu, anh sang manh, anh sang vang va bong do cuc bo."""
    f = frame.astype(np.float32)
    v = f.max(axis=2)

    # 1. Bu bong do cuc bo theo chieu ngang (uneven illumination / shadow)
    is_bright_bg = np.percentile(np.median(v, axis=0), 90) > 110
    q_col = 85 if is_bright_bg else 50
    col_ref = np.percentile(v, q_col, axis=0)
    col_smooth = cv2.GaussianBlur(col_ref.reshape(1, -1), (1, 101), 0).ravel()
    ref_v = np.percentile(col_smooth, 90)
    if ref_v > 30 and col_smooth.min() < 0.65 * ref_v:
        gain_x = np.clip(ref_v / np.maximum(col_smooth, 25.0 if is_bright_bg else 15.0), 1.0, 3.2)
        f = np.clip(f * gain_x[None, :, None], 0, 255)
        v = f.max(axis=2)

    # 2. Bu anh sang yeu (underexposed)
    v_p95 = np.percentile(v, 95)
    if 20 < v_p95 < 155:
        f = np.clip(f * (225.0 / v_p95), 0, 255)
        v = f.max(axis=2)

    # 3. Bu anh sang manh / nang nen den (overexposed pedestal)
    p0 = np.percentile(f.min(axis=2), 0.5)
    if p0 >= 12 and np.median(v) > 215:
        clipped = f >= 254.0
        f_un = np.clip((f - p0) / 1.04, 0, 255)
        sat_span = f.max(axis=2, keepdims=True) - f.min(axis=2, keepdims=True)
        f = np.where(clipped & (sat_span > 50), 255.0, f_un)

    # 4. Bu anh sang vang / lech mau (White-Patch p95 hoac nen trung tinh)
    p95 = np.percentile(f.reshape(-1, 3), 95, axis=0) + 1e-6
    med = np.median(f.reshape(-1, 3), axis=0) + 1e-6
    if (p95[0] < 0.75 * p95[1] and p95[2] > 0.95 * p95[1]) or \
       (not is_bright_bg and med[0] < 0.72 * med[1] and med[2] > 1.05 * med[1]):
        b_gain = min(2.2, p95[1] / p95[0]) if p95[0] < 0.75 * p95[1] else (1.0 / 0.55)
        gains = np.array([b_gain, 1.0 / 0.95, 1.0 / 1.10], dtype=np.float32)
        f = np.clip(f * gains, 0, 255)

    return f.astype(np.uint8)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Bu anh sang, lam mo Gaussian roi chuyen BGR -> HSV."""
    balanced = compensate_lighting(frame)
    blurred = cv2.GaussianBlur(balanced, BLUR_KERNEL, 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


def raw_mask(hsv: np.ndarray, profile: ColorProfile) -> np.ndarray:
    """Phan doan theo nguong HSV (chua loc hinh thai)."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in profile.hsv_ranges():
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def clean_mask(mask: np.ndarray, profile: ColorProfile) -> np.ndarray:
    """MORPH_OPEN bo nhieu hat, MORPH_CLOSE lap lo hong."""
    if profile.achromatic:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_ACHRO_OPEN)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_ACHRO_CLOSE, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_CHROMATIC)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_CHROMATIC, iterations=2)


def find_objects(clean: np.ndarray, raw: np.ndarray,
                 profile: ColorProfile) -> List[Detection]:
    """Contour -> dien tich, tam, bbox, hinh dang, do thuan khiet cua tung vat the."""
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = clean.shape[:2]
    frame_area = fh * fw
    max_area = (MAX_AREA_RATIO_ACHRO if profile.achromatic else MAX_AREA_RATIO) * frame_area
    found: List[Detection] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= MIN_CONTOUR_AREA or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if min(w, h) < MIN_SIDE or area / float(w * h) < MIN_EXTENT:
            continue
        if profile.achromatic:
            touches = (x <= BORDER_MARGIN) + (y <= BORDER_MARGIN) + \
                      (x + w >= fw - BORDER_MARGIN) + (y + h >= fh - BORDER_MARGIN)
            if touches >= 1:
                continue

        m = cv2.moments(cnt)
        if m["m00"] > 0:
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        canvas = np.zeros((h, w), np.uint8)
        cv2.drawContours(canvas, [cnt], -1, 255, -1, offset=(-x, -y))
        inside = max(1, cv2.countNonZero(canvas))
        purity = cv2.countNonZero(cv2.bitwise_and(canvas, raw[y:y + h, x:x + w])) / inside

        shape_name = classify_shape(cnt)
        found.append(Detection(profile, (x, y, w, h), (cx, cy), area, purity, shape_name, cnt))

    kept = []
    for i, d in enumerate(found):
        x, y, w, h = d.bbox
        nested = any(
            i != j and d.area < o.area and x >= o.bbox[0] and y >= o.bbox[1]
            and x + w <= o.bbox[0] + o.bbox[2] and y + h <= o.bbox[1] + o.bbox[3]
            for j, o in enumerate(found))
        if not nested:
            kept.append(d)
    return kept


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def resolve_overlaps(dets: List[Detection]) -> List[Detection]:
    """Hai mau cung bat 1 vat the (mau gan giong): giu mau thuan khiet hon."""
    dets = sorted(dets, key=lambda d: (d.purity, d.area), reverse=True)
    kept: List[Detection] = []
    for d in dets:
        if all(iou(d.bbox, k.bbox) < OVERLAP_IOU for k in kept):
            kept.append(d)
    return kept


def foreground_mask(blurred: np.ndarray, bg_blur: np.ndarray) -> np.ndarray:
    """Vung khac anh nen da chup (vat moi xuat hien)."""
    diff = cv2.absdiff(blurred, bg_blur).max(axis=2)
    fg = (diff > BG_THRESH).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, K_CHROMATIC)
    return cv2.dilate(fg, K_ACHRO_OPEN, iterations=1)


def detect(frame: np.ndarray, profiles: List[ColorProfile], selected: int = 0,
           bg_blur: Optional[np.ndarray] = None):
    """Tra ve (danh sach Detection, mask mau dang chon, hsv)."""
    balanced = compensate_lighting(frame)
    blurred = cv2.GaussianBlur(balanced, BLUR_KERNEL, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    raws = {p.label: raw_mask(hsv, p) for p in profiles}
    if bg_blur is not None:
        raw_blur = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
        fg = foreground_mask(raw_blur, bg_blur)
        for k in raws:
            raws[k] = cv2.bitwise_and(raws[k], fg)
    for p in profiles:
        for other in p.exclude:
            if other in raws:
                raws[p.label] = cv2.bitwise_and(raws[p.label], cv2.bitwise_not(raws[other]))

    all_dets: List[Detection] = []
    selected_mask = None
    for idx, p in enumerate(profiles):
        clean = clean_mask(raws[p.label], p)
        if idx == selected:
            selected_mask = clean
        all_dets.extend(find_objects(clean, raws[p.label], p))
    return resolve_overlaps(all_dets), selected_mask, hsv


# --------------------------------------------------------------------------- #
# VE KET QUA (Kieu 0: Tach rieng vat the voi nen den + phan mau giong anh mau muc 3)
# --------------------------------------------------------------------------- #
def make_black_board_bg(h: int, w: int) -> np.ndarray:
    """Tao nen den van doc nhe giong anh mau muc 3."""
    rng = np.random.default_rng(7)
    streaks = cv2.GaussianBlur(rng.normal(0, 2.5, size=(h, w)).astype(np.float32), (3, 35), 0)
    bg_val = np.clip(22.0 + streaks, 14, 30).astype(np.uint8)
    return cv2.merge([bg_val, bg_val, bg_val])


def draw_detection_sample_label(frame, det: Detection):
    """Ghi chu trang '<color>' tai tam vat the bang Tieng Viet."""
    cx, cy = det.center
    text = f"{det.profile.label}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = int(np.clip(cx - tw // 2 + 6, 6, frame.shape[1] - tw - 6))
    ty = int(np.clip(cy + th // 2, th + 6, frame.shape[0] - 6))
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (2, 1)):
        cv2.putText(frame, text, (tx + dx, ty + dy), font, scale, (15, 15, 15), thick, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def smooth_contour_for_render(cnt: np.ndarray, shape_name: str, h: int, w: int) -> np.ndarray:
    """Lam muot bien vat the (cat gai bong do nho) truoc khi ve phan mau tren nen den."""
    if shape_name == "circle":
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
    return cv2.convexHull(cnt)


def render_segmented_on_black(frame: np.ndarray, dets: List[Detection]):
    """Tach rieng tung vat the khoi nen, to mau phan doan tren nen den + vien xanh la."""
    h, w = frame.shape[:2]
    canvas = make_black_board_bg(h, w)

    # Luot 1: Ve vung phan mau cua tung vat the va vien Contour xanh la (0, 255, 0)
    for d in dets:
        if d.contour is None:
            continue
        poly = smooth_contour_for_render(d.contour, d.shape, h, w)
        c_bgr = SEGMENT_FILL_COLORS.get(d.profile.en_label, d.profile.box_color)
        cv2.drawContours(canvas, [poly], -1, c_bgr, -1, cv2.LINE_AA)
        cv2.drawContours(canvas, [poly], -1, (0, 255, 0), 3, cv2.LINE_AA)

    # Luot 2: Ghi chu trang '<color> <shape>' len tren cung
    for d in dets:
        draw_detection_sample_label(canvas, d)

    frame[:] = canvas


def draw_detection_box(frame, det: Detection):
    """Kieu 1: Ve duong bao contour va bounding box cua vat the tren anh goc."""
    x, y, w, h = det.bbox
    color = det.profile.box_color
    if det.contour is not None:
        cv2.drawContours(frame, [det.contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_detection_label(frame, det: Detection):
    """Kieu 1: Ve tam va nhan mau + toa do + dien tich tren anh goc."""
    x, y, w, h = det.bbox
    cx, cy = det.center
    color = det.profile.box_color
    cv2.circle(frame, (cx, cy), 6, (0, 0, 0), -1)
    cv2.circle(frame, (cx, cy), 4, color, -1)

    purity_str = f"{det.purity * 100:.0f}%"
    lines = [f"{det.profile.label}",
             f"{purity_str}"]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    box_w = max(s[0] for s in sizes) + 8
    line_h = sizes[0][1] + 6
    box_h = line_h * len(lines) + 4
    by = y - box_h if y - box_h > 0 else y + h
    cv2.rectangle(frame, (x, by), (x + box_w, by + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, by), (x + box_w, by + box_h), color, 1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x + 4, by + (i + 1) * line_h),
                    font, scale, color, thick + (i == 0), cv2.LINE_AA)


def render_detections(frame, dets: List[Detection], style: int = 0):
    """Ve toan bo vat the theo kieu hien thi (0: tach nen den phan mau, 1: tren anh goc)."""
    if style == 0:
        render_segmented_on_black(frame, dets)
    else:
        for d in dets:
            draw_detection_box(frame, d)
        for d in dets:
            draw_detection_label(frame, d)


def draw_hud(frame, dets: List[Detection], profiles, fps: float, mode: int,
             wb: bool = False, bg: bool = False, src_name: str = ""):
    """Bang thong ke: so vat the theo tung mau + che do anh sang + FPS."""
    counts = {p.label: 0 for p in profiles}
    for d in dets:
        counts[d.profile.label] += 1

    header = f"FPS {fps:.0f} | {LIGHTING_MODES[mode]}"
    if src_name:
        header += f" | {src_name}"
    if wb:
        header += " | WB"
    if bg:
        header += " | BG"
    rows = [header] + [f"{p.label}: {counts[p.label]}" for p in profiles]
    pw, ph = 260, 18 * len(rows) + 8
    roi = frame[0:ph, 0:pw]
    frame[0:ph, 0:pw] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    for i, text in enumerate(rows):
        color = (0, 255, 255) if i == 0 else profiles[i - 1].box_color
        cv2.putText(frame, text, (8, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, color, 1, cv2.LINE_AA)

    hint = "o:chon anh n:doi anh/cam v:kieu ve h:HUD c:calib m:mask 0-4:anh sang s:luu q:thoat"
    cv2.putText(frame, hint, (8, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hint, (8, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# CALIBRATION (TRACKBAR)
# --------------------------------------------------------------------------- #
class Calibrator:
    """Cua so phu co thanh truot de tinh chinh nguong HSV cua tung mau."""

    TB_NAMES = ["H_min", "H_max", "S_min", "S_max", "V_min", "V_max"]
    TB_LIMITS = [180, 180, 255, 255, 255, 255]

    def __init__(self, profiles: List[ColorProfile]):
        self.profiles = profiles
        self.active = False
        self.selected = 0
        self.selector = "Color (0..9)"

    @staticmethod
    def _noop(_):
        pass

    def open(self):
        cv2.namedWindow(CALIB_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CALIB_WINDOW, 480, 340)
        cv2.createTrackbar(self.selector, CALIB_WINDOW, self.selected,
                           len(self.profiles) - 1, self._noop)
        for name, limit in zip(self.TB_NAMES, self.TB_LIMITS):
            cv2.createTrackbar(name, CALIB_WINDOW, 0, limit, self._noop)
        self._push_to_trackbars()
        self.active = True

    def close(self):
        cv2.destroyWindow(CALIB_WINDOW)
        self.active = False

    def toggle(self):
        self.close() if self.active else self.open()

    def _push_to_trackbars(self):
        for name, value in zip(self.TB_NAMES, self.profiles[self.selected].params):
            cv2.setTrackbarPos(name, CALIB_WINDOW, int(value))

    def update(self):
        if not self.active:
            return
        sel = cv2.getTrackbarPos(self.selector, CALIB_WINDOW)
        if sel != self.selected:
            self.selected = sel
            self._push_to_trackbars()
            return
        self.profiles[self.selected].params = [
            cv2.getTrackbarPos(n, CALIB_WINDOW) for n in self.TB_NAMES]

    def sample_from_click(self, hsv: np.ndarray, x: int, y: int, r: int = 5):
        """Lay HSV trung vi quanh diem click, dat nguong cho mau dang chon."""
        hh, ww = hsv.shape[:2]
        patch = hsv[max(0, y - r):min(hh, y + r + 1),
                    max(0, x - r):min(ww, x + r + 1)].reshape(-1, 3)
        if patch.size == 0:
            return
        h, s, v = [int(t) for t in np.median(patch, axis=0)]
        print(f"[CLICK] ({x},{y}) HSV = ({h},{s},{v})")
        if not self.active:
            return

        p = self.profiles[self.selected]
        clip = lambda val, lo, hi: int(max(lo, min(hi, val)))
        if p.achromatic:
            if v > 127:
                vlo, vhi = clip(v - 35, 0, 255), 255
            else:
                vlo, vhi = 0, clip(v + 25, 0, 255)
            p.params = [0, 180, 0, clip(s + 35, 0, 255), vlo, vhi]
        elif p.wraps_hue:
            hc = h if h <= 90 else 180 - h
            p.params = [clip(hc - 10, 0, 90), clip(hc + 10, 0, 90),
                        clip(s - 50, 0, 255), 255, clip(v - 50, 0, 255), 255]
        else:
            p.params = [clip(h - 10, 0, 180), clip(h + 10, 0, 180),
                        clip(s - 50, 0, 255), clip(s + 50, 0, 255),
                        clip(v - 50, 0, 255), clip(v + 50, 0, 255)]
        self._push_to_trackbars()

    def reset_selected(self):
        if self.active:
            self.profiles[self.selected].reset()
            self._push_to_trackbars()

    def draw_info(self):
        if not self.active:
            return
        p = self.profiles[self.selected]
        panel = np.zeros((90, 480, 3), np.uint8)
        cv2.putText(panel, f"Dang chinh: {p.label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, p.box_color, 2, cv2.LINE_AA)
        hmin, hmax, smin, smax, vmin, vmax = p.params
        cv2.putText(panel, f"H[{hmin},{hmax}] S[{smin},{smax}] V[{vmin},{vmax}]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        note = ("Do: dai Hue thu 2 doi xung qua 180" if p.wraps_hue
                else "Click len vat the de tu lay nguong")
        cv2.putText(panel, f"({note})", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(CALIB_WINDOW, panel)


# --------------------------------------------------------------------------- #
# NGUON ANH DAU VAO
# --------------------------------------------------------------------------- #
def choose_image_file(title: str = "Chọn ảnh để nhận diện màu sắc",
                      initial_dir: str = ".") -> Optional[str]:
    """Mo hop thoai do hoa chon tap tin anh tu may tinh (ho tro Unicode)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=[
                ("Tệp hình ảnh", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("PNG", "*.png"),
                ("Tất cả tệp", "*.*")
            ]
        )
        root.destroy()
        return path if path else None
    except Exception as e:
        print("[LOI] Khong the mo hop thoai chon file:", e)
        return None


def open_source(source: str):
    """Tra ve (cap, anh_tinh, lat_guong). Ho tro duong dan anh Unicode tren Windows."""
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return cap, None, True
    if os.path.splitext(source)[1].lower() in IMAGE_EXT:
        try:
            data = np.fromfile(source, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = cv2.imread(source)
        if img is not None and img.shape[1] > 960:
            scale = 960 / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale)
        return None, img, False
    return cv2.VideoCapture(source), None, False


# --------------------------------------------------------------------------- #
# CHUONG TRINH CHINH
# --------------------------------------------------------------------------- #
def main():
    candidates = ["sample_test.png", "AnhTest/Lego1.jpg", "AnhTest/NapChai.jpg", "AnhTest/rubik.jpg", "rubik.jpg", "nap_chai.jpg", "lego.jpg"]
    avail = [s for s in candidates if os.path.exists(s)]
    default_src = avail[0] if avail else None

    ap = argparse.ArgumentParser(
        description="Nhan dien 10 mau sac (tach rieng voi nen den phan mau)")
    ap.add_argument("--source", default=default_src,
                    help="duong dan anh/video hoac chi so webcam (0,1..)")
    ap.add_argument("--style", type=int, default=0, choices=[0, 1],
                    help="0: tach nen den phan mau giong anh mau (mac dinh), 1: ve tren anh goc + HUD")
    ap.add_argument("--gui", action="store_true",
                    help="Mo giao dien do hoa truc quan (Tkinter GUI)")
    args = ap.parse_args()

    # Neu nguoi dung yeu cau giao dien do hoa GUI
    if args.gui:
        try:
            import giaoDien
            giaoDien.main()
            return
        except Exception as e:
            print("[LOI] Khong mo duoc giao dien do hoa:", e)

    # Neu chua chi dinh nguon va khong co anh mau mac dinh -> mo hop thoai chon anh
    if args.source is None:
        print("[INFO] Khong tim thay anh mau mac dinh (rubik.jpg, nap_chai.jpg, lego.jpg).")
        print("[INFO] Dang mo hop thoai de ban chon anh tu may tinh...")
        chosen = choose_image_file()
        if chosen:
            args.source = chosen
        else:
            print("[INFO] Ban chua chon file anh. Thu chuyen sang Webcam (0)...")
            args.source = "0"

    sources = [s for s in (avail + ["0"]) if s]
    if args.source not in sources:
        sources.insert(0, args.source)
    src_idx = sources.index(args.source)

    cap, still, mirror = open_source(args.source)
    if cap is not None and not cap.isOpened():
        print("[LOI] Khong mo duoc nguon:", args.source)
        print("[GOI Y] Ban co the chay 'python giaoDien.py' de dung giao dien do hoa chon anh!")
        return
    if cap is None and still is None:
        print("[LOI] Khong doc duoc anh:", args.source)
        return

    profiles = build_profiles()
    calibrator = Calibrator(profiles)
    show_mask = False
    style = args.style
    show_hud = (style == 1)
    mode = 0
    white_balance = False
    bg_blur = None
    state = {"hsv": None, "last": None, "clean": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["hsv"] is not None:
            calibrator.sample_from_click(state["hsv"], x, y)

    cv2.namedWindow(MAIN_WINDOW)
    cv2.setMouseCallback(MAIN_WINDOW, on_mouse)
    print(__doc__)
    print(">>> Meo: Nhan phim 'o' de mo hop thoai chon anh tu may tinh bat cu luc nao!")
    print(">>> Meo: Nhan phim 'g' hoac chay 'python giaoDien.py' de mo giao dien do hoa day du!\n")
    tick_prev = cv2.getTickCount()

    while True:
        if still is not None:
            frame = still.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                print("[HET] Khong con khung hinh.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

        frame = simulate_lighting(frame, mode)
        if white_balance:
            frame = gray_world_balance(frame)
        calibrator.update()
        dets, selected_mask, state["hsv"] = detect(frame, profiles, calibrator.selected, bg_blur)
        state["clean"] = frame.copy()

        render_detections(frame, dets, style=style)

        tick_now = cv2.getTickCount()
        fps = cv2.getTickFrequency() / max(1, tick_now - tick_prev)
        tick_prev = tick_now
        src_label = os.path.basename(sources[src_idx]) if not sources[src_idx].isdigit() else f"CAM {sources[src_idx]}"
        if show_hud:
            draw_hud(frame, dets, profiles, fps, mode, white_balance, bg_blur is not None, src_label)

        state["last"] = frame
        cv2.imshow(MAIN_WINDOW, frame)
        calibrator.draw_info()
        if show_mask and selected_mask is not None:
            cv2.imshow(MASK_WINDOW, selected_mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('o'), ord('O')):
            print("[CHON ANH] Dang mo hop thoai chon anh tu may tinh...")
            chosen = choose_image_file()
            if chosen:
                if cap is not None:
                    cap.release()
                    cap = None
                if chosen not in sources:
                    sources.insert(0, chosen)
                src_idx = sources.index(chosen)
                cap, still, mirror = open_source(chosen)
                bg_blur = None
                print(f"[NGUON ANH MOI] {chosen}")
        elif key in (ord('g'), ord('G')):
            print("[GIAO DIEN] Dang khoi dong giao dien do hoa Tkinter...")
            try:
                import giaoDien
                giaoDien.main()
            except Exception as e:
                print("[LOI GIAO DIEN]", e)
        elif key in (ord('n'), 9):
            if cap is not None:
                cap.release()
            src_idx = (src_idx + 1) % len(sources)
            next_src = sources[src_idx]
            cap, still, mirror = open_source(next_src)
            bg_blur = None
            print("[NGUON]", next_src)
        elif key == ord('v'):
            style = 1 - style
            show_hud = (style == 1)
            print("[KIEU HIEN THI]", "Tren anh goc + BBox + HUD" if style == 1 else "Tach nen den phan mau giong anh mau")
        elif key == ord('h'):
            show_hud = not show_hud
            print("[HUD]", "BAT" if show_hud else "TAT")
        elif key == ord('c'):
            calibrator.toggle()
        elif key == ord('r'):
            calibrator.reset_selected()
        elif key == ord('b'):
            if bg_blur is None and state["clean"] is not None:
                bg_blur = cv2.GaussianBlur(state["clean"], BLUR_KERNEL, 0)
                print('[NEN] Da chup nen - chi nhan dien vung khac nen')
            else:
                bg_blur = None
                print('[NEN] Da xoa nen')
        elif key == ord('w'):
            white_balance = not white_balance
            print('[CAN BANG TRANG]', 'BAT' if white_balance else 'TAT')
        elif key == ord('m'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow(MASK_WINDOW)
        elif ord('0') <= key <= ord('4'):
            mode = key - ord('0')
            print(f"[ANH SANG] {LIGHTING_MODES[mode]}")
        elif key == ord('s') and state["last"] is not None:
            os.makedirs("output", exist_ok=True)
            path = os.path.join("output", time.strftime("kq_%Y%m%d_%H%M%S") + f"_L{mode}.png")
            cv2.imwrite(path, state["last"])
            print("[LUU]", path)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
