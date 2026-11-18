#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giao diện nhận diện màu sắc & hình học (Tkinter + OpenCV).
Thiết kế chuẩn theo mẫu hiện đại (Dark Navy Theme, trực quan, tối giản).
"""

import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image, ImageTk

from main import (
    build_profiles,
    ColorProfile,
    Detection,
    simulate_lighting,
    gray_world_balance,
    compensate_lighting,
    raw_mask,
    clean_mask,
    detect,
    render_detections,
    Calibrator,
)

COLOR_NAMES = {
    "DO": "Đỏ",
    "CAM": "Cam",
    "VANG": "Vàng",
    "XANH LA MA": "Xanh lá mạ",
    "XANH LA": "Xanh lá",
    "XANH MINT": "Xanh bạc hà",
    "XANH DUONG": "Xanh dương",
    "TIM": "Tím",
    "TRANG": "Trắng",
    "DEN": "Đen",
}


def read_image_unicode(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_image_unicode(path: str, img: np.ndarray) -> bool:
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
        return False
    except Exception:
        return False


class ColorShapeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Nhận diện màu sắc")
        self.root.geometry("1100x720")
        self.root.minsize(980, 620)
        self.root.configure(bg="#0b1329")

        # Thuật toán & Profiles
        self.profiles = build_profiles()
        self.calibrator = Calibrator(self.profiles)

        # Dữ liệu ảnh
        self.raw_image: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None
        self.detections: List[Detection] = []
        self.current_filename = ""

        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera = False

        # Biến điều khiển
        self.var_style = tk.IntVar(value=0)       # 0: Tách nền đen, 1: Bounding box
        self.var_light = tk.IntVar(value=0)       # 0..4
        self.var_wb = tk.BooleanVar(value=False)  # Cân bằng trắng

        # Cache hình hiển thị
        self.photo_orig = None
        self.photo_result = None

        self._setup_styles()
        self._build_header_and_toolbar()
        self._build_dual_view()
        self._build_table_section()
        self._build_action_buttons()
        self._build_statusbar()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Configure>", self._on_resize)
        self._resize_job = None

        # Vẽ trạng thái trống ban đầu
        self.root.after(50, self.reset_views)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Bảng màu Dark Navy
        bg_main = "#0b1329"
        bg_card = "#0c162d"
        bg_header = "#14244d"
        text_light = "#cad5e8"

        style.configure("TFrame", background=bg_main)
        style.configure("Card.TFrame", background=bg_card)
        style.configure("Header.TFrame", background=bg_header)

        # Checkbutton
        style.configure("Dark.TCheckbutton",
                        background=bg_main,
                        foreground=text_light,
                        font=("Segoe UI", 9))
        style.map("Dark.TCheckbutton",
                  background=[("active", bg_main)],
                  foreground=[("active", "#ffffff")])

        # Combobox
        style.configure("TCombobox",
                        fieldbackground="#16233f",
                        background="#16233f",
                        foreground="#ffffff",
                        arrowcolor="#ffffff",
                        darkcolor="#1b2e56",
                        lightcolor="#1b2e56",
                        font=("Segoe UI", 9))
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#16233f")],
                  selectbackground=[("readonly", "#1d68d8")])

        # Bảng Treeview
        style.configure("Treeview",
                        background="#0c162d",
                        foreground="#ffffff",
                        fieldbackground="#0c162d",
                        font=("Segoe UI", 9),
                        rowheight=25)
        style.configure("Treeview.Heading",
                        background="#14244d",
                        foreground="#cbd5e1",
                        font=("Segoe UI", 9, "bold"),
                        relief="flat")
        style.map("Treeview.Heading", background=[("active", "#1e3368")])
        style.map("Treeview",
                  background=[("selected", "#1d68d8")],
                  foreground=[("selected", "#ffffff")])

    def _create_btn(self, parent, text, bg, hover_bg, fg="#ffffff", font=("Segoe UI", 9),
                    padx=14, pady=5, command=None):
        """Tạo nút phẳng tùy chỉnh màu sắc và hover mượt mà."""
        btn = tk.Button(parent, text=text, bg=bg, fg=fg, activebackground=hover_bg,
                        activeforeground=fg, bd=0, relief="flat", padx=padx, pady=pady,
                        font=font, cursor="hand2", command=command)
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    def _build_header_and_toolbar(self):
        """Thanh tiêu đề và thanh công cụ."""
        top_container = tk.Frame(self.root, bg="#0b1329", padx=16, pady=8)
        top_container.pack(fill=tk.X, side=tk.TOP)

        # DÒNG 1: BIỂU TƯỢNG VÀ TIÊU ĐỀ ỨNG DỤNG
        title_row = tk.Frame(top_container, bg="#0b1329")
        title_row.pack(fill=tk.X, anchor="w", pady=(0, 10))

        # Vẽ Icon khối lập phương 3D Isometric màu xanh
        icon_cv = tk.Canvas(title_row, width=28, height=28, bg="#0b1329", highlightthickness=0)
        icon_cv.pack(side=tk.LEFT, padx=(0, 10))

        top_poly = [14, 2, 25, 8, 14, 14, 3, 8]
        left_poly = [3, 8, 14, 14, 14, 25, 3, 19]
        right_poly = [25, 8, 14, 14, 14, 25, 25, 19]
        icon_cv.create_polygon(top_poly, fill="#1e3a6e", outline="#38bdf8", width=1.5)
        icon_cv.create_polygon(left_poly, fill="#162d55", outline="#38bdf8", width=1.5)
        icon_cv.create_polygon(right_poly, fill="#0f2142", outline="#38bdf8", width=1.5)

        lbl_app_name = tk.Label(title_row, text="Nhận diện màu sắc",
                                font=("Segoe UI", 15, "bold"),
                                bg="#0b1329", fg="#ffffff")
        lbl_app_name.pack(side=tk.LEFT)

        # DÒNG 2: THANH CÔNG CỤ (TOOLBAR)
        tb = tk.Frame(top_container, bg="#0b1329")
        tb.pack(fill=tk.X)

        # [Chọn ảnh] (Primary Blue Button)
        btn_open = self._create_btn(tb, "📁  Chọn ảnh", bg="#1d68d8", hover_bg="#2575e6",
                                    font=("Segoe UI", 9, "bold"), padx=14, pady=5,
                                    command=self.choose_image)
        btn_open.pack(side=tk.LEFT, padx=(0, 6))

        # [Webcam]
        self.btn_webcam = self._create_btn(tb, "📹  Webcam", bg="#1b2a47", hover_bg="#263b63",
                                           command=self.toggle_webcam)
        self.btn_webcam.pack(side=tk.LEFT, padx=(0, 6))

        # [Lưu ảnh]
        btn_save = self._create_btn(tb, "💾  Lưu ảnh", bg="#1b2a47", hover_bg="#263b63",
                                    command=self.save_image)
        btn_save.pack(side=tk.LEFT, padx=(0, 14))

        # Phân cách |
        sep1 = tk.Frame(tb, bg="#293e68", width=1, height=22)
        sep1.pack(side=tk.LEFT, padx=8, pady=3)

        # Chế độ
        tk.Label(tb, text="Chế độ:", bg="#0b1329", fg="#cad5e8", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(6, 4))
        self.cb_style = ttk.Combobox(tb, state="readonly", width=13,
                                     values=["Tách nền đen", "Bounding box"])
        self.cb_style.current(0)
        self.cb_style.bind("<<ComboboxSelected>>", self._on_style_change)
        self.cb_style.pack(side=tk.LEFT, padx=(0, 12))

        # Ánh sáng
        tk.Label(tb, text="Ánh sáng:", bg="#0b1329", fg="#cad5e8", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(6, 4))
        self.cb_light = ttk.Combobox(tb, state="readonly", width=12,
                                     values=["Bình thường", "Sáng mạnh", "Sáng yếu", "Ánh sáng vàng", "Bóng đổ"])
        self.cb_light.current(0)
        self.cb_light.bind("<<ComboboxSelected>>", self._on_light_change)
        self.cb_light.pack(side=tk.LEFT, padx=(0, 6))

        # Cân bằng trắng
        self.chk_wb = ttk.Checkbutton(tb, text="Cân bằng trắng", variable=self.var_wb,
                                      style="Dark.TCheckbutton", command=self._on_wb_toggle)
        self.chk_wb.pack(side=tk.LEFT, padx=(2, 10))

        # Phân cách |
        sep2 = tk.Frame(tb, bg="#293e68", width=1, height=22)
        sep2.pack(side=tk.LEFT, padx=8, pady=3)

        # Xem các bước 2, 3, 4, 5 (Đổi bước khi bấm hoặc chọn)
        tk.Label(tb, text="Bước:", bg="#0b1329", fg="#cad5e8", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(4, 4))
        self.step_options = [
            "Bước 2: Tiền xử lý (Blur & HSV)",
            "Bước 3: Phân đoạn màu (Masks)",
            "Bước 4: Nhận diện & Gắn nhãn",
            "Bước 5: Độ chính xác"
        ]
        self.cb_step = ttk.Combobox(tb, state="readonly", width=22, values=self.step_options)
        self.cb_step.current(2)  # Mặc định là Bước 4 (Nhận diện & Gắn nhãn)
        self.cb_step.bind("<<ComboboxSelected>>", self._on_step_change)
        self.cb_step.pack(side=tk.LEFT, padx=(0, 6))

        # Nút bấm đổi bước nhanh: [Đổi bước ❯]
        btn_next_step = self._create_btn(tb, "Đổi bước ❯", bg="#1e2f52", hover_bg="#2b426f",
                                         font=("Segoe UI", 9, "bold"), padx=10, pady=4,
                                         command=self.cycle_next_step)
        btn_next_step.pack(side=tk.LEFT, padx=(0, 6))

        # [Calib HSV] (ở góc phải)
        btn_calib = self._create_btn(tb, "⚙  Calib HSV", bg="#1b2a47", hover_bg="#263b63",
                                     command=self.calibrator.toggle)
        btn_calib.pack(side=tk.RIGHT)

    def _build_dual_view(self):
        """Khung hiển thị 2 ảnh song song: Ảnh gốc và Kết quả."""
        container = tk.Frame(self.root, bg="#0b1329", padx=16, pady=4)
        container.pack(fill=tk.BOTH, expand=True)

        # CỘT TRÁI: ẢNH GỐC
        card_orig = tk.Frame(container, bg="#1b2e56", bd=1)
        card_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        header_orig = tk.Frame(card_orig, bg="#14244d", padx=12, pady=6)
        header_orig.pack(fill=tk.X)
        tk.Label(header_orig, text="Ảnh gốc", font=("Segoe UI", 10, "bold"),
                 bg="#14244d", fg="#ffffff").pack(anchor="w")

        self.canvas_orig = tk.Canvas(card_orig, bg="#0c162d", highlightthickness=0)
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)

        # CỘT PHẢI: KẾT QUẢ
        card_res = tk.Frame(container, bg="#1b2e56", bd=1)
        card_res.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        header_res = tk.Frame(card_res, bg="#14244d", padx=12, pady=6)
        header_res.pack(fill=tk.X)
        self.lbl_card_res = tk.Label(header_res, text="Kết quả", font=("Segoe UI", 10, "bold"),
                                     bg="#14244d", fg="#ffffff")
        self.lbl_card_res.pack(anchor="w")

        self.canvas_result = tk.Canvas(card_res, bg="#0c162d", highlightthickness=0)
        self.canvas_result.pack(fill=tk.BOTH, expand=True)

    def _draw_placeholder(self, canvas: tk.Canvas, title: str, subtitle: str = ""):
        """Vẽ biểu tượng khung ảnh placeholder với núi và mặt trời."""
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 10 or h <= 10:
            return

        cx = w // 2
        cy = h // 2 - 18
        rw, rh = 34, 25

        # Khung tranh bo viền
        canvas.create_rectangle(cx - rw, cy - rh, cx + rw, cy + rh, outline="#4a5e84", width=2)
        # Mặt trời
        canvas.create_oval(cx + 12, cy - 17, cx + 22, cy - 7, outline="#4a5e84", width=2)
        # Đỉnh núi
        canvas.create_line(cx - rw, cy + 14, cx - 10, cy - 4, cx + 4, cy + 10,
                           cx + 18, cy + 2, cx + rw, cy + 16, fill="#4a5e84", width=2)

        # Dòng chữ chính
        canvas.create_text(cx, cy + rh + 22, text=title, fill="#ffffff", font=("Segoe UI", 11, "bold"))
        # Dòng phụ chú
        if subtitle:
            canvas.create_text(cx, cy + rh + 42, text=subtitle, fill="#8fa2c4", font=("Segoe UI", 9))

    def reset_views(self):
        """Đặt lại giao diện về trạng thái chờ."""
        self._draw_placeholder(self.canvas_orig, "Chưa có ảnh", 'Bấm "Chọn ảnh" hoặc "Webcam"')
        self._draw_placeholder(self.canvas_result, "Chờ xử lý...", "")
        if hasattr(self, "lbl_card_res"):
            self.lbl_card_res.config(text="Kết quả")

    def _build_table_section(self):
        """Bảng hiển thị danh sách vật thể phát hiện."""
        table_card = tk.Frame(self.root, bg="#1b2e56", bd=1)
        table_card.pack(fill=tk.X, padx=16, pady=(4, 6))

        # Tiêu đề bảng
        t_header = tk.Frame(table_card, bg="#14244d", padx=12, pady=5)
        t_header.pack(fill=tk.X)
        self.lbl_table_title = tk.Label(t_header, text="Vật thể phát hiện (0)",
                                        font=("Segoe UI", 10, "bold"),
                                        bg="#14244d", fg="#ffffff")
        self.lbl_table_title.pack(anchor="w")

        # Khung chứa Treeview
        tree_box = tk.Frame(table_card, bg="#0c162d")
        tree_box.pack(fill=tk.BOTH, expand=True)

        cols = ("stt", "color", "center", "bbox", "area", "purity")
        self.tree = ttk.Treeview(tree_box, columns=cols, show="headings", height=4)

        self.tree.heading("stt", text="#")
        self.tree.heading("color", text="Màu sắc")
        self.tree.heading("center", text="Tọa độ tâm (x, y)")
        self.tree.heading("bbox", text="Bounding Box (x, y, w, h)")
        self.tree.heading("area", text="Diện tích (px)")
        self.tree.heading("purity", text="Độ chính xác (%)")

        self.tree.column("stt", width=45, anchor="center")
        self.tree.column("color", width=140, anchor="center")
        self.tree.column("center", width=140, anchor="center")
        self.tree.column("bbox", width=180, anchor="center")
        self.tree.column("area", width=130, anchor="center")
        self.tree.column("purity", width=130, anchor="center")

        scroll = ttk.Scrollbar(tree_box, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_action_buttons(self):
        """Hai nút lớn ở giữa: [▶ Xử lý] và [↺ Xóa kết quả]."""
        btn_row = tk.Frame(self.root, bg="#0b1329", pady=6)
        btn_row.pack(fill=tk.X)

        center_box = tk.Frame(btn_row, bg="#0b1329")
        center_box.pack(anchor="center")

        # Nút xanh lá: [▶ Xử lý]
        btn_process = self._create_btn(center_box, "  ▶   Xử lý  ", bg="#16a34a", hover_bg="#22c55e",
                                       font=("Segoe UI", 10, "bold"), padx=22, pady=6,
                                       command=self.execute_processing)
        btn_process.pack(side=tk.LEFT, padx=10)

        # Nút xám xanh: [↺ Xóa kết quả]
        btn_clear = self._create_btn(center_box, "  ↺   Xóa kết quả  ", bg="#1e2f52", hover_bg="#2b426f",
                                     font=("Segoe UI", 10, "bold"), padx=20, pady=6,
                                     command=self.clear_results)
        btn_clear.pack(side=tk.LEFT, padx=10)

    def _build_statusbar(self):
        """Thanh trạng thái mảnh dưới cùng."""
        bar = tk.Frame(self.root, bg="#070e1e", height=24, padx=16)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Trạng thái bên trái: chấm xanh + chữ
        left_box = tk.Frame(bar, bg="#070e1e")
        left_box.pack(side=tk.LEFT, pady=3)

        self.dot_canvas = tk.Canvas(left_box, width=10, height=10, bg="#070e1e", highlightthickness=0)
        self.dot_canvas.pack(side=tk.LEFT, padx=(0, 6))
        self.dot_canvas.create_oval(2, 2, 8, 8, fill="#22c55e", outline="")

        self.lbl_status = tk.Label(left_box, text="Sẵn sàng", bg="#070e1e", fg="#cad5e8", font=("Segoe UI", 8))
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_stats = tk.Label(bar, text="0 vật thể  |  0 ms", bg="#070e1e", fg="#8899b5", font=("Segoe UI", 8))
        self.lbl_stats.pack(side=tk.RIGHT, pady=3)

    def choose_image(self):
        self._stop_webcam()
        path = filedialog.askopenfilename(
            title="Chọn ảnh để nhận diện",
            filetypes=[("Hình ảnh", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("Tất cả", "*.*")]
        )
        if not path:
            return

        img = read_image_unicode(path)
        if img is None:
            messagebox.showerror("Lỗi", f"Không thể đọc file ảnh:\n{path}")
            return

        if img.shape[1] > 960:
            scale = 960.0 / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self.current_filename = os.path.basename(path)
        self.raw_image = img

        self._render_canvas_image(self.canvas_orig, self.raw_image, is_orig=True)
        self._draw_placeholder(self.canvas_result, "Chờ xử lý...", "")

        self.lbl_status.config(text=f"Đã nạp: {self.current_filename}")
        self.execute_processing()

    def _on_style_change(self, _=None):
        self.var_style.set(self.cb_style.current())
        if self.raw_image is not None and not self.is_camera:
            self.execute_processing()

    def _on_light_change(self, _=None):
        self.var_light.set(self.cb_light.current())
        if self.raw_image is not None and not self.is_camera:
            self.execute_processing()

    def _on_wb_toggle(self):
        if self.raw_image is not None and not self.is_camera:
            self.execute_processing()

    def _on_step_change(self, _=None):
        if self.raw_image is not None:
            self.execute_processing()

    def cycle_next_step(self):
        """Bấm nút thì chuyển sang bước tiếp theo (2 -> 3 -> 4 -> 5 -> 2)."""
        idx = (self.cb_step.current() + 1) % len(self.step_options)
        self.cb_step.current(idx)
        if self.raw_image is not None:
            self.execute_processing()
        else:
            self.lbl_card_res.config(text=f"Kết quả - {self.step_options[idx]}")

    def render_step2(self, frame_raw: np.ndarray) -> np.ndarray:
        """Bước 2: Tiền xử lý - Áp dụng bù sáng, lọc Gaussian Blur (11x11), chuyển BGR -> HSV."""
        frame_proc = frame_raw
        if self.var_wb.get():
            frame_proc = gray_world_balance(frame_proc)
        balanced = compensate_lighting(frame_proc)
        blurred = cv2.GaussianBlur(balanced, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        out = blurred.copy()
        h, w = out.shape[:2]

        ih, iw = max(60, h // 4), max(80, w // 4)
        hue_channel = hsv[:, :, 0]
        hue_color = cv2.applyColorMap(hue_channel, cv2.COLORMAP_HSV)
        inset = cv2.resize(hue_color, (iw, ih), interpolation=cv2.INTER_AREA)
        cv2.rectangle(inset, (0, 0), (iw - 1, ih - 1), (255, 255, 255), 1)
        out[h - ih - 10:h - 10, w - iw - 10:w - 10] = inset
        cv2.putText(out, "Ban do Hue (H)", (w - iw - 8, h - ih - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        overlay = out.copy()
        cv2.rectangle(overlay, (8, 8), (400, 80), (11, 19, 41), -1)
        cv2.addWeighted(overlay, 0.8, out, 0.2, 0, out)
        cv2.rectangle(out, (8, 8), (400, 80), (56, 189, 248), 1)

        cv2.putText(out, "BUOC 2: TIEN XU LY ANH", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 2, cv2.LINE_AA)
        cv2.putText(out, "- Giam nhieu: Gaussian Blur (ksize=11x11)", (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
        cv2.putText(out, "- Chuyen doi: BGR -> HSV & Bu sang thich nghi", (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
        return out

    def render_step3(self, frame_raw: np.ndarray) -> np.ndarray:
        """Bước 3: Phân đoạn màu sắc theo ngưỡng HSV (xử lý màu Đỏ 2 đầu Hue & Morphological)."""
        frame_proc = frame_raw
        if self.var_wb.get():
            frame_proc = gray_world_balance(frame_proc)
        balanced = compensate_lighting(frame_proc)
        blurred = cv2.GaussianBlur(balanced, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, w = frame_raw.shape[:2]

        canvas = np.full((h, w, 3), 18, dtype=np.uint8)

        for p in self.profiles:
            m = raw_mask(hsv, p)
            mc = clean_mask(m, p)
            canvas[mc > 0] = p.box_color

        # Viền bao quanh các vùng mặt nạ
        gray_all = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        cnts, _ = cv2.findContours((gray_all > 20).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)

        # Bảng thông tin Bước 3
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (430, 80), (11, 19, 41), -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
        cv2.rectangle(canvas, (8, 8), (430, 80), (74, 222, 128), 1)

        cv2.putText(canvas, "BUOC 3: PHAN DOAN MAU SAC (HSV MASKS)", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (74, 222, 128), 2, cv2.LINE_AA)
        cv2.putText(canvas, "- Ap dung 10 dai nguong HSV doc lap", (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Mau Do: Gop 2 dai Hue [0,5] & [170,180]", (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
        return canvas

    def render_step4(self, frame_raw: np.ndarray, dets: List[Detection]) -> np.ndarray:
        """Bước 4: Nhận diện & phân loại - Contour, Bounding Box, Tâm, Diện tích, Gắn nhãn."""
        res = frame_raw.copy()
        render_detections(res, dets, style=self.var_style.get())

        overlay = res.copy()
        cv2.rectangle(overlay, (8, 8), (400, 80), (11, 19, 41), -1)
        cv2.addWeighted(overlay, 0.8, res, 0.2, 0, res)
        cv2.rectangle(res, (8, 8), (400, 80), (250, 204, 21), 1)

        cv2.putText(res, "BUOC 4: NHAN DIEN & GAN NHAN MAU", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (250, 204, 21), 2, cv2.LINE_AA)
        cv2.putText(res, f"- Phat hien: {len(dets)} vat the bang Contour", (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
        cv2.putText(res, "- Tinh tam (Moments), BBox, dien tich & gan nhan", (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
        return res

    def render_step5(self, frame_raw: np.ndarray, dets: List[Detection]) -> np.ndarray:
        """Bước 5: Đánh giá độ chính xác - Tình huống ánh sáng & xử lý màu gần giống nhau."""
        mode = self.var_light.get()
        mode_names = ["Binh thuong", "Sang manh", "Sang yeu", "Sang vang", "Bong do"]
        mode_str = mode_names[mode] if 0 <= mode < len(mode_names) else "Binh thuong"

        res = frame_raw.copy()
        render_detections(res, dets, style=self.var_style.get())

        avg_purity = (sum(d.purity for d in dets) / max(1, len(dets))) * 100

        overlay = res.copy()
        cv2.rectangle(overlay, (8, 8), (440, 80), (11, 19, 41), -1)
        cv2.addWeighted(overlay, 0.8, res, 0.2, 0, res)
        cv2.rectangle(res, (8, 8), (440, 80), (244, 63, 94), 1)

        cv2.putText(res, "BUOC 5: DANH GIA DO CHINH XAC", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (244, 63, 94), 2, cv2.LINE_AA)
        cv2.putText(res, f"- Tinh huong anh sang: {mode_str}", (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
        cv2.putText(res, f"- Loc mau gan nhau: IoU > 0.50 | Do chinh xac: {avg_purity:.1f}%", (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
        return res

    def execute_processing(self):
        """Thực hiện xử lý ảnh theo bước đang chọn (Bước 2, 3, 4 hoặc 5)."""
        if self.raw_image is None:
            messagebox.showinfo("Thông báo", "Vui lòng chọn ảnh hoặc bật Webcam trước khi xử lý.")
            return

        t0 = time.perf_counter()
        frame = self.raw_image.copy()

        # Ánh sáng
        mode = self.var_light.get()
        frame = simulate_lighting(frame, mode)

        # Cân bằng trắng
        if self.var_wb.get():
            frame = gray_world_balance(frame)

        # Nhận diện cơ sở (Contour, BBox, v.v.)
        dets, _, _ = detect(frame, self.profiles, selected=self.calibrator.selected)
        self.detections = dets

        # Sinh ảnh theo bước đang chọn
        step_idx = self.cb_step.current()
        if step_idx == 0:     # Bước 2: Tiền xử lý
            res = self.render_step2(frame)
            step_name = "Bước 2: Tiền xử lý (Blur & HSV)"
        elif step_idx == 1:   # Bước 3: Phân đoạn màu sắc
            res = self.render_step3(frame)
            step_name = "Bước 3: Phân đoạn màu sắc (Masks)"
        elif step_idx == 2:   # Bước 4: Nhận diện & Gắn nhãn
            res = self.render_step4(frame, dets)
            step_name = "Bước 4: Nhận diện & Gắn nhãn màu"
        else:                 # Bước 5: Đánh giá độ chính xác
            res = self.render_step5(frame, dets)
            step_name = "Bước 5: Đánh giá độ chính xác"

        self.result_image = res
        cost_ms = (time.perf_counter() - t0) * 1000.0

        # Cập nhật tiêu đề khung kết quả và ảnh
        self.lbl_card_res.config(text=f"Kết quả - {step_name}")
        self._render_canvas_image(self.canvas_result, self.result_image, is_orig=False)
        self._update_table(dets)

        self.lbl_stats.config(text=f"{len(dets)} vật thể  |  {cost_ms:.0f} ms")
        self.lbl_status.config(text=f"Đang hiển thị: {step_name}")

    def clear_results(self):
        """Xóa bỏ kết quả nhận diện, đưa về trạng thái chờ."""
        self.result_image = None
        self.detections = []
        self._draw_placeholder(self.canvas_result, "Chờ xử lý...", "")
        self.lbl_card_res.config(text="Kết quả")

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.lbl_table_title.config(text="Vật thể phát hiện (0)")
        self.lbl_stats.config(text="0 vật thể  |  0 ms")
        self.lbl_status.config(text="Sẵn sàng")

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.lbl_table_title.config(text="Vật thể phát hiện (0)")
        self.lbl_stats.config(text="0 vật thể  |  0 ms")
        self.lbl_status.config(text="Sẵn sàng")

    def _update_table(self, dets: List[Detection]):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for i, d in enumerate(dets, 1):
            vn_color = COLOR_NAMES.get(d.profile.label, d.profile.label)
            cx, cy = d.center
            x, y, w, h = d.bbox
            self.tree.insert("", "end", values=(
                i,
                vn_color,
                f"({cx}, {cy})",
                f"({x}, {y}, {w}, {h})",
                f"{int(d.area):,}",
                f"{d.purity * 100:.0f}%"
            ))

        self.lbl_table_title.config(text=f"Vật thể phát hiện ({len(dets)})")

    # --------------------------------------------------------------------------- #
    # WEBCAM & LƯU ẢNH
    # --------------------------------------------------------------------------- #
    def toggle_webcam(self):
        if self.is_camera:
            self._stop_webcam()
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showwarning("Thông báo", "Không thể kết nối với Webcam thiết bị.")
                self.cap = None
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_camera = True
            self.btn_webcam.config(text="⏹  Dừng Webcam", bg="#852222")
            self.lbl_status.config(text="Webcam đang chạy...")
            self._camera_loop()

    def _stop_webcam(self):
        if self.is_camera:
            self.is_camera = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.btn_webcam.config(text="📹  Webcam", bg="#1b2a47")
            self.lbl_status.config(text="Đã dừng Webcam")

    def _camera_loop(self):
        if not self.is_camera or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self.raw_image = cv2.flip(frame, 1)
            self._render_canvas_image(self.canvas_orig, self.raw_image, is_orig=True)
            self.execute_processing()
        if self.is_camera:
            self.root.after(40, self._camera_loop)

    def save_image(self):
        if self.result_image is None:
            messagebox.showinfo("Thông báo", "Chưa có ảnh kết quả để lưu.")
            return

        os.makedirs("output", exist_ok=True)
        default_name = f"kq_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            initialdir="output",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")]
        )
        if path and save_image_unicode(path, self.result_image):
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{os.path.basename(path)}")

    # --------------------------------------------------------------------------- #
    # RENDER ẢNH LÊN CANVAS
    # --------------------------------------------------------------------------- #
    def _render_canvas_image(self, canvas: tk.Canvas, cv_img: np.ndarray, is_orig: bool):
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 10 or h <= 10 or cv_img is None:
            return

        ih, iw = cv_img.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized = cv2.resize(cv_img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        if is_orig:
            self.photo_orig = photo
        else:
            self.photo_result = photo

        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, image=photo, anchor="center")

    def _on_resize(self, _):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._delayed_resize)

    def _delayed_resize(self):
        if self.raw_image is not None:
            self._render_canvas_image(self.canvas_orig, self.raw_image, is_orig=True)
        else:
            self._draw_placeholder(self.canvas_orig, "Chưa có ảnh", 'Bấm "Chọn ảnh" hoặc "Webcam"')

        if self.result_image is not None:
            self._render_canvas_image(self.canvas_result, self.result_image, is_orig=False)
        else:
            self._draw_placeholder(self.canvas_result, "Chờ xử lý...", "")

    def on_close(self):
        self._stop_webcam()
        cv2.destroyAllWindows()
        self.root.destroy()


def main():
    root = tk.Tk()
    ColorShapeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
