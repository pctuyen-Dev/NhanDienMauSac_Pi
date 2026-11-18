# Hệ Thống Nhận Diện Màu Sắc Vật Thể (Color Recognition System)

Hệ thống nhận diện, phân loại màu sắc và định vị vật thể thời gian thực sử dụng thị giác máy tính (OpenCV + Python), tối ưu hóa cho môi trường PC và thiết bị nhúng như **Raspberry Pi**.

---

## 🌟 Tính Năng Nổi Bật

1. **Thu nhận ảnh & Video:**
   - Hỗ trợ tải ảnh tĩnh định dạng đa dạng (`.jpg`, `.png`, `.bmp`) và hỗ trợ đường dẫn chứa ký tự tiếng Việt.
   - Hỗ trợ luồng camera/webcam thời gian thực mượt mà.

2. **Tiền xử lý & Cân bằng sáng:**
   - Chuyển đổi không gian màu chuẩn từ BGR sang HSV.
   - Khử nhiễu ảnh hiệu quả bằng bộ lọc làm mịn Gaussian Blur ($11 \times 11$).
   - Thuật toán **Cân bằng trắng (Gray-World Assumption)** giúp triệt tiêu hiện tượng ám màu do nguồn sáng môi trường (đèn vàng, đèn huỳnh quang).
   - Tự động bù sáng thích nghi theo cường độ kênh Value ($V$).

3. **Phân đoạn màu sắc chuyên sâu:**
   - Nhận diện chính xác 10 nhóm màu: **Đỏ, Cam, Vàng, Xanh lá mạ, Xanh lá, Xanh bạc hà, Xanh dương, Tím, Trắng, Đen**.
   - Xử lý hoàn hảo hiện tượng gián đoạn dải Hue của màu Đỏ (kết hợp 2 dải $[0, 5]$ và $[170, 180]$).
   - Lọc hình thái học Morphological (Open & Close) làm sạch mặt nạ nhị phân.

4. **Định vị & Gắn nhãn vật thể:**
   - Phát hiện biên dạng đường bao Contour của từng vật thể.
   - Xác định chính xác trọng tâm hình học $(c_x, c_y)$ qua Moment bậc một ($m_{10}/m_{00}, m_{01}/m_{00}$).
   - Tính toán khung bao viền Bounding Box, diện tích (Area), và độ thuần màu (Purity %).
   - Cơ chế triệt tiêu bóng đổ (Shadow Suppression) thông qua kiểm soát ngưỡng $S_{min}$ và lọc diện tích $A_{min}$.
   - Khử xung đột trùng lặp vật thể bằng chỉ số IoU (Intersection over Union).

5. **Giao diện trực quan (Tkinter Dark Navy):**
   - Thiết kế hiện đại, bảng màu Dark Navy tối ưu trải nghiệm.
   - Cho phép quan sát từng bước xử lý theo yêu cầu đồ án:
     - **Bước 2:** Tiền xử lý (Gaussian Blur, BGR $\to$ HSV, bản đồ kênh Hue).
     - **Bước 3:** Phân đoạn màu sắc (HSV Masks).
     - **Bước 4:** Nhận diện và gắn nhãn (Bounding Box, tâm, diện tích).
     - **Bước 5:** Đánh giá độ chính xác trong các điều kiện ánh sáng (Bình thường, Sáng mạnh, Sáng yếu, Ánh sáng vàng, Bóng đổ).
   - Bảng kết quả định dạng tiếng Việt rõ ràng, chi tiết từng vật thể.
   - Tích hợp cửa sổ **Calib HSV** động để tinh chỉnh ngưỡng màu trực tiếp.

---

## 📁 Cấu Trúc Dự Án

```text
NhanDienMauSac/
├── main.py              # Engine xử lý thị giác máy tính, phân đoạn màu HSV & thuật toán
├── giaoDien.py          # Ứng dụng giao diện người dùng Tkinter hiện đại
├── AnhTest/             # Bộ ảnh dữ liệu thử nghiệm thực tế (Lego, Nắp chai, Rubik)
├── requirements.txt     # Danh sách các thư viện phụ thuộc
├── .gitignore           # Cấu hình bỏ qua các tệp không cần thiết
└── README.md            # Tài liệu hướng dẫn sử dụng dự án
```

---

## 🚀 Cài Đặt & Sử Dụng

### 1. Yêu cầu môi trường
- Python 3.8+ (khuyến nghị Python 3.10 - 3.14)
- Webcam (nếu sử dụng tính năng nhận diện trực tiếp)

### 2. Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng giao diện
```bash
python giaoDien.py
```

### 4. Chạy chế độ dòng lệnh (CLI / Test Engine)
```bash
# Chạy với webcam mặc định
python main.py

# Chạy kiểm thử trên file ảnh
python main.py --image AnhTest/Lego1.jpg
```

---

## 🛠️ Phím Tắt Khi Chạy Engine (main.py)
- `S`: Chuyển đổi giữa chế độ tách nền đen và vẽ bounding box trên ảnh gốc.
- `L`: Giả lập đổi chế độ ánh sáng (Bình thường $\to$ Sáng mạnh $\to$ Sáng yếu $\to$ Ánh sáng vàng $\to$ Bóng đổ).
- `W`: Bật/tắt thuật toán Cân bằng trắng (Gray-World).
- `C`: Mở/đóng cửa sổ Calibrator tinh chỉnh ngưỡng màu HSV.
- `Q` hoặc `ESC`: Thoát chương trình.
