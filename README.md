# 📈 Ứng dụng Tối ưu hóa Danh mục Đầu tư bằng LSTM-GRU

Ứng dụng Streamlit cao cấp này được xây dựng nhằm mục đích tối ưu hóa tỷ trọng phân bổ danh mục đầu tư cho thị trường chứng khoán Việt Nam. Bằng cách kết hợp mô hình học sâu lai giữa **LSTM** (Long Short-Term Memory) và **GRU** (Gated Recurrent Unit), ứng dụng tự động phân tích dữ liệu lịch sử giá để dự đoán và phân bổ tỷ trọng tối ưu nhằm đạt được **hệ số Sharpe (Sharpe Ratio)** cao nhất.

---

## ✨ Các tính năng chính

1. **📥 Tải dữ liệu linh hoạt**: 
   - Tải dữ liệu giá lịch sử trực tiếp từ API `vnstock` dựa trên danh mục ngành học từ file `industry_tickers.py`.
   - Cơ chế tự động thử lại (retry) và dự phòng (fallback) qua các nguồn dữ liệu khác nhau (KBS, TCBS, DNSE) để tránh lỗi API.
   - Hỗ trợ tải lên **file CSV tự chọn** chứa thông tin giá đóng cửa lịch sử của bạn để chạy phân tích trực tiếp.

2. **📊 Khám phá dữ liệu trực quan**:
   - Biểu đồ tương tác (Plotly) hiển thị biến động giá đóng cửa chuẩn hóa.
   - Phân tích và trực quan hóa hệ số Sharpe của các cổ phiếu trong tập huấn luyện để lọc ra các mã tối ưu.

3. **🧠 Huấn luyện mô hình sâu**:
   - Giao diện huấn luyện mô hình hiển thị tiến trình (progress bar) và thông tin loss trực quan theo thời gian thực (real-time).
   - Cơ chế huấn luyện ensemble qua nhiều seed ngẫu nhiên nhằm tìm kiếm seed tốt nhất hoạt động trên tập dữ liệu Test.
   - Tùy chỉnh trực tiếp các siêu tham số học sâu: Epochs, Batch Size, Lookback Window, Learning Rate, Lãi suất phi rủi ro ngay trên giao diện.

4. **📈 Phân bổ tỷ trọng & Treemap**:
   - Hiển thị bảng tỷ trọng tối ưu được tính toán từ mô hình.
   - Biểu đồ khối phân bổ tỷ trọng (**Treemap**) và biểu đồ cột trực quan, hiện đại.

5. **🏁 So sánh hiệu quả các chiến lược**:
   - So sánh chi tiết 3 chiến lược: **LSTM-GRU (Dynamic)**, **Phân bổ đều (1/N)**, và **Phân bổ 80-20**.
   - Biểu đồ trục kép (Dual Axis) so sánh Lợi nhuận kỳ vọng, Độ lệch chuẩn (Rủi ro) và Hệ số Sharpe.
   - Biểu đồ tăng trưởng lũy kế của 1 đồng vốn ban đầu trên tập Test (Cumulative Returns).

---

## 📁 Cấu trúc thư mục dự án

```text
├── app.py                  # Mã nguồn chính ứng dụng Streamlit
├── industry_tickers.py     # Từ điển phân loại ngành và mã cổ phiếu Việt Nam
├── requirements.txt        # Danh sách thư viện Python cần thiết
└── README.md               # Hướng dẫn sử dụng và triển khai dự án
```

---

## 💻 Hướng dẫn chạy ứng dụng cục bộ (Local)

### 1. Chuẩn bị môi trường
Yêu cầu máy tính cài đặt sẵn **Python 3.9 - 3.11**. Bạn nên tạo một môi trường ảo (virtual environment) để tránh xung đột thư viện:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
.\venv\Scripts\activate

# Kích hoạt môi trường ảo (macOS/Linux)
source venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
Cài đặt tất cả các dependencies trong file `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng
Chạy lệnh Streamlit để khởi động ứng dụng trên trình duyệt web:

```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở tại địa chỉ mặc định: `http://localhost:8501`.

---

## 🚀 Hướng dẫn triển khai (Deploy) lên Streamlit Community Cloud

Bạn có thể đưa ứng dụng này lên internet miễn phí bằng cách thực hiện các bước sau:

1. **Đẩy mã nguồn lên GitHub**:
   - Tạo một GitHub repository mới (chế độ Public).
   - Đẩy tất cả các file trong thư mục này (`app.py`, `industry_tickers.py`, `requirements.txt`, `README.md`) lên nhánh `main` (hoặc `master`) của repository.

2. **Truy cập Streamlit Cloud**:
   - Truy cập vào trang web [Streamlit Share](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.

3. **Tạo ứng dụng mới**:
   - Nhấn vào nút **"New app"**.
   - Chọn repository bạn vừa tải code lên.
   - Điền nhánh tương ứng (ví dụ: `main`).
   - Điền file chạy chính (Main file path): `app.py`.

4. **Nhấn "Deploy"**:
   - Hệ thống của Streamlit sẽ tự động tạo máy chủ ảo, tải thư viện từ `requirements.txt` và khởi chạy ứng dụng của bạn trên internet. Bạn có thể chia sẻ đường dẫn đó cho bất kỳ ai!

---

## ⚠️ Lưu ý quan trọng khi triển khai trên Streamlit Cloud

- **Giới hạn RAM**: Streamlit Community Cloud giới hạn bộ nhớ RAM ở mức **1 GB**. Do thư viện TensorFlow và việc huấn luyện mạng học sâu có thể tiêu tốn khá nhiều RAM, khuyến nghị người dùng nên giữ số lượng **Epochs huấn luyện nhỏ (khoảng 10 - 30)** khi chạy thử nghiệm trên đám mây để tránh trường hợp máy chủ bị tràn bộ nhớ và sập ứng dụng (crashed).
- **Giới hạn Rate Limit của API**: Việc tải dữ liệu liên tục từ các sàn qua API vnstock có thể bị hệ thống chặn tạm thời nếu gọi quá nhiều yêu cầu cùng lúc. Ứng dụng đã tích hợp sẵn cơ chế nghỉ nghỉ giãn cách, tuy nhiên nếu gặp sự cố tải, bạn hãy chuẩn bị sẵn file CSV dữ liệu giá đóng cửa lịch sử và tải lên qua tính năng **"Tải lên file CSV"**.
