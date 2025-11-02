# Ứng dụng Dự báo Nhu cầu Sản phẩm bằng Django và AI

 <!-- Bạn có thể thay link này bằng ảnh chụp màn hình ứng dụng của bạn -->

Dự án này là một ứng dụng web được xây dựng bằng Django (Python), cho phép người dùng tải lên dữ liệu lịch sử và thực hiện dự báo nhu cầu sản phẩm trong tương lai. Ứng dụng sử dụng đồng thời ba mô hình Time Series mạnh mẽ: **SARIMA**, **Facebook Prophet**, và **LSTM (Mạng nơ-ron hồi quy)** để đưa ra các kết quả dự báo đa dạng, kèm theo nhận xét và so sánh tự động.

Đây là đồ án môn học "Phân Tích Dữ Liệu Bằng Python" của Nhóm 7 - Lớp MIS3041_49K29.1, Trường Đại học Kinh tế - Đại học Đà Nẵng.

## 🌟 Tính năng chính

- **Giao diện Web thân thiện**: Người dùng có thể dễ dàng tải lên file dữ liệu (`.csv`) và nhập số ngày cần dự báo.
- **Xử lý dữ liệu tự động**:
  - Đọc và làm sạch dữ liệu từ file CSV.
  - Tổng hợp dữ liệu theo tuần (`resample`).
  - Phát hiện và xử lý các điểm dữ liệu bất thường (outliers).
- **Huấn luyện đồng thời 3 mô hình**:
  - **SARIMA**: Mô hình thống kê mạnh mẽ cho dữ liệu có tính mùa vụ.
  - **Prophet**: Mô hình của Facebook, linh hoạt với các thay đổi về xu hướng và các sự kiện đặc biệt.
  - **LSTM**: Mô hình học sâu (Deep Learning) có khả năng nắm bắt các quy luật phi tuyến tính phức tạp.
- **Trực quan hóa kết quả**:
  - Hiển thị biểu đồ so sánh dữ liệu lịch sử và kết quả dự báo của cả ba mô hình.
  - Cung cấp bảng dữ liệu dự báo chi tiết.
- **Tạo nhận xét tự động**: Hệ thống tự động phân tích và đưa ra các nhận xét về kết quả của từng mô hình cũng như so sánh tổng quan giữa chúng.

## 🛠️ Công nghệ sử dụng

- **Backend**: Django, Python
- **Phân tích & Mô hình AI**:
  - `pandas` & `numpy` cho xử lý dữ liệu.
  - `pmdarima` cho mô hình SARIMA.
  - `prophet` cho mô hình Prophet.
  - `tensorflow` & `scikit-learn` cho mô hình LSTM.
- **Frontend**: HTML, Bootstrap 5, Chart.js (để vẽ biểu đồ).

---

## 🚀 Hướng dẫn Cài đặt và Chạy dự án

### Yêu cầu

1.  **Python** (khuyến nghị phiên bản 3.10).
    - *Lưu ý: Khi cài đặt, nhớ tick vào ô "Add Python to PATH".*
2.  **Git** (để clone repository).

### Các bước cài đặt

1.  **Clone repository về máy**:
    Mở Terminal (hoặc Git Bash) và chạy lệnh sau:
    ```bash
    git clone <URL-repository-cua-ban>
    cd <ten-thu-muc-du-an>
    ```

2.  **Tạo và kích hoạt môi trường ảo**:
    Môi trường ảo giúp cô lập các thư viện của dự án, tránh xung đột.
    ```bash
    # Tạo môi trường ảo tên là 'venv'
    python -m venv venv

    # Kích hoạt môi trường ảo
    # Trên Windows (PowerShell/CMD):
    .\venv\Scripts\activate
    # Trên macOS/Linux:
    source venv/bin/activate
    ```
    *Sau khi kích hoạt, bạn sẽ thấy `(venv)` ở đầu dòng lệnh.*

    #Lệnh Tắt Môi Trường Ảo (Deactivate)
    deactivate

3.  **Cài đặt các thư viện cần thiết**:
    Tất cả các thư viện yêu cầu đã được liệt kê trong file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Lưu ý: Quá trình này có thể mất vài phút vì cần tải các thư viện lớn như `tensorflow`.*

4.  **Xử lý lỗi `Microsoft Visual C++` (Nếu có)**:
    Nếu quá trình cài đặt ở bước 3 báo lỗi `Microsoft Visual C++ 14.0 or greater is required`, bạn cần cài đặt **Build Tools for Visual Studio**:
    - Truy cập: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    - Tải và chạy trình cài đặt.
    - Trong tab "Workloads", chọn **"Desktop development with C++"** và tiến hành cài đặt.
    - **Khởi động lại máy tính**, sau đó kích hoạt lại môi trường ảo và chạy lại lệnh `pip install -r requirements.txt`.

5.  **Khởi tạo cơ sở dữ liệu**:
    Lệnh này sẽ tạo file database `db.sqlite3` cho Django.
    ```bash
    python manage.py migrate
    ```

6.  **Chạy ứng dụng**:
    ```bash
    python manage.py runserver
    ```

7.  **Truy cập ứng dụng**:
    Mở trình duyệt và truy cập vào địa chỉ: http://127.0.0.1:8000/

---

## 📖 Cách sử dụng

1.  Tại trang chủ, nhấn nút **"Choose File"** và chọn một file `.csv` chứa dữ liệu chuỗi thời gian.
    - *Lưu ý: File CSV cần có các cột `Date`, `Order_Demand`, `Warehouse`, và `Product_Category` để hệ thống xử lý đúng.*
2.  Nhập **số ngày cần dự báo** vào ô tương ứng (ví dụ: `365`).
3.  Nhấn nút **"Bắt đầu Dự báo"**.
4.  Chờ trong giây lát để hệ thống huấn luyện các mô hình. Sau khi hoàn tất, bạn sẽ được chuyển đến trang kết quả với các biểu đồ và phân tích chi tiết.

## 📂 Cấu trúc thư mục

```
├── forecast/         # Django app chính của dự án
│   ├── migrations/
│   ├── engine.py     # "Bộ não" xử lý, chứa logic huấn luyện và dự báo
│   ├── views.py      # Xử lý các request HTTP và logic của trang web
│   └── ...
├── templates/        # Chứa các file HTML
├── venv/             # Thư mục môi trường ảo (được gitignore)
├── db.sqlite3        # File cơ sở dữ liệu
├── manage.py         # File quản lý của Django
└── requirements.txt  # Danh sách các thư viện Python cần thiết
```