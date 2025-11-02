# File này chứa các "view" của Django, xử lý các yêu cầu HTTP từ người dùng.
from django.shortcuts import render
from .forms import ForecastForm     # Import lớp Form để xử lý dữ liệu từ form HTML.
from .engine import run_forecast    # Import hàm xử lý dự báo chính từ file engine.py.
import os
from django.conf import settings   # Dùng để truy cập các cài đặt của dự án, ví dụ MEDIA_ROOT.
import json                        # Dùng để chuyển đổi đối tượng Python (list, dict) thành chuỗi JSON cho JavaScript.


# View này xử lý cả việc hiển thị trang upload và xử lý dữ liệu khi người dùng submit form.
def upload_view(request):

    # --- Xử lý khi người dùng gửi dữ liệu lên bằng phương thức POST (nhấn nút submit) ---
    if request.method == 'POST':
        # Khởi tạo một đối tượng Form với dữ liệu từ request.POST (dữ liệu text) và request.FILES (dữ liệu file).
        form = ForecastForm(request.POST, request.FILES)

        # Kiểm tra xem dữ liệu người dùng gửi lên có hợp lệ theo các quy tắc đã định nghĩa trong Form không.
        if form.is_valid():
            # 1. Lưu thông tin từ form (tên file, số ngày) vào cơ sở dữ liệu.
            #    Hành động này cũng tự động lưu file được upload vào thư mục MEDIA_ROOT/uploads.
            #    `instance` là một đối tượng của model `ForecastRequest` vừa được tạo.
            instance = form.save() 

            # 2. Lấy đường dẫn file và số ngày
            file_path = instance.data_file.path # `path` trả về đường dẫn tuyệt đối tới file.
            num_days = instance.forecast_days # Lấy số ngày từ instance model.
            
            try:
                # 3. GỌI HÀM DỰ BÁO CHÍNH. Đây là bước tốn nhiều thời gian nhất.
                # Hàm này giờ trả về kết quả của cả 3 mô hình
                sarima_results, prophet_results, lstm_results, history_data, sarima_commentary, prophet_commentary, lstm_commentary, comparison_commentary = run_forecast(file_path, num_days)

                # 4. Chuẩn bị dữ liệu để vẽ biểu đồ (cho Chart.js)

                # Chuyển đổi index (ngày tháng) và giá trị của dữ liệu lịch sử thành các list.
                history_dates = history_data.index.strftime('%Y-%m-%d').tolist()
                history_values = [float(v) for v in history_data.values] # Chuyển đổi sang float gốc của Python
                
                # Trích xuất ngày và giá trị dự báo cho SARIMA
                sarima_dates = [item['date'] for item in sarima_results]
                sarima_values = [float(item['forecast']) for item in sarima_results] # Chuyển đổi sang float
                
                # Trích xuất ngày và giá trị dự báo cho Prophet
                prophet_dates = [item['date'] for item in prophet_results]
                prophet_values = [float(item['forecast']) for item in prophet_results] # Chuyển đổi sang float
                
                # Trích xuất ngày và giá trị dự báo cho LSTM
                lstm_dates = [item['date'] for item in lstm_results]
                lstm_values = [float(item['forecast']) for item in lstm_results] # Chuyển đổi sang float

                # 5. Tạo một dictionary `context` để chứa tất cả dữ liệu cần gửi sang template HTML.
                context = {
                    'sarima_results': sarima_results, # Gửi kết quả SARIMA
                    'prophet_results': prophet_results, # Gửi kết quả Prophet
                    'lstm_results': lstm_results, # Gửi kết quả LSTM
                    'num_days': num_days,
                    'file_name': os.path.basename(instance.data_file.name), # Chỉ lấy tên file, không lấy đường dẫn.
                    # Dùng json.dumps để chuyển đổi list Python thành chuỗi có định dạng mảng JavaScript.
                    'history_dates_json': json.dumps(history_dates),
                    'history_values_json': json.dumps(history_values),
                    'sarima_json': json.dumps({'dates': sarima_dates, 'values': sarima_values}),
                    'prophet_json': json.dumps({'dates': prophet_dates, 'values': prophet_values}),
                    'lstm_json': json.dumps({'dates': lstm_dates, 'values': lstm_values}),
                    'sarima_commentary': sarima_commentary,
                    'prophet_commentary': prophet_commentary,
                    'lstm_commentary': lstm_commentary,
                    'comparison_commentary': comparison_commentary,
                }

                # 6. Render (dựng) trang results_page.html với dữ liệu trong `context` và trả về cho người dùng.
                return render(request, 'forecast/results_page.html', context)

            except Exception as e:
                # 7. Nếu có bất kỳ lỗi nào xảy ra trong quá trình xử lý của `run_forecast`...
                # Xóa bản ghi đã tạo trong CSDL để tránh lưu lại các yêu cầu lỗi.
                instance.delete() 

                # Thêm thông báo lỗi chung vào form để hiển thị cho người dùng trên trang upload.
                form.add_error(None, f"LỖI XỬ LÝ FILE: {e}")

                # Render lại trang upload, truyền vào form chứa lỗi để người dùng biết vấn đề.
                return render(request, 'forecast/upload_page.html', {'form': form, 'error': str(e)})

    # --- Xử lý khi người dùng mới truy cập trang (phương thức GET) ---
    else:
        form = ForecastForm() # Tạo một đối tượng Form rỗng, chưa có dữ liệu.

    # Render trang upload_page.html và truyền form rỗng vào để hiển thị.
    return render(request, 'forecast/upload_page.html', {'form': form})