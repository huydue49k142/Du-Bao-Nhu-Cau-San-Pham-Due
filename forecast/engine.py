# File này chứa "bộ não" xử lý chính của ứng dụng dự báo.
import pandas as pd

# --- KHẮC PHỤC LỖI PROPHET ---
# Đoạn code này kiểm tra và cài đặt backend 'cmdstan' nếu nó chưa tồn tại.
# Đây là bước quan trọng để giải quyết lỗi "'Prophet' object has no attribute 'stan_backend'".
import cmdstanpy

try:
    # Cách kiểm tra đúng: gọi cmdstan_path() và xem nó có trả về đường dẫn hợp lệ không.
    # Nếu không tìm thấy, nó sẽ báo lỗi và chúng ta sẽ bắt lỗi đó để cài đặt.
    path = cmdstanpy.cmdstan_path()
    print(f"Đã tìm thấy backend CmdStan tại: {path}")
except Exception:
    print("Không tìm thấy backend CmdStan. Bắt đầu cài đặt (có thể mất vài phút)...")
    cmdstanpy.install_cmdstan(overwrite=True, verbose=True)
# -----------------------------

import numpy as np
import pmdarima as pm
import math
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose # Dùng để xử lý ngoại lệ
from prophet import Prophet # Import thư viện Prophet
from sklearn.preprocessing import MinMaxScaler # Dùng để chuẩn hóa dữ liệu cho LSTM
from tensorflow.keras.models import Sequential # Dùng để xây dựng mô hình LSTM
from tensorflow.keras.layers import LSTM, Dense # Các lớp trong mô hình LSTM

warnings.filterwarnings('ignore') # Tắt các cảnh báo không quan trọng từ các thư viện


# --- Hằng số cho mô hình LSTM ---
LOOK_BACK = 52 # Sử dụng 52 tuần trước đó để dự đoán tuần tiếp theo

# --- Hàm 1: Phát hiện ngoại lệ (Copy từ .ipynb) ---
def get_outlier(dec_func):
    """
    Hàm này phát hiện các điểm ngoại lệ (outlier) 
    bằng phương pháp 1.5xIQR từ phần dư (residual).
    - dec_func: Chuỗi pandas Series chứa phần dư (residual) từ phân rã chuỗi thời gian.
    """
    dec_func = dec_func.dropna() # Xóa các giá trị NaN có thể có
    q1 = dec_func.quantile(0.25) # Tìm phân vị thứ nhất (Q1)
    q3 = dec_func.quantile(0.75) # Tìm phân vị thứ ba (Q3)
    iqr = q3 - q1 # Tính khoảng tứ phân vị (Interquartile Range)
    # Tạo một bộ lọc boolean: True cho các giá trị nằm ngoài khoảng [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    filt = (dec_func <= (q1 - 1.5 * iqr)) + (dec_func >= (q3 + 1.5 * iqr))
    return dec_func[filt] # Trả về các giá trị được xác định là ngoại lệ

# --- Hàm mới: Tạo Nhận Xét Tự Động ---
def generate_commentary(history_data, sarima_results, prophet_results, lstm_results, num_days):
    """
    Tự động tạo ra các nhận xét, phân tích dựa trên dữ liệu lịch sử và kết quả dự báo.
    Trả về một dictionary chứa các nhận xét riêng biệt cho SARIMA, Prophet, LSTM và một nhận xét so sánh.
    """
    commentaries = {
        'sarima': [],
        'prophet': [],
        'lstm': [],
        'comparison': []
    }

    # --- Nhận xét cho SARIMA ---
    if sarima_results:
        forecast_values = [item['forecast'] for item in sarima_results]
        total_forecast_demand = sum(forecast_values)
        avg_forecast_demand = total_forecast_demand / len(forecast_values) if forecast_values else 0

        commentaries['sarima'].append(f"- Triển vọng dự báo SARIMA cho {num_days} ngày tới: Tổng nhu cầu dự báo là khoảng {int(total_forecast_demand):,} đơn vị, trung bình {int(avg_forecast_demand):,} đơn vị/tuần.")
        commentaries['sarima'].append("- Mô hình SARIMA đã tính đến yếu tố mùa vụ (chu kỳ 52 tuần), giúp dự báo các đỉnh và đáy nhu cầu trong tương lai một cách tương đối. Mô hình này phù hợp với các chuỗi thời gian có tính mùa vụ và xu hướng rõ ràng.")
        # Có thể bổ sung thêm nhận xét về xu hướng tăng/giảm nếu có

    # --- Nhận xét cho Prophet ---
    if prophet_results:
        forecast_values = [item['forecast'] for item in prophet_results]
        total_forecast_demand = sum(forecast_values)
        avg_forecast_demand = total_forecast_demand / len(forecast_values) if forecast_values else 0

        commentaries['prophet'].append(f"- Triển vọng dự báo Prophet cho {num_days} ngày tới: Tổng nhu cầu dự báo là khoảng {int(total_forecast_demand):,} đơn vị, trung bình {int(avg_forecast_demand):,} đơn vị/tuần.")
        commentaries['prophet'].append("- Mô hình Prophet (Facebook Prophet) được thiết kế để xử lý tốt các chuỗi thời gian có tính mùa vụ mạnh, các ngày lễ và các thay đổi đột ngột của xu hướng. Nó thường linh hoạt hơn trong việc nắm bắt các sự kiện bất thường.")
        # Có thể bổ sung thêm nhận xét về các thành phần mùa vụ, xu hướng của Prophet

    # --- Nhận xét cho LSTM ---
    if lstm_results:
        forecast_values = [item['forecast'] for item in lstm_results]
        total_forecast_demand = sum(forecast_values)
        avg_forecast_demand = total_forecast_demand / len(forecast_values) if forecast_values else 0

        commentaries['lstm'].append(f"- Triển vọng dự báo LSTM cho {num_days} ngày tới: Tổng nhu cầu dự báo là khoảng {int(total_forecast_demand):,} đơn vị, trung bình {int(avg_forecast_demand):,} đơn vị/tuần.")
        commentaries['lstm'].append("- Mô hình LSTM (Long Short-Term Memory) là một mạng nơ-ron hồi quy, có khả năng học các phụ thuộc dài hạn phức tạp trong dữ liệu. Nó không dựa vào các giả định thống kê như SARIMA và có thể nắm bắt các mẫu phi tuyến tính mà các mô hình khác có thể bỏ lỡ.")

    # --- Nhận xét so sánh ---
    if sarima_results and prophet_results and lstm_results:
        sarima_total = sum([item['forecast'] for item in sarima_results])
        prophet_total = sum([item['forecast'] for item in prophet_results])
        lstm_total = sum([item['forecast'] for item in lstm_results])

        commentaries['comparison'].append("--- Đánh giá và So sánh Mô hình Dự báo ---")
        commentaries['comparison'].append(f"Tổng nhu cầu dự báo: SARIMA ({int(sarima_total):,} đơn vị), Prophet ({int(prophet_total):,} đơn vị), và LSTM ({int(lstm_total):,} đơn vị).")

        # So sánh sự tương đồng
        all_totals = [sarima_total, prophet_total, lstm_total]
        if (max(all_totals) - min(all_totals)) / np.mean(all_totals) < 0.15: # Nếu chênh lệch tương đối dưới 15%
            commentaries['comparison'].append("Cả ba mô hình đều đưa ra kết quả dự báo khá tương đồng, cho thấy sự đồng thuận cao trong việc ước tính nhu cầu tương lai.")
        else:
            commentaries['comparison'].append("Có sự khác biệt đáng chú ý giữa các mô hình. SARIMA tập trung vào tính mùa vụ và xu hướng tuyến tính. Prophet linh hoạt với các sự kiện và thay đổi đột ngột. LSTM có khả năng học các mẫu phi tuyến tính phức tạp mà hai mô hình kia có thể không nắm bắt được.")

        commentaries['comparison'].append("Để lựa chọn mô hình tốt nhất, cần đánh giá dựa trên các tiêu chí như độ chính xác (MAE, RMSE) trên tập dữ liệu kiểm định. SARIMA phù hợp với dữ liệu ổn định, có mùa vụ rõ ràng. Prophet mạnh khi có ngày lễ hoặc thay đổi xu hướng. LSTM có thể vượt trội nếu dữ liệu có các mối quan hệ phức tạp, phi tuyến tính, nhưng đòi hỏi nhiều dữ liệu hơn để huấn luyện hiệu quả.")

    return {k: "\n".join(v) for k, v in commentaries.items()} # Nối các dòng nhận xét lại với nhau, cách nhau bởi ký tự xuống dòng

# --- Hàm định dạng kết quả ---
def format_results(dates, values):
    """Hàm tiện ích để định dạng kết quả dự báo thành danh sách các dictionary."""
    return [{'date': date.strftime('%Y-%m-%d'), 'forecast': round(value, 2)} for date, value in zip(dates, values)]

# --- Hàm 2: HÀM LOGIC CHÍNH (Bộ não) ---
def run_forecast(file_path, num_days):
    """
    Hàm này nhận đường dẫn file CSV và số ngày, sau đó chạy toàn bộ quy trình huấn luyện và dự báo cho cả SARIMA, Prophet và LSTM.
    """

    # === 1. Đọc và Làm sạch Dữ liệu ===
    try:
        # Dùng 'utf-8-sig' để xử lý các file CSV có thể có ký tự lạ ở đầu
        df = pd.read_csv(file_path, parse_dates=['Date'], encoding='utf-8-sig')
    except Exception as e:
        # Nếu không đọc được file, báo lỗi cụ thể
        raise ValueError(f"Không thể đọc file CSV. Lỗi: {e}. Hãy đảm bảo file có định dạng UTF-8.")

    df.sort_values(by=['Date'], inplace=True) # Sắp xếp dữ liệu theo ngày tháng tăng dần
    df.dropna(subset=['Date'], inplace=True) # Xóa các dòng không có giá trị ở cột 'Date'

    # Làm sạch cột Order_Demand: loại bỏ khoảng trắng, dấu ngoặc đơn
    df.Order_Demand = df.Order_Demand.astype(str).str.replace(pat=' ', repl = '')
    df.Order_Demand = df.Order_Demand.str.replace(pat='(', repl = '')
    df.Order_Demand = df.Order_Demand.str.replace(pat=')', repl = '')

    # Chuyển cột Order_Demand sang dạng số. Nếu giá trị nào không thể chuyển, nó sẽ thành NaN (Not a Number).
    df.Order_Demand = pd.to_numeric(df.Order_Demand, errors='coerce')
    df.dropna(subset=['Order_Demand'], inplace=True) # Xóa các dòng có giá trị NaN ở cột Order_Demand
    df.Order_Demand = df.Order_Demand.astype('int32') # Chuyển kiểu dữ liệu sang số nguyên 32-bit để tiết kiệm bộ nhớ

    # === 2. Lọc và Chuẩn bị Chuỗi thời gian ===
    # Lọc dữ liệu chỉ cho kho 'Whse_J' và danh mục sản phẩm 'Category_019'
    try:
        dft = df[(df['Warehouse'] == 'Whse_J') & (df['Product_Category'] == 'Category_019')]
    except KeyError:
        # Nếu file không có các cột cần thiết, báo lỗi
        raise KeyError("File CSV tải lên phải chứa cột 'Warehouse' và 'Product_Category'.")

    if dft.empty:
        # Nếu sau khi lọc không còn dữ liệu, báo lỗi
        raise ValueError("Không tìm thấy dữ liệu cho 'Whse_J' và 'Category_019' trong file.")

    dft.set_index(dft['Date'], inplace=True) # Đặt cột 'Date' làm chỉ số (index) của DataFrame
    pd.to_datetime(dft.index) # Đảm bảo index có kiểu dữ liệu là datetime

    # Tổng hợp dữ liệu từ hàng ngày/hàng giờ thành hàng tuần ('W' = Weekly), giá trị là tổng của các ngày trong tuần.
    dft_week = dft.Order_Demand.resample('W').sum()
    dft_week = dft_week.interpolate() # Nội suy để điền vào các tuần có thể bị thiếu dữ liệu (giá trị = 0)
    dft_week = dft_week.to_frame(name = 'Order_Demand') # Chuyển từ Series về lại DataFrame

    # Chỉ lấy dữ liệu từ ngày 1/1/2012 trở đi để đồng bộ với phân tích trong file notebook
    dft_week = dft_week[dft_week.index >= '2012-01-01']

    # Kiểm tra xem có đủ dữ liệu để huấn luyện mô hình mùa vụ không (cần ít nhất 2 chu kỳ, tức 104 tuần)
    if dft_week.empty or len(dft_week) < LOOK_BACK + 1: # Cần đủ dữ liệu cho cửa sổ nhìn lại của LSTM
         raise ValueError(f"Không đủ dữ liệu (cần ít nhất {LOOK_BACK + 1} tuần) từ 2012 trở đi để huấn luyện.")

    # === 3. Xử lý Ngoại lệ (Outlier) ===
    # Phân rã chuỗi thời gian thành các thành phần: xu hướng, mùa vụ, và phần dư.
    # `period=52` vì dữ liệu theo tuần và có tính mùa vụ theo năm (52 tuần).
    dec = seasonal_decompose(dft_week.Order_Demand, model='addictive', period=52)
    outliers = get_outlier(dec.resid.dropna()) # Tìm các điểm ngoại lệ trên phần dư

    # Thay thế các giá trị ngoại lệ bằng NaN, sau đó dùng phương pháp nội suy tuyến tính để điền vào.
    dft_week.loc[outliers.index] = np.nan
    dft_week.Order_Demand.interpolate(inplace=True)

    ts_train = dft_week['Order_Demand'] # Đây là dữ liệu cuối cùng để huấn luyện

    # Chuyển đổi số ngày người dùng nhập sang số tuần, luôn làm tròn lên để đảm bảo bao phủ hết khoảng thời gian.
    n_weeks = math.ceil(num_days / 7)
    last_date = ts_train.index[-1]
    
    # --- 4a. Chạy mô hình SARIMA ---
    # Sử dụng các tham số tốt nhất đã tìm được từ quá trình phân tích.
    m_sarima = pm.arima.ARIMA(order=(1, 1, 1), seasonal_order=(0, 1, 1, 52), suppress_warnings=True)
    m_sarima.fit(ts_train) # Huấn luyện mô hình
    
    # Dự báo với SARIMA
    sarima_forecast_values = m_sarima.predict(n_periods=n_weeks)
    sarima_future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_weeks, freq='W')
    
    # Định dạng kết quả SARIMA
    sarima_results = format_results(sarima_future_dates, sarima_forecast_values)

    # --- 4b. Chạy mô hình Prophet ---
    # Prophet yêu cầu DataFrame có cột 'ds' (datestamp) và 'y' (value).
    prophet_df = dft_week.reset_index().rename(columns={'Date': 'ds', 'Order_Demand': 'y'})

    # Khởi tạo mô hình Prophet với mùa vụ hàng năm (tương đương 52 tuần).
    m_prophet = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
    m_prophet.fit(prophet_df) # Huấn luyện mô hình

    # Tạo DataFrame cho các ngày trong tương lai để dự báo
    future_df = m_prophet.make_future_dataframe(periods=n_weeks, freq='W')
    
    # Thực hiện dự báo với Prophet
    prophet_forecast_df = m_prophet.predict(future_df)
    
    # Lấy các giá trị và ngày dự báo cho tương lai
    prophet_forecast_values = prophet_forecast_df['yhat'].iloc[-n_weeks:]
    prophet_future_dates = prophet_forecast_df['ds'].iloc[-n_weeks:]

    # Định dạng kết quả Prophet
    prophet_results = format_results(prophet_future_dates, prophet_forecast_values)

    # --- 4c. Chạy mô hình LSTM ---
    # Chuẩn bị dữ liệu cho LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ts_train.values.reshape(-1, 1))

    # Tạo dữ liệu huấn luyện dạng sequence
    def create_dataset(dataset, look_back=LOOK_BACK):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    train_X, train_Y = create_dataset(scaled_data)

    # Reshape đầu vào thành [samples, time steps, features]
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))

    # Xây dựng và huấn luyện mô hình LSTM
    m_lstm = Sequential()
    m_lstm.add(LSTM(50, input_shape=(LOOK_BACK, 1))) # 50 units
    m_lstm.add(Dense(1))
    m_lstm.compile(loss='mean_squared_error', optimizer='adam')
    m_lstm.fit(train_X, train_Y, epochs=500, batch_size=1, verbose=0) # Huấn luyện
    m_lstm.fit(train_X, train_Y, epochs=lstm_epochs, batch_size=1, verbose=0) # Huấn luyện

    # Dự báo với LSTM
    lstm_forecast_values_scaled = []
    # Lấy `look_back` giá trị cuối cùng từ dữ liệu đã scale để bắt đầu dự báo
    input_seq = scaled_data[-LOOK_BACK:].flatten().tolist()

    for _ in range(n_weeks):
        # Reshape lại input để phù hợp với mô hình
        current_input = np.array(input_seq).reshape((1, LOOK_BACK, 1))
        # Dự báo giá trị tiếp theo
        pred_scaled = m_lstm.predict(current_input, verbose=0)
        # Lưu giá trị dự báo (đã scale)
        lstm_forecast_values_scaled.append(pred_scaled[0, 0])
        # Cập nhật chuỗi input: bỏ giá trị đầu tiên, thêm giá trị dự báo vào cuối
        input_seq.pop(0)
        input_seq.append(pred_scaled[0, 0])

    # Chuyển đổi các giá trị dự báo về thang đo ban đầu
    lstm_forecast_values = scaler.inverse_transform(np.array(lstm_forecast_values_scaled).reshape(-1, 1))
    # Làm phẳng mảng kết quả
    lstm_forecast_values = lstm_forecast_values.flatten()

    # Tạo ngày cho các giá trị dự báo
    lstm_future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_weeks, freq='W')

    # Đảm bảo không có giá trị âm
    lstm_forecast_values[lstm_forecast_values < 0] = 0

    # Định dạng kết quả LSTM
    lstm_results = format_results(lstm_future_dates, lstm_forecast_values)


    # === 5. Tạo nhận xét ===
    all_commentaries = generate_commentary(ts_train, sarima_results, prophet_results, lstm_results, num_days)
    sarima_commentary = all_commentaries['sarima']
    prophet_commentary = all_commentaries['prophet']
    lstm_commentary = all_commentaries['lstm']
    comparison_commentary = all_commentaries['comparison']

    # Trả về kết quả của cả 3 mô hình và các nhận xét
    return sarima_results, prophet_results, lstm_results, ts_train, sarima_commentary, prophet_commentary, lstm_commentary, comparison_commentary