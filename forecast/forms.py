from django import forms
from .models import ForecastRequest # Import "bản thiết kế" model của bạn

# Tạo một lớp Form dựa trên Model
class ForecastForm(forms.ModelForm):
    class Meta:
        model = ForecastRequest  # Dùng model ForecastRequest
        fields = ['data_file', 'forecast_days'] # Chỉ hiện 2 trường này

        # Tùy chỉnh tên nhãn (cho đẹp)
        labels = {
            'data_file': 'Tải lên file CSV của bạn',
            'forecast_days': 'Số ngày cần dự báo (ví dụ: 30, 60, 90)'
        }

        # Tùy chỉnh giao diện (thêm class của Bootstrap cho đẹp)
        widgets = {
            'data_file': forms.FileInput(attrs={
                'class': 'form-control',
                'required': True # Bắt buộc phải upload
            }),
            'forecast_days': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Ví dụ: 30',
                'required': True # Bắt buộc phải nhập
            }),
        }