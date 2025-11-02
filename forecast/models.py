from django.db import models

class ForecastRequest(models.Model):
    
    # Sửa lỗi thiếu dấu '='
    data_file = models.FileField(upload_to='uploads/')
    
    forecast_days = models.IntegerField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"File: {self.data_file.name} ({self.forecast_days} days)"