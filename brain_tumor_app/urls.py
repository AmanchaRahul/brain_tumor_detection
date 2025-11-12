from django.urls import path
from . import views

app_name = 'brain_tumor_app'

urlpatterns = [
    path('', views.predict_tumor, name='predict_tumor'),
]
