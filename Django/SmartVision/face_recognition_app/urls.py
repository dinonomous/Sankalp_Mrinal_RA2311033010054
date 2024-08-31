from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('live_feed/', views.live_feed, name='live_feed'),
]
