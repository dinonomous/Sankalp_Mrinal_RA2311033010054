from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('face_recognition_app.urls')),  # Your app's URLs
]
