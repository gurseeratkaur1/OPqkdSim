from django.urls import path
from webapp import views

urlpatterns = [
    path("", views.home, name="home"),
    path("bb84", views.bb84, name="bb84"),
]