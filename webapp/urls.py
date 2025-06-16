from django.urls import path
from webapp import views

urlpatterns = [
    path("", views.home, name="home"),
    path("bbm92", views.bbm92, name="bbm92"),
    path("decoy_bb84", views.decoy_bb84, name="decoy_bb84"),
    path("cow_qkd", views.cowqkd, name="cow_qkd"),
    path("dps_qkd", views.dpsqkd, name="dps_qkd"),
    path("bb84", views.bb84, name="bb84"),
]