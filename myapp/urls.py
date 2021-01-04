from django.urls import path, include
from rest_framework import routers
from . import views
router = routers.DefaultRouter()
router.register('myapp', views.ApprovalsView)
urlpatterns = [
    path('',views.approvereject,name="approvereject"),
    path('form/',views.cxcontact,name="cxcontact"),
    path('api/', include(router.urls)),
]