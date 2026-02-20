# video_analyzer/api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    HealthCheckAPIView,
    VideoPredictAPIView,
    VideoAnalysisStatusAPIView,
    VideoAnalysisViewSet
)

router = DefaultRouter()
router.register(r'analyses', VideoAnalysisViewSet, basename='analysis')

urlpatterns = [
    path('health/', HealthCheckAPIView.as_view(), name='health-check'),
    path('predict/', VideoPredictAPIView.as_view(), name='video-predict'),
    path('analyses/<uuid:analysis_id>/status/', 
         VideoAnalysisStatusAPIView.as_view(), name='analysis-status'),
    path('', include(router.urls)),
]