# ============================================
# FILE 2: video_analyzer/api/permissions.py
# ============================================
from rest_framework import permissions
from video_analyzer.models import APIKey

class HasValidAPIKey(permissions.BasePermission):
    message = 'Valid API key required'
    
    def has_permission(self, request, view):
        return request.auth is not None and isinstance(request.auth, APIKey)