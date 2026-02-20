# ============================================
# FILE 1: video_analyzer/api/authentication.py (FIXED)
# ============================================
from rest_framework import authentication, exceptions
from django.core.cache import cache
from django.utils.timezone import now
from datetime import timedelta
import logging
from video_analyzer.models import APIKey
from django.db.models import F
logger = logging.getLogger(__name__)
from drf_spectacular.extensions import OpenApiAuthenticationExtension

class APIKeySchema(OpenApiAuthenticationExtension):
    target_class = 'video_analyzer.api.authentication.APIKeyAuthentication'  # full import path
    name = 'API Key Authentication'

    def get_security_definition(self, auto_schema):
        return {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-KEY',
            'description': 'API Key Authentication with rate limiting'
        }

class APIKeyAuthentication(authentication.BaseAuthentication):
    """
    API Key Authentication dengan rate limiting
    """
    
    def authenticate(self, request):
        # Get API key from header
        api_key = request.META.get('HTTP_X_API_KEY')
        
        if not api_key:
            logger.debug("No API key provided in headers")
            return None
        
        return self.authenticate_credentials(api_key)
    
    def authenticate_credentials(self, key):
        """
        Authenticate API key dengan caching dan rate limiting
        """
        # Cache key untuk API Key object
        cache_key = f'api_key_{key}'
        
        # Coba ambil dari cache
        api_key_obj = cache.get(cache_key)
        
        if not api_key_obj:
            try:
                # Get from database dengan select_related
                api_key_obj = APIKey.objects.select_related('user').get(
                    key=key, 
                    is_active=True
                )
                # Cache untuk 5 menit
                cache.set(cache_key, api_key_obj, 300)
                logger.info(f"API Key authenticated from DB: {api_key_obj.name}")
                
            except APIKey.DoesNotExist:
                logger.warning(f"Invalid API key attempted: {key[:8]}...")
                raise exceptions.AuthenticationFailed('Invalid API key')
            
        # Check rate limit
        self.check_rate_limit(api_key_obj)
        
        try:
            APIKey.objects.filter(pk=api_key_obj.pk).update(
                usage_count=F('usage_count') + 1,
                last_used=now()
            )
        
        except Exception as e:
            logger.error(f"Failed to update API usage stats: {e}")
        
        # Return (user, auth) tuple
        return (api_key_obj.user, api_key_obj)
    
    def check_rate_limit(self, api_key_obj):
        """
        Check rate limit dengan sliding window
        Returns: (is_allowed, remaining_requests, reset_time)
        """
        # Key berdasarkan jam saat ini (sliding window per jam)
        if not api_key_obj.rate_limit:
            return
        
        current_time = now()
        window_start = current_time.replace(
            minute=0, 
            second=0, 
            microsecond=0
        )
        
        window_timestamp = int(window_start.timestamp())
        
        rate_key = f'ratelimit:{api_key_obj.key}:{window_timestamp}'
        
        # Waktu kadaluarsa cache: sisa waktu menuju jam berikutnya
        next_hour = window_start + timedelta(hours=1)
        timeout = int((next_hour - current_time).total_seconds())
        
        # ---------------------------------------------------------
        # LOGIKA ATOMIC CACHE (PENTING UNTUK MENGHINDARI RACE CONDITION)
        # ---------------------------------------------------------
        
        # 1. Coba tambahkan key baru (add hanya berhasil jika key belum ada)
        # Ini menginisialisasi counter jadi 1 jika key belum ada
        added = cache.add(rate_key, 1, timeout)
        if added:
            current_count = 1
            
        else:
            # 2. Jika key sudah ada, increment counternya
            try:
                current_count = cache.incr(rate_key)
            except ValueError:
                # Fallback langka jika key expire tepat setelah check 'added'
                cache.set(rate_key, 1, timeout)
                current_count = 1
        
        if current_count >= api_key_obj.rate_limit:
            wait_seconds = int((next_hour - current_time).total_seconds())
            
            raise exceptions.Throttled(
                wait=wait_seconds,
                detail=f"Rate limit exceeded. Limit: {api_key_obj.rate_limit}/hour."
            )
    
    def authenticate_header(self, request):
        """Return WWW-Authenticate header string"""
        return 'API-Key'