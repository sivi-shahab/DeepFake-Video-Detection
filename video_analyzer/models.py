# video_analyzer/models.py
from django.db import models
from django.contrib.auth.models import User
import uuid
import secrets
from django.conf import settings
from django.utils import timezone


class APIKey(models.Model):
    """
    API Key for authentication
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, help_text="Descriptive name for this API key")
    key = models.CharField(max_length=64, unique=True, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='api_keys'
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True, blank=True)
    usage_count = models.PositiveIntegerField(default=0)
    rate_limit = models.PositiveIntegerField(
        default=1000,
        help_text="Maximum requests per hour"
    )
    
    _raw_key = None
    
    class Meta:
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"
        ordering = ['-created_at']
        
        indexes = [
            models.Index(fields=['user', 'is_active'])
        ]
    
    def __str__(self):
        return f"{self.name} ({self.mask_key()})"
    
    def save(self, *args, **kwargs):
        if not self.key:
            # 1. Generate Key dengan Prefix
            # Prefix 'sk_' (Secret Key) standar industri memudahkan scanning security
            prefix = "sk_ml_" 
            random_part = secrets.token_urlsafe(32)
            self.key = f"{prefix}{random_part}"
            
            # 2. Simpan di variabel sementara untuk ditampilkan ke User (hanya saat create)
            self._raw_key = self.key
            
        super().save(*args, **kwargs)
    
    def increment_usage(self):
        """
        Atomic update tanpa race condition.
        Dihapus: refresh_from_db() untuk menghemat 1 query DB per request.
        """
        APIKey.objects.filter(pk=self.pk).update(
            usage_count=F('usage_count') + 1,
            last_used=timezone.now()
        )
    
    def mask_key(self):
        """Return masked key for display (e.g., sk_ml_...XY12)"""
        if self.key and len(self.key) > 10:
            return f"{self.key[:6]}...{self.key[-4:]}"
        return "****"


class VideoAnalysis(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    PREDICTION_CHOICES = [
        ('REAL', 'Real'),
        ('FAKE', 'Fake'),
    ]
    
    # Identifiers
    analysis_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True
    )
    
    # HYBRID SOLUTION: Keep both fields for compatibility
    # 1. ForeignKey untuk relasi proper (baru)
    api_key_rel = models.ForeignKey(
        APIKey,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='video_analyses',
        verbose_name='API Key (Object)'
    )
    
    # 2. CharField untuk backward compatibility (sudah ada data)
    api_key = models.CharField(
        max_length=255, 
        null=True, 
        blank=True,
        verbose_name='API Key (String)'
    )
    
    # Video info
    original_filename = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/%Y/%m/%d/')
    video_size = models.BigIntegerField()  # in bytes
    sequence_length = models.IntegerField(default=60)
    
    # Processing
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    prediction = models.CharField(max_length=10, choices=PREDICTION_CHOICES, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)  # 0-100
    real_probability = models.FloatField(null=True, blank=True)  # 0-1
    fake_probability = models.FloatField(null=True, blank=True)  # 0-1
    processing_time = models.FloatField(null=True, blank=True)  # in seconds
    model_used = models.CharField(max_length=255, null=True, blank=True)
    frames_processed = models.IntegerField(null=True, blank=True)
    heatmap_file = models.ImageField(upload_to='heatmaps/%Y/%m/%d/', null=True, blank=True)
    
    # Errors
    error_message = models.TextField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.analysis_id} - {self.status}"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Video Analysis'
        verbose_name_plural = 'Video Analyses'
    
    def save(self, *args, **kwargs):
        """Sync both API key fields"""
        # Jika api_key_rel diset, sync ke api_key (string)
        if self.api_key_rel and not self.api_key:
            self.api_key = self.api_key_rel.key
        
        # Jika api_key (string) diset, coba cari object-nya
        elif self.api_key and not self.api_key_rel:
            try:
                self.api_key_rel = APIKey.objects.get(key=self.api_key)
            except APIKey.DoesNotExist:
                # Biarkan api_key_rel null jika key tidak ditemukan
                pass
        
        super().save(*args, **kwargs)
    
    @property
    def effective_api_key(self):
        """Get API key object, prioritizing api_key_rel"""
        return self.api_key_rel or self.get_api_key_object()
    
    def get_api_key_object(self):
        """Try to get APIKey object from string"""
        if self.api_key:
            try:
                return APIKey.objects.get(key=self.api_key)
            except APIKey.DoesNotExist:
                return None
        return None

class UserProfile(models.Model):
    """Extended User Profile"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'user_profiles'
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
    
    def __str__(self):
        return f"Profile: {self.user.username}"


class VideoAnalysis(models.Model):
    """Model untuk menyimpan hasil analisis video"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    PREDICTION_CHOICES = [
        ('REAL', 'Real'),
        ('FAKE', 'Fake'),
    ]
    
    id = models.AutoField(primary_key=True)
    analysis_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses', 
                            null=True, blank=True)
    api_key = models.ForeignKey(APIKey, on_delete=models.SET_NULL, null=True, 
                                blank=True, related_name='analyses')
    
    # Video Information
    original_filename = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    video_size = models.IntegerField(help_text="Size in bytes")
    video_duration = models.FloatField(null=True, blank=True, help_text="Duration in seconds")
    
    # Analysis Settings
    sequence_length = models.IntegerField(default=60)
    
    # Results
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    prediction = models.CharField(max_length=10, choices=PREDICTION_CHOICES, 
                                    null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    real_probability = models.FloatField(null=True, blank=True)
    fake_probability = models.FloatField(null=True, blank=True)
    
    # Processing Information
    processing_time = models.FloatField(null=True, blank=True, help_text="Time in seconds")
    model_used = models.CharField(max_length=100, null=True, blank=True)
    frames_processed = models.IntegerField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    
    # Heatmap
    heatmap_file = models.FileField(upload_to='heatmaps/', null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'video_analyses'
        ordering = ['-created_at']
        verbose_name = 'Video Analysis'
        verbose_name_plural = 'Video Analyses'
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['analysis_id']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"Analysis {self.analysis_id} - {self.status}"
    
    def get_result_summary(self):
        if self.status != 'completed':
            return None
        return {
            'analysis_id': str(self.analysis_id),
            'prediction': self.prediction,
            'confidence': self.confidence,
            'probabilities': {
                'real': self.real_probability,
                'fake': self.fake_probability,
            },
            'processing_time': self.processing_time,
        }


# Signal to create UserProfile automatically
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()