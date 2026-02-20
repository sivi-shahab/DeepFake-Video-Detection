# video_analyzer/api/serializers.py
from rest_framework import serializers
from video_analyzer.models import VideoAnalysis, APIKey
from drf_spectacular.utils import extend_schema_field
from drf_spectacular.types import OpenApiTypes
import base64
import re
from django.conf import settings


class VideoUploadSerializer(serializers.Serializer):
    """Serializer untuk upload video via base64"""
    video_base64 = serializers.CharField(
        required=True,
        help_text="Base64 encoded video data. Remove data URI prefix if present.",
        style={'base_template': 'textarea.html'}
    )
    filename = serializers.CharField(
        required=False,
        default='uploaded_video.mp4',
        max_length=255,
        help_text="Original filename (e.g., 'video.mp4')"
    )
    sequence_length = serializers.IntegerField(
        required=False,
        default=60,
        min_value=10,
        max_value=120,
        help_text="Number of frames to process. Higher = more accurate but slower. Options: 30, 60, 90"
    )
    generate_heatmap = serializers.BooleanField(
        required=False,
        default=False,
        help_text="Generate Grad-CAM attention heatmap visualization"
    )
    
    # video_base64 = serializers.CharField(required=True, write_only=True)
    # filename = serializers.CharField(required=False, default='uploaded_video.mp4')
    # sequence_length = serializers.IntegerField(required=False, default=60, min_value=10, max_value=300)
    # generate_heatmap = serializers.BooleanField(required=False, default=False)
    
    
    def validate_video_base64(self, value):
        """Validate base64 string"""
        try:
            if not value or len(value.strip()) == 0:
                raise serializers.ValidationError("Base64 data cannot be empty")
        
        # Basic cleaning before validation
            cleaned_value = value.strip()
        
        # Remove data URI prefix if present
            if cleaned_value.startswith('data:'):
                if ',' in cleaned_value:
                    cleaned_value = cleaned_value.split(',')[1]
        
            # Remove whitespace
            cleaned_value = re.sub(r'\s+', '', cleaned_value)
        
            # Handle URL-safe base64
            cleaned_value = cleaned_value.replace('-', '+').replace('_', '/')
        
        # Add padding if needed
            padding_needed = len(cleaned_value) % 4
            if padding_needed:
                cleaned_value += '=' * (4 - padding_needed)
        
        # Validate base64
            decoded = base64.b64decode(cleaned_value, validate=True)
        
        # Check size
            
            max_size = getattr(settings, 'MAX_VIDEO_SIZE', 100 * 1024 * 1024)  # 100MB default
            min_size = getattr(settings, 'MIN_VIDEO_SIZE', 1024)  # 1KB minimum
        
            if len(decoded) < min_size:
                raise serializers.ValidationError(
                    f"Video file is too small (minimum {min_size/1024:.0f}KB)"
                )
        
            if len(decoded) > max_size:
                raise serializers.ValidationError(
                    f"Video size exceeds maximum allowed size ({max_size / 1024 / 1024:.0f}MB)"
                )
        
            return value  # Return original value, not cleaned value
        
        except (base64.binascii.Error, UnicodeDecodeError) as e:
            raise serializers.ValidationError(f"Invalid base64 data: {str(e)}")
        except Exception as e:
            raise serializers.ValidationError(f"Failed to validate video data: {str(e)}")
    
    def validate_filename(self, value):
        """Validate filename"""
        # Remove path components
        value = value.split('/')[-1].split('\\')[-1]
        
        # Check extension
        allowed_formats = ['mp4', 'avi', 'mov', 'mkv']
        ext = value.split('.')[-1].lower()
        if ext not in allowed_formats:
            raise serializers.ValidationError(
                f"Invalid file format. Allowed: {', '.join(allowed_formats)}"
            )
        
        return value


class VideoAnalysisSerializer(serializers.ModelSerializer):
    """Serializer untuk VideoAnalysis model"""
    analysis_id = serializers.UUIDField(read_only=True)
    video_url = serializers.SerializerMethodField()
    heatmap_url = serializers.SerializerMethodField()
    
    class Meta:
        model = VideoAnalysis
        fields = [
            'analysis_id',
            'original_filename',
            'video_size',
            'video_duration',
            'sequence_length',
            'status',
            'prediction',
            'confidence',
            'real_probability',
            'fake_probability',
            'processing_time',
            'model_used',
            'frames_processed',
            'error_message',
            'video_url',
            'heatmap_url',
            'created_at',
            'updated_at',
            'completed_at',
        ]
        read_only_fields = [
            'analysis_id',
            'status',
            'prediction',
            'confidence',
            'real_probability',
            'fake_probability',
            'processing_time',
            'model_used',
            'frames_processed',
            'error_message',
            'created_at',
            'updated_at',
            'completed_at',
        ]
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_video_url(self, obj) -> str:
        """Get absolute URL for video file"""
        if obj.video_file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.video_file.url)
        return None
    
    @extend_schema_field(OpenApiTypes.STR)
    def get_heatmap_url(self, obj) -> str:
        """Get absolute URL for heatmap file"""
        if obj.heatmap_file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.heatmap_file.url)
        return None


class VideoAnalysisResultSerializer(serializers.ModelSerializer):
    """Simplified serializer untuk hasil prediksi"""
    analysis_id = serializers.UUIDField(read_only=True)
    probabilities = serializers.SerializerMethodField()
    
    class Meta:
        model = VideoAnalysis
        fields = [
            'analysis_id',
            'status',
            'prediction',
            'confidence',
            'probabilities',
            'processing_time',
            'model_used',
            'frames_processed',
            'created_at',
            'completed_at',
        ]
    
    @extend_schema_field({
        'type': 'object',
        'properties': {
            'real': {'type': 'number', 'format': 'float', 'example': 0.1246},
            'fake': {'type': 'number', 'format': 'float', 'example': 0.8754},
        },
        'nullable': True
    })
    def get_probabilities(self, obj):
        """Get probability scores for each class"""
        if obj.real_probability is not None and obj.fake_probability is not None:
            return {
                'real': round(obj.real_probability, 4),
                'fake': round(obj.fake_probability, 4),
            }
        return None


class APIKeySerializer(serializers.ModelSerializer):
    """Serializer untuk API Key management"""
    key = serializers.CharField(read_only=True)
    
    class Meta:
        model = APIKey
        fields = [
            'id',
            'name',
            'key',
            'is_active',
            'created_at',
            'last_used',
            'usage_count',
            'rate_limit',
        ]
        read_only_fields = ['id', 'key', 'created_at', 'last_used', 'usage_count']