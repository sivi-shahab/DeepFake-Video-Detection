# ============================================
# FILE 4: video_analyzer/admin.py
# ============================================
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from .models import APIKey, VideoAnalysis, UserProfile


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ['id', 'user_link', 'name', 'masked_key', 'is_active', 
                   'usage_count', 'rate_limit', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['user__username', 'name', 'key']
    readonly_fields = ['key', 'created_at', 'last_used', 'usage_count']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'name', 'key')
        }),
        ('Settings', {
            'fields': ('is_active', 'rate_limit')
        }),
        ('Statistics', {
            'fields': ('usage_count', 'created_at', 'last_used')
        }),
    )
    
    def user_link(self, obj):
        url = reverse('admin:auth_user_change', args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.username)
    user_link.short_description = 'User'
    
    def masked_key(self, obj):
        return f"{obj.key[:10]}...{obj.key[-10:]}"
    masked_key.short_description = 'API Key'
    
    actions = ['activate_keys', 'deactivate_keys', 'reset_usage']
    
    def activate_keys(self, request, queryset):
        count = queryset.update(is_active=True)
        self.message_user(request, f'{count} API key(s) activated.')
    
    def deactivate_keys(self, request, queryset):
        count = queryset.update(is_active=False)
        self.message_user(request, f'{count} API key(s) deactivated.')
    
    def reset_usage(self, request, queryset):
        count = queryset.update(usage_count=0)
        self.message_user(request, f'Usage reset for {count} key(s).')


@admin.register(VideoAnalysis)
class VideoAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        'analysis_id_short',
        'user_link',
        'original_filename',
        'status_badge',
        'prediction_badge',
        'confidence',
        'processing_time',
        'created_at'
    ]
    list_filter = ['status', 'prediction', 'created_at']
    search_fields = ['analysis_id', 'user__username', 'original_filename']
    readonly_fields = [
        'analysis_id', 'created_at', 'updated_at', 'completed_at',
        'video_preview', 'heatmap_preview'
    ]
    
    fieldsets = (
        ('Analysis Info', {
            'fields': ('analysis_id', 'user', 'api_key', 'status', 'error_message')
        }),
        ('Video Info', {
            'fields': ('original_filename', 'video_file', 'video_preview', 
                      'video_size', 'video_duration')
        }),
        ('Settings', {
            'fields': ('sequence_length',)
        }),
        ('Results', {
            'fields': ('prediction', 'confidence', 'real_probability', 
                      'fake_probability', 'processing_time', 'model_used', 
                      'frames_processed')
        }),
        ('Heatmap', {
            'fields': ('heatmap_file', 'heatmap_preview')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'completed_at')
        }),
    )
    
    def analysis_id_short(self, obj):
        return str(obj.analysis_id)[:8]
    analysis_id_short.short_description = 'ID'
    
    def user_link(self, obj):
        if obj.user:
            url = reverse('admin:auth_user_change', args=[obj.user.id])
            return format_html('<a href="{}">{}</a>', url, obj.user.username)
        elif obj.api_key:
            return format_html('<span style="color: #999;">API: {}</span>', 
                             obj.api_key.name)
        return '-'
    user_link.short_description = 'User'
    
    def status_badge(self, obj):
        colors = {
            'pending': '#ffc107',
            'processing': '#17a2b8',
            'completed': '#28a745',
            'failed': '#dc3545'
        }
        color = colors.get(obj.status, '#6c757d')
        return format_html(
            '<span style="background: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px;">{}</span>',
            color, obj.status.upper()
        )
    status_badge.short_description = 'Status'
    
    def prediction_badge(self, obj):
        if obj.prediction == 'REAL':
            color = '#28a745'
        elif obj.prediction == 'FAKE':
            color = '#dc3545'
        else:
            return '-'
        return format_html(
            '<span style="background: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px; font-weight: bold;">{}</span>',
            color, obj.prediction
        )
    prediction_badge.short_description = 'Prediction'
    
    def video_preview(self, obj):
        if obj.video_file:
            return format_html(
                '<video width="320" height="240" controls>'
                '<source src="{}" type="video/mp4"></video>',
                obj.video_file.url
            )
        return '-'
    video_preview.short_description = 'Video'
    
    def heatmap_preview(self, obj):
        if obj.heatmap_file:
            return format_html(
                '<img src="{}" style="max-width: 400px;" />',
                obj.heatmap_file.url
            )
        return '-'
    heatmap_preview.short_description = 'Heatmap'
    
    actions = ['reprocess_videos']
    
    def reprocess_videos(self, request, queryset):
        from .api.tasks import process_video_analysis_task
        count = 0
        for analysis in queryset.filter(status__in=['failed', 'pending']):
            analysis.status = 'pending'
            analysis.error_message = None
            analysis.save()
            try:
                process_video_analysis_task.delay(analysis.id)
            except:
                pass
            count += 1
        self.message_user(request, f'{count} video(s) queued for reprocessing.')


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'phone', 'created_at']
    search_fields = ['user__username', 'phone']


# Customize Admin Site
admin.site.site_header = "Video Authenticity Analyzer Admin"
admin.site.site_title = "Video Analyzer Admin"
admin.site.index_title = "Welcome to Video Analyzer Administration"