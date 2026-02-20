# video_analyzer/api/tasks.py
from celery import shared_task
from django.conf import settings
from django.utils import timezone
import time
import os
import logging
import traceback

logger = logging.getLogger(__name__)

# Django setup for Celery
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'video_authenticity_api.settings')
django.setup()


@shared_task(bind=True, max_retries=3, name="video_analyzer.process_video_analysis")
def process_video_analysis_task(self, analysis_id, generate_heatmap=False):
    """
    Celery task for async video processing
    """
    logger.info(f"🚀 Starting Celery task for analysis {analysis_id}")
    
    try:
        from video_analyzer.models import VideoAnalysis
        
        # Load analysis record
        analysis = VideoAnalysis.objects.get(id=analysis_id)
        analysis.status = 'processing'
        analysis.save()
        
        # Process video synchronously
        result = process_video_analysis_sync(analysis_id, generate_heatmap)
        
        logger.info(f"✅ Task completed: {result.get('prediction')}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Task failed: {e}")
        
        # Update analysis status
        try:
            analysis = VideoAnalysis.objects.get(id=analysis_id)
            analysis.status = 'failed'
            analysis.error_message = str(e)[:500]
            analysis.save()
        except Exception:
            pass
        
        # Retry logic
        if self.request.retries < self.max_retries:
            retry_count = self.request.retries + 1
            logger.info(f"🔄 Retry {retry_count}/{self.max_retries}")
            raise self.retry(exc=e, countdown=60 * retry_count)
        else:
            raise


def process_video_analysis_sync(analysis_id, generate_heatmap=False):
    """
    Synchronous video processing using VideoClassificationModel
    """
    from video_analyzer.models import VideoAnalysis
    from video_analyzer.utils import get_or_load_model
    
    logger.info(f"🎬 Processing video analysis {analysis_id}")
    
    try:
        # Load analysis
        analysis = VideoAnalysis.objects.get(id=analysis_id)
        start_time = time.time()
        video_path = analysis.video_file.path
        
        # Check if video exists
        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")
        
        # Load ML model
        logger.info(f"Loading model for sequence_length={analysis.sequence_length}")
        analyzer, device = get_or_load_model(analysis.sequence_length)
        
        # Make prediction
        result = analyzer.predict_video(
            video_path,
            sequence_length=analysis.sequence_length
        )
        
        # Generate heatmap if requested
        if generate_heatmap:
            try:
                import cv2
                
                overlay, _, _ = analyzer.generate_heatmap(
                    video_path,
                    sequence_length=analysis.sequence_length
                )
                
                heatmap_filename = f'{analysis.analysis_id}_heatmap.png'
                heatmap_path = os.path.join(settings.MEDIA_ROOT, 'heatmaps', heatmap_filename)
                os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(heatmap_path, overlay_bgr)
                
                analysis.heatmap_file.name = f'heatmaps/{heatmap_filename}'
                logger.info("✓ Heatmap generated")
            except Exception as e:
                logger.warning(f"Heatmap failed: {e}")
        
        # Save results
        processing_time = round(time.time() - start_time, 2)
        
        analysis.status = 'completed'
        analysis.prediction = result['prediction']
        analysis.confidence = result['confidence']
        analysis.real_probability = result['probabilities']['REAL']
        analysis.fake_probability = result['probabilities']['FAKE']
        analysis.processing_time = processing_time
        analysis.model_used = f'VideoClassificationModel_{analysis.sequence_length}frames'
        analysis.frames_processed = analysis.sequence_length
        analysis.completed_at = timezone.now()
        analysis.save()
        
        logger.info(f"✅ Completed in {processing_time}s")
        
        return {
            'success': True,
            'analysis_id': str(analysis.analysis_id),
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'processing_time': processing_time
        }
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        
        # Update analysis as failed
        try:
            analysis = VideoAnalysis.objects.get(id=analysis_id)
            analysis.status = 'failed'
            analysis.error_message = str(e)[:500]
            analysis.completed_at = timezone.now()
            analysis.save()
        except Exception as save_error:
            logger.error(f"Save error: {save_error}")
        
        raise