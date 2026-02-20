#!/usr/bin/env python3
"""
Quick Diagnostic Script
Run: python diagnose.py
"""

import sys
import os

print("="*60)
print("  Django API - Quick Diagnostic")
print("="*60)

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

errors = []
warnings = []

# Test 1: Core Dependencies
print("\n[1/8] Checking Core Dependencies...")
try:
    import django
    print(f"  ✓ Django {django.get_version()}")
except ImportError as e:
    errors.append(f"Django not installed: {e}")
    print(f"  ✗ Django: {e}")

try:
    import rest_framework
    print(f"  ✓ Django REST Framework")
except ImportError as e:
    errors.append(f"DRF not installed: {e}")
    print(f"  ✗ DRF: {e}")

# Test 2: ML Dependencies
print("\n[2/8] Checking ML Dependencies...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    errors.append(f"PyTorch not installed: {e}")
    print(f"  ✗ PyTorch: {e}")

try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError as e:
    errors.append(f"OpenCV not installed: {e}")
    print(f"  ✗ OpenCV: {e}")

try:
    import numpy
    print(f"  ✓ NumPy {numpy.__version__}")
except ImportError as e:
    errors.append(f"NumPy not installed: {e}")
    print(f"  ✗ NumPy: {e}")

# Test 3: Optional Dependencies
print("\n[3/8] Checking Optional Dependencies...")
try:
    import mediapipe
    print(f"  ✓ MediaPipe (face detection enabled)")
except ImportError:
    warnings.append("MediaPipe not installed - face detection will use fallback")
    print(f"  ⚠ MediaPipe (face detection will use OpenCV fallback)")

try:
    import celery
    print(f"  ✓ Celery (async processing enabled)")
except ImportError:
    warnings.append("Celery not installed - will use sync processing")
    print(f"  ⚠ Celery (will use synchronous processing)")

try:
    import drf_spectacular
    print(f"  ✓ drf-spectacular (API docs enabled)")
except ImportError:
    warnings.append("drf-spectacular not installed - no API docs")
    print(f"  ⚠ drf-spectacular (API documentation disabled)")

# Test 4: Django Setup
print("\n[4/8] Checking Django Setup...")
try:
    import django
    django.setup()
    print(f"  ✓ Django setup successful")
except Exception as e:
    errors.append(f"Django setup failed: {e}")
    print(f"  ✗ Django setup: {e}")
    sys.exit(1)

# Test 5: Settings
print("\n[5/8] Checking Settings...")
try:
    from django.conf import settings
    print(f"  ✓ Settings loaded")
    print(f"    DEBUG: {settings.DEBUG}")
    print(f"    INSTALLED_APPS: {len(settings.INSTALLED_APPS)} apps")
    
    # Check video_analyzer in INSTALLED_APPS
    if 'video_analyzer' in settings.INSTALLED_APPS:
        print(f"  ✓ video_analyzer in INSTALLED_APPS")
    else:
        errors.append("video_analyzer not in INSTALLED_APPS")
        print(f"  ✗ video_analyzer not in INSTALLED_APPS")
    
    # Check REST_FRAMEWORK
    if hasattr(settings, 'REST_FRAMEWORK'):
        print(f"  ✓ REST_FRAMEWORK configured")
    else:
        warnings.append("REST_FRAMEWORK not configured")
        print(f"  ⚠ REST_FRAMEWORK not configured")
    
except Exception as e:
    errors.append(f"Settings error: {e}")
    print(f"  ✗ Settings: {e}")

# Test 6: Models
print("\n[6/8] Checking Models...")
try:
    from video_analyzer.models import APIKey, VideoAnalysis, UserProfile
    print(f"  ✓ APIKey model")
    print(f"  ✓ VideoAnalysis model")
    print(f"  ✓ UserProfile model")
except ImportError as e:
    errors.append(f"Models import error: {e}")
    print(f"  ✗ Models: {e}")

# Test 7: Utils
print("\n[7/8] Checking Utils...")
try:
    from video_analyzer import utils
    print(f"  ✓ utils.py imported")
    
    # Check key classes
    if hasattr(utils, 'VideoClassificationModel'):
        print(f"  ✓ VideoClassificationModel")
    else:
        errors.append("VideoClassificationModel not found in utils")
        print(f"  ✗ VideoClassificationModel not found")
    
    if hasattr(utils, 'VideoPreprocessor'):
        print(f"  ✓ VideoPreprocessor")
    else:
        errors.append("VideoPreprocessor not found in utils")
        print(f"  ✗ VideoPreprocessor not found")
    
    if hasattr(utils, 'VideoAnalyzer'):
        print(f"  ✓ VideoAnalyzer")
    else:
        errors.append("VideoAnalyzer not found in utils")
        print(f"  ✗ VideoAnalyzer not found")
        
except ImportError as e:
    errors.append(f"Utils import error: {e}")
    print(f"  ✗ Utils: {e}")

# Test 8: API Views
print("\n[8/8] Checking API Views...")
try:
    from video_analyzer.api import views
    print(f"  ✓ views.py imported")
    
    # Check key views
    if hasattr(views, 'HealthCheckAPIView'):
        print(f"  ✓ HealthCheckAPIView")
    
    if hasattr(views, 'VideoPredictAPIView'):
        print(f"  ✓ VideoPredictAPIView")
    
    if hasattr(views, 'VideoAnalysisStatusAPIView'):
        print(f"  ✓ VideoAnalysisStatusAPIView")
        
except ImportError as e:
    errors.append(f"Views import error: {e}")
    print(f"  ✗ Views: {e}")
    print(f"\n  Detailed error:")
    import traceback
    traceback.print_exc()

try:
    from video_analyzer.api import serializers
    print(f"  ✓ serializers.py imported")
except ImportError as e:
    errors.append(f"Serializers import error: {e}")
    print(f"  ✗ Serializers: {e}")

try:
    from video_analyzer.api import authentication
    print(f"  ✓ authentication.py imported")
except ImportError as e:
    errors.append(f"Authentication import error: {e}")
    print(f"  ✗ Authentication: {e}")

try:
    from video_analyzer.api import tasks
    print(f"  ✓ tasks.py imported")
except ImportError as e:
    warnings.append(f"Tasks import warning: {e}")
    print(f"  ⚠ Tasks: {e}")

# Test 9: Database
print("\n[9/8] Checking Database...")
try:
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
    print(f"  ✓ Database connection OK")
except Exception as e:
    warnings.append(f"Database error: {e}")
    print(f"  ⚠ Database: {e}")

# Summary
print("\n" + "="*60)
print("  DIAGNOSTIC SUMMARY")
print("="*60)

if errors:
    print(f"\n❌ {len(errors)} ERROR(S) FOUND:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
else:
    print(f"\n✅ No critical errors found!")

if warnings:
    print(f"\n⚠️  {len(warnings)} WARNING(S):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")

print("\n" + "="*60)

if errors:
    print("\n🔧 RECOMMENDED ACTIONS:")
    print("  1. Fix errors listed above")
    print("  2. Run: pip install -r requirements.txt")
    print("  3. Run: python manage.py migrate")
    print("  4. Try: python manage.py runserver --traceback")
    sys.exit(1)
else:
    print("\n✅ System is ready!")
    print("\n🚀 Next steps:")
    print("  1. Run: python manage.py migrate")
    print("  2. Run: python manage.py createsuperuser")
    print("  3. Run: python manage.py generate_test_key")
    print("  4. Run: python manage.py runserver")
    sys.exit(0)
