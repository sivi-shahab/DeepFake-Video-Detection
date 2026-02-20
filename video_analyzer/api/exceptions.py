# ============================================
# FILE 3: video_analyzer/api/exceptions.py
# ============================================
from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status
import logging

logger = logging.getLogger(__name__)

def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)
    
    if response is not None:
        custom_response_data = {
            'success': False,
            'error': 'An error occurred',
            'details': None
        }
        
        if isinstance(response.data, dict):
            if 'detail' in response.data:
                custom_response_data['error'] = response.data['detail']
            else:
                custom_response_data['details'] = response.data
        else:
            custom_response_data['error'] = str(response.data)
        
        custom_response_data['status_code'] = response.status_code
        response.data = custom_response_data
        logger.error(f"API Error: {custom_response_data['error']}", exc_info=exc)
    else:
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=exc)
        response = Response({
            'success': False,
            'error': 'Internal server error',
            'details': str(exc),
            'status_code': status.HTTP_500_INTERNAL_SERVER_ERROR
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return response
