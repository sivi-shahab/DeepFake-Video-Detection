# video_analyzer/api/views.py
import logging
import base64
import os
import re
import binascii
import traceback
import time
import uuid
import cv2

from django.conf import settings
from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from video_analyzer.utils import get_or_load_model
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes

from .serializers import VideoUploadSerializer, VideoAnalysisSerializer, VideoAnalysisResultSerializer
from video_analyzer.models import VideoAnalysis
from .authentication import APIKeyAuthentication
from .permissions import HasValidAPIKey

from rest_framework import status, parsers
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample, OpenApiResponse
from video_analyzer.models import APIKey, VideoAnalysis

logger = logging.getLogger(__name__)




@extend_schema(tags=['Health'], summary='Health Check', auth=[])
class HealthCheckAPIView(APIView):
    """Health check endpoint - tidak perlu authentication"""
    authentication_classes = []
    permission_classes = []
    
    def get(self, request):
        return Response({
            'status': 'healthy',
            'service': 'Video Authenticity Analyzer API',
            'timestamp': timezone.now().isoformat()
        })


# @extend_schema(
#     tags=['Prediction'],
#     summary='Predict Video Authenticity',
#     description='Upload video dan dapatkan prediksi keaslian video menggunakan model ML',
#     request=VideoUploadSerializer,
#     responses={
#         200: VideoAnalysisResultSerializer,
#         400: {'description': 'Invalid input'},
#         401: {'description': 'Authentication required'},
#         403: {'description': 'Invalid API key'},
#         500: {'description': 'Server error'}
#     },
#     parameters=[
#         OpenApiParameter(
#             name='X-API-Key',
#             type=str,
#             location=OpenApiParameter.HEADER,
#             required=True,
#             description='API Key Anda. Dapatkan dari administrator.'
#         )
#     ],
#     examples=[
#         OpenApiExample(
#             'Example Request',
#             value={
#                 'video_base64': 'data:video/mp4;base64,AAAAIGZ0eXBpc29t...',
#                 'filename': 'test_video.mp4',
#                 'sequence_length': 60,
#                 'generate_heatmap': True
#             },
#             request_only=True
#         ),
#         OpenApiExample(
#             'Example Response',
#             value={
#                 'success': True,
#                 'message': 'Video analyzed successfully',
#                 'data': {
#                     'analysis_id': '123e4567-e89b-12d3-a456-426614174000',
#                     'status': 'completed',
#                     'prediction': 'REAL',
#                     'confidence': 87.46,
#                     'probabilities': {'REAL': 0.8746, 'FAKE': 0.1254},
#                     'processing_time': 3.45,
#                     'model_used': 'VideoClassificationModel_60frames',
#                     'frames_processed': 60,
#                     'video_url': 'http://localhost:8000/media/videos/...',
#                     'heatmap_url': 'http://localhost:8000/media/heatmaps/...',
#                     'created_at': '2024-01-01T12:00:00Z',
#                     'completed_at': '2024-01-01T12:00:03Z'
#                 }
#             },
#             response_only=True
#         )
#     ]
# )

class VideoPredictAPIView(APIView):
    """Upload and predict video authenticity using ML model"""
    # authentication_classes = [APIKeyAuthentication]
    # permission_classes = [HasValidAPIKey]
    # serializer_class = VideoUploadSerializer
    # parser_classes = [JSONParser, MultiPartParser, FormParser]
    
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [IsAuthenticated]
    
    # Parser khusus untuk menangani upload file
    # parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    
    serializer_class = VideoUploadSerializer
    
    # def clean_base64_data(self, base64_string):
    #     """Clean base64 string"""
    #     if not base64_string:
    #         raise ValidationError({"video_base64": "Base64 data is empty"})
        
    #     if isinstance(base64_string, bytes):
    #         base64_string = base64_string.decode('utf-8')
        
    #     # Remove data URL prefix
    #     if base64_string.startswith('data:'):
    #         if ',' in base64_string:
    #             base64_string = base64_string.split(',', 1)[1]
        
    #     # Remove whitespace and fix padding
    #     base64_string = re.sub(r'\s+', '', base64_string)
    #     base64_string = base64_string.replace('-', '+').replace('_', '/')
        
    #     padding_needed = len(base64_string) % 4
    #     if padding_needed:
    #         base64_string += '=' * (4 - padding_needed)
        
    #     return base64_string
    
    def decode_base64(self, data):
        """
        Robust Base64 Decoder:
        1. Menghapus header data URI
        2. Menghapus whitespace/newline (\n \r) yang sering muncul di string panjang
        3. Memperbaiki padding (=) secara otomatis
        4. Mengganti URL-safe chars (-_) menjadi standar (+/)
        """
        import base64
        import binascii
        from django.core.files.base import ContentFile

        if not data:
            return None, "Data Base64 kosong"

        try:
            # Pastikan data adalah string
            if isinstance(data, bytes):
                data = data.decode('utf-8')

            # 1. Hapus Header (data:video/mp4;base64,...)
            if 'base64,' in data:
                data = data.split('base64,')[1]

            # 2. Hapus whitespace/newline (PENTING untuk string panjang)
            # String panjang sering memiliki \n atau spasi yang membuat corrupt
            data = data.translate({ord(c): None for c in ' \t\n\r'})

            # 3. Handle URL-Safe Base64 (jika dikirim dari web tertentu)
            data = data.replace('-', '+').replace('_', '/')

            # 4. Fix Padding
            # Base64 harus kelipatan 4. Jika kurang, tambah '='
            padding_needed = len(data) % 4
            if padding_needed > 0:
                data += '=' * (4 - padding_needed)

            # 5. Decode
            decoded_file = base64.b64decode(data, validate=True)

            # 6. Validasi Ukuran (Misal Max 50MB)
            if len(decoded_file) > 50 * 1024 * 1024:
                return None, "File terlalu besar (>50MB)"
            
            return ContentFile(decoded_file), None

        except (binascii.Error, ValueError) as e:
            # Tampilkan detail error di log untuk debugging
            logger.error(f"Base64 Decode Error: {str(e)}")
            # Potong string di log agar tidak memenuhi layar
            logger.error(f"Sample data received (last 50 chars): ...{data[-50:] if data else 'Empty'}")
            return None, "Format string Base64 rusak atau korup"
            
        except Exception as e:
            logger.error(f"Unexpected Base64 Error: {str(e)}")
            return None, f"Gagal memproses video: {str(e)}"
    
    # def decode_base64(self, data):
    #     """
    #     Mengubah string base64 menjadi object ContentFile Django.
    #     """
        
    #     try:
            
    #         if 'base64,' in data:
    #             data = data.split('base64,')[1]
                
    #             decoded_file = base64.b64decode(data, validate=True)
                
    #             if len(decoded_file) > 50 * 1024 * 1024:  # 50 MB limit
    #                 return None, "File terlalu besar. Maksimum 50MB."
    #             return ContentFile(decoded_file, name='uploaded_video.mp4'), None
        
    #     except (binascii.Error, ValueError):
    #         return None, "Data base64 tidak valid."
            
    @extend_schema(
        tags=['Prediction'],
        summary='Predict Video (Base64)',
        request=VideoUploadSerializer,
        responses={200: VideoAnalysisResultSerializer},
        examples=[
            OpenApiExample(
                'Valid Request',
                value={
                    'video_base64': 'data:video/mp4;base64,AAAAJGZ0eXBpc29t...', 
                    'filename': 'suspect.mp4',
                    'sequence_length': 60,
                    'generate_heatmap': True
                }
            )
        ]
    )
    
    def delete_video_file(self, analysis_obj):
        """Hapus file video fisik setelah proses selesai"""
        try:
            if analysis_obj.video_file:
                path = analysis_obj.video_file.path
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"🗑️ Video deleted: {path}")
                
                # Update DB
                analysis_obj.video_file = None
                analysis_obj.save(update_fields=['video_file'])
        except Exception as e:
            logger.error(f"Failed cleanup: {e}")
            
    # =========================================================================
    # CORE LOGIC: MENGGUNAKAN UTILS.PY ANDA
    # =========================================================================
    def _run_prediction(self, analysis, generate_heatmap):
        """
        Menjalankan inferensi menggunakan VideoClassificationModel (ResNeXt50+LSTM)
        dari utils.py
        """
        start_time = time.time()
        
        try:
            # 1. Load Model (Cache enabled via utils.py)
            # utils.py akan mencari model di settings.MODEL_PATH
            logger.info(f"🔄 Loading model for sequence: {analysis.sequence_length}")
            analyzer, device = get_or_load_model(analysis.sequence_length)
            
            # 2. Run Inference
            # predict_video() di utils mengembalikan dict result
            result = analyzer.predict_video(
                analysis.video_file.path, 
                sequence_length=analysis.sequence_length
            )

            # 3. Heatmap Generation (Optional)
            if generate_heatmap:
                logger.info("🔥 Generating heatmap...")
                # generate_heatmap() di utils mengembalikan overlay (RGB), pred_idx, conf
                overlay, _, _ = analyzer.generate_heatmap(
                    analysis.video_file.path,
                    sequence_length=analysis.sequence_length
                )
                
                # Setup path penyimpanan
                heatmap_filename = f"{analysis.analysis_id}_heatmap.png"
                # Path absolut untuk penyimpanan fisik
                heatmap_abs_path = os.path.join(settings.MEDIA_ROOT, 'heatmaps', heatmap_filename)
                os.makedirs(os.path.dirname(heatmap_abs_path), exist_ok=True)
                
                # Convert RGB (dari utils) ke BGR (untuk OpenCV save)
                # Utils Anda menggunakan cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                # jadi overlay keluarannya RGB. Kita perlu balik ke BGR untuk imwrite.
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                
                # Simpan gambar
                cv2.imwrite(heatmap_abs_path, overlay_bgr)
                
                # Simpan path relatif ke database
                analysis.heatmap_file = f"heatmaps/{heatmap_filename}"

            # 4. Save Results to Database
            analysis.status = 'completed'
            analysis.prediction = result['prediction'] # 'REAL' or 'FAKE'
            analysis.confidence = result['confidence']
            
            # Akses nested dictionary probabilities
            probs = result.get('probabilities', {})
            analysis.real_probability = probs.get('REAL', 0.0)
            analysis.fake_probability = probs.get('FAKE', 0.0)
            
            analysis.processing_time = round(time.time() - start_time, 2)
            analysis.completed_at = timezone.now()
            analysis.model_used = f"ResNeXt50_LSTM_{analysis.sequence_length}"
            
            analysis.save()
            logger.info(f"✅ Analysis {analysis.analysis_id} completed: {analysis.prediction}")

        except Exception as e:
            logger.error(f"❌ ML Prediction Error: {str(e)}")
            # Tandai status failed di DB agar tidak stuck di 'processing'
            analysis.status = 'failed'
            analysis.save()
            # Re-raise agar ditangkap oleh blok try-except di method post()
            raise e

    def post(self, request):
        logger.info(f"🎬 POST /predict (Base64) - User: {request.user}")

        # 1. Validasi Schema JSON
        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            return Response({'success': False, 'error': serializer.errors}, status=400)

        validated_data = serializer.validated_data
        
        # 2. Decode Base64 (CPU Intensive)
        file_content, error_msg = self.decode_base64(validated_data['video_base64'])
        
        if error_msg:
            return Response({'success': False, 'error': error_msg}, status=400)
        
        analysis = None

        try:
            # 3. Buat Record Database
            # Generate nama file unik
            ext = os.path.splitext(validated_data['filename'])[1] or '.mp4'
            unique_filename = f"{uuid.uuid4()}{ext}"
            current_api_key = None
            if request.auth and isinstance(request.auth, APIKey):
                current_api_key = request.auth

            analysis = VideoAnalysis.objects.create(
                user=request.user,
                api_key=current_api_key,
                original_filename=validated_data['filename'],
                sequence_length=validated_data['sequence_length'],
                status='processing',
                video_size=file_content.size
            )

            # 4. Simpan File Fisik
            # 'file_content' adalah ContentFile yang dibuat dari hasil decode memory
            analysis.video_file.save(unique_filename, file_content)

            # 5. Jalankan Prediksi ML
            # (Fungsi ini sama seperti sebelumnya)
            self._run_prediction(analysis, validated_data['generate_heatmap'])
            
            # Refresh untuk mengambil hasil update dari _run_prediction
            analysis.refresh_from_db()

            return Response({
                'success': True,
                'message': 'Analysis completed',
                'data': VideoAnalysisResultSerializer(analysis).data
            })

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            if analysis:
                analysis.status = 'failed'
                analysis.save()
            return Response({
                'success': False, 
                'error': f'Internal processing error: {str(e)}'
            }, status=500)
            
        finally:
            if analysis:
                self.delete_video_file(analysis)

    # def _run_prediction(self, analysis, generate_heatmap):
    #     """
    #     Logika ML dipisahkan agar rapi.
    #     """
    #     from video_analyzer.utils import get_or_load_model
    #     import cv2 

    #     start_time = time.time()
        
    #     try:
    #         # Load Model    
    #         analyzer, device = get_or_load_model(analysis.sequence_length)
            
    #         # Run Inference (Path file diambil dari field FileField)
    #         result = analyzer.predict_video(
    #             analysis.video_file.path, 
    #             sequence_length=analysis.sequence_length
    #         )

    #         # Heatmap Logic
    #         if generate_heatmap:
    #             overlay, _, _ = analyzer.generate_heatmap(
    #                 analysis.video_file.path,
    #                 sequence_length=analysis.sequence_length
    #             )
                
    #             # Simpan Heatmap
    #             heatmap_name = f"{analysis.analysis_id}_heatmap.png"
    #             heatmap_path = os.path.join(settings.MEDIA_ROOT, 'heatmaps', heatmap_name)
    #             os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                
    #             cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    #             analysis.heatmap_file = f"heatmaps/{heatmap_name}"

    #         # Save Result
    #         analysis.status = 'completed'
    #         analysis.prediction = result['prediction']
    #         analysis.confidence = result['confidence']
    #         analysis.real_probability = result['probabilities']['REAL']
    #         analysis.fake_probability = result['probabilities']['FAKE']
    #         analysis.processing_time = round(time.time() - start_time, 2)
    #         analysis.completed_at = timezone.now()
    #         analysis.save()
            
    #     except Exception as e:
    #         logger.error(f"ML Error: {e}")
    #         analysis.status = 'failed'
    #         analysis.save()
    #         raise e # Lempar ke blok try-except utama


@extend_schema(
    tags=['Analysis'],
    summary='Get Analysis Status',
    description='Check status and get results of a video analysis',
    responses={
        200: VideoAnalysisResultSerializer,
        401: {'description': 'Authentication required'},
        403: {'description': 'Invalid API key or access denied'},
        404: {'description': 'Analysis not found'}
    },
    parameters=[
        OpenApiParameter(
            name='X-API-Key',
            type=str,
            location=OpenApiParameter.HEADER,
            required=True,
            description='API Key Anda. Dapatkan dari administrator.'
        ),
        OpenApiParameter(
            name='analysis_id',
            type=OpenApiTypes.UUID,
            location=OpenApiParameter.PATH,
            description='ID analisis yang ingin dicek'
        )
    ]
)
class VideoAnalysisStatusAPIView(APIView):
    """Check analysis status"""
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [HasValidAPIKey]
    
    def get(self, request, analysis_id):
        try:
            analysis = VideoAnalysis.objects.get(analysis_id=analysis_id)
            
            # Check ownership
            if request.auth and analysis.api_key != request.auth.key:
                return Response({
                    'success': False,
                    'error': 'Access denied'
                }, status=status.HTTP_403_FORBIDDEN)
            
            serializer = VideoAnalysisResultSerializer(
                analysis,
                context={'request': request}
            )
            
            return Response({
                'success': True,
                'data': serializer.data
            })
            
        except VideoAnalysis.DoesNotExist:
            return Response({
                'success': False,
                'error': 'Analysis not found'
            }, status=status.HTTP_404_NOT_FOUND)


@extend_schema_view(
    list=extend_schema(
        summary='List Video Analyses',
        description='Get list of all your video analyses',
        parameters=[
            OpenApiParameter(
                name='X-API-Key',
                type=str,
                location=OpenApiParameter.HEADER,
                required=True,
                description='API Key Anda'
            )
        ]
    ),
    retrieve=extend_schema(
        summary='Get Analysis Details',
        description='Get detailed information about a specific analysis',
        parameters=[
            OpenApiParameter(
                name='X-API-Key',
                type=str,
                location=OpenApiParameter.HEADER,
                required=True,
                description='API Key Anda'
            ),
            OpenApiParameter(
                name='analysis_id',
                type=OpenApiTypes.UUID,
                location=OpenApiParameter.PATH,
                description='ID analisis'
            )
        ]
    )
)
class VideoAnalysisViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet untuk melihat daftar dan detail analisis video
    Hanya boleh melihat analisis sendiri berdasarkan API Key
    """
    authentication_classes = [APIKeyAuthentication]
    permission_classes = [HasValidAPIKey]
    serializer_class = VideoAnalysisSerializer
    lookup_field = 'analysis_id'
    
    def get_queryset(self):
        """
        Hanya return analisis yang dibuat dengan API Key yang sama
        """
        if self.request.auth:
            return VideoAnalysis.objects.filter(
                api_key=self.request.auth.key
            ).order_by('-created_at')
        return VideoAnalysis.objects.none()
    
    def retrieve(self, request, *args, **kwargs):
        """
        Get detail analisis berdasarkan analysis_id
        """
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response({
            'success': True,
            'data': serializer.data
        })
    
    def list(self, request, *args, **kwargs):
        """
        Get list analisis dengan pagination
        """
        queryset = self.filter_queryset(self.get_queryset())
        
        # Pagination
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response({
                'success': True,
                'data': serializer.data
            })
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'data': serializer.data,
            'count': len(serializer.data)
        })