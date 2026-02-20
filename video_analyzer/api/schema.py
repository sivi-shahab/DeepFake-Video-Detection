from drf_spectacular.extensions import OpenApiAuthenticationExtension

class APIKeyScheme(OpenApiAuthenticationExtension):
    """
    Menghubungkan Custom Authentication Class kita dengan Swagger UI.
    Otomatis aktif jika view menggunakan APIKeyAuthentication.
    """
    
    target_class = 'video_analyzer.api.authentication.APIKeyAuthentication'
    name = 'ApiKeyAuth'
    
    def get_security_definition(self, auto_schema):
        return {
            'type': 'apiKey',
            'in': 'header',
            'name': 'X-API-Key', # Header yang sebenarnya dikirim
            'description': 'Masukkan API Key Anda (Format: sk_ml_...)'
        }