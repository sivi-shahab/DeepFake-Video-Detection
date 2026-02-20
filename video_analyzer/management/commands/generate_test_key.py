# ============================================
# FILE 6: video_analyzer/management/commands/generate_test_key.py
# ============================================
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from video_analyzer.models import APIKey


class Command(BaseCommand):
    help = 'Generate test API key with test user'
    
    def handle(self, *args, **options):
        # Create or get test user
        username = 'testuser'
        email = 'test@example.com'
        password = 'testpass123'
        
        user, created = User.objects.get_or_create(
            username=username,
            defaults={
                'email': email,
                'is_staff': False,
                'is_superuser': False
            }
        )
        
        if created:
            user.set_password(password)
            user.save()
            self.stdout.write(
                self.style.SUCCESS(f'✓ Test user created: {username}')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'⚠ Test user already exists: {username}')
            )
        
        # Create API key
        api_key = APIKey.objects.create(
            user=user,
            name='Test API Key',
            rate_limit=1000
        )
        
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('✓ Test API key created!'))
        self.stdout.write('')
        self.stdout.write('='*60)
        self.stdout.write('TEST CREDENTIALS:')
        self.stdout.write('='*60)
        self.stdout.write(f'Username:   {username}')
        self.stdout.write(f'Password:   {password}')
        self.stdout.write(f'API Key:    {api_key.key}')
        self.stdout.write('='*60)
        self.stdout.write('')
        self.stdout.write('Save this API key for testing!')
        self.stdout.write('')
        self.stdout.write('Quick test:')
        self.stdout.write(
            f'  curl -H "X-API-Key: {api_key.key}" '
            'http://localhost:8000/api/v1/health/'
        )