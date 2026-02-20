# ============================================
# FILE 3: video_analyzer/management/commands/create_api_key.py
# ============================================
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from video_analyzer.models import APIKey


class Command(BaseCommand):
    help = 'Create API key for a user'
    
    def add_arguments(self, parser):
        parser.add_argument('username', type=str, help='Username')
        parser.add_argument(
            '--name', 
            type=str, 
            default='Default API Key', 
            help='API Key name'
        )
        parser.add_argument(
            '--rate-limit', 
            type=int, 
            default=100, 
            help='Rate limit per hour'
        )
    
    def handle(self, *args, **options):
        username = options['username']
        key_name = options['name']
        rate_limit = options['rate_limit']
        
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'User "{username}" does not exist')
            )
            self.stdout.write(
                self.style.WARNING('Create user first with: python manage.py createsuperuser')
            )
            return
        
        api_key = APIKey.objects.create(
            user=user,
            name=key_name,
            rate_limit=rate_limit
        )
        
        self.stdout.write(self.style.SUCCESS('✓ API Key created successfully!'))
        self.stdout.write('')
        self.stdout.write('='*60)
        self.stdout.write(f'User:       {user.username}')
        self.stdout.write(f'Name:       {api_key.name}')
        self.stdout.write(f'API Key:    {api_key.key}')
        self.stdout.write(f'Rate Limit: {api_key.rate_limit} requests/hour')
        self.stdout.write(f'Status: Active')
        self.stdout.write('='*60)
        self.stdout.write('')
        self.stdout.write(
            self.style.WARNING('⚠ IMPORTANT: Save this key securely!')
        )
        self.stdout.write(
            self.style.WARNING('   It will not be shown again.')
        )
        self.stdout.write('')
        self.stdout.write('Test the API with:')
        self.stdout.write(
            f'  curl -H "X-API-Key: {api_key.key}" '
            'http://localhost:8000/api/v1/health/'
        )
