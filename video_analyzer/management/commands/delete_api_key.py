# ============================================
# FILE 5: video_analyzer/management/commands/delete_api_key.py
# ============================================
from django.core.management.base import BaseCommand
from video_analyzer.models import APIKey


class Command(BaseCommand):
    help = 'Delete an API key'
    
    def add_arguments(self, parser):
        parser.add_argument('key_id', type=str, help='API Key ID (UUID)')
    
    def handle(self, *args, **options):
        key_id = options['key_id']
        
        try:
            api_key = APIKey.objects.get(id=key_id)
            
            self.stdout.write('API Key Details:')
            self.stdout.write(f'  User: {api_key.user.username}')
            self.stdout.write(f'  Name: {api_key.name}')
            self.stdout.write(f'  Key:  {api_key.key[:20]}...')
            self.stdout.write('')
            
            confirm = input('Are you sure you want to delete this key? (yes/no): ')
            
            if confirm.lower() == 'yes':
                api_key.delete()
                self.stdout.write(
                    self.style.SUCCESS('✓ API key deleted successfully')
                )
            else:
                self.stdout.write(
                    self.style.WARNING('✗ Deletion cancelled')
                )
                
        except APIKey.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'API key with ID "{key_id}" not found')
            )