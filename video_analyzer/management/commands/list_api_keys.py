# ============================================
# FILE 4: video_analyzer/management/commands/list_api_keys.py
# ============================================
from django.core.management.base import BaseCommand
from video_analyzer.models import APIKey


class Command(BaseCommand):
    help = 'List all API keys'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--username', 
            type=str, 
            help='Filter by username'
        )
        parser.add_argument(
            '--active-only',
            action='store_true',
            help='Show only active keys'
        )
    
    def handle(self, *args, **options):
        username = options.get('username')
        active_only = options.get('active_only', False)
        
        api_keys = APIKey.objects.all()
        
        if username:
            api_keys = api_keys.filter(user__username=username)
        
        if active_only:
            api_keys = api_keys.filter(is_active=True)
        
        if not api_keys.exists():
            self.stdout.write(self.style.WARNING('No API keys found'))
            return
        
        self.stdout.write(
            self.style.SUCCESS(f'Found {api_keys.count()} API key(s):\n')
        )
        
        for key in api_keys:
            status_color = self.style.SUCCESS if key.is_active else self.style.ERROR
            status_text = 'Active' if key.is_active else 'Inactive'
            
            self.stdout.write('='*60)
            self.stdout.write(f'ID:         {key.id}')
            self.stdout.write(f'User:       {key.user.username}')
            self.stdout.write(f'Name:       {key.name}')
            self.stdout.write(f'Key:        {key.key[:20]}...')
            self.stdout.write(status_color(f'Status:     {status_text}'))
            self.stdout.write(f'Usage:      {key.usage_count} requests')
            self.stdout.write(f'Rate Limit: {key.rate_limit}/hour')
            self.stdout.write(f'Created:    {key.created_at}')
            if key.last_used:
                self.stdout.write(f'Last Used:  {key.last_used}')
        
        self.stdout.write('='*60)
