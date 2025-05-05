# RabbitMQ Connection Settings
RABBITMQ_HOST = 'localhost'  # or your RabbitMQ server address
RABBITMQ_PORT = 5672  # default RabbitMQ port
RABBITMQ_USER = 'guest'  # default username, change if you've set custom credentials
RABBITMQ_PASS = 'guest'  # default password, change if you've set custom credentials
RABBITMQ_VHOST = '/'  # default virtual host

# Queue Names
DOWNLOAD_QUEUE = 'download_queue'  # queue for video download tasks
TRANSCRIPTION_QUEUE = 'transcription_queue'  # queue for transcription tasks
CLIP_QUEUE = 'clip_queue'  # queue for clip extraction tasks

# Redis settings (since you're using Redis in celeryconfig.py)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# Other settings
UPLOAD_FOLDER = 'uploads'  # folder to store uploaded/downloaded videos
RESULTS_FOLDER = 'results'  # folder to store processing results
# RabbitMQ connection settings
RABBITMQ_HOST = 'localhost'  # Change this if your RabbitMQ server is on a different host

# Queue names
VIDEO_QUEUE = 'video_paths'  # Queue for receiving video file paths
TRANSCRIPTION_QUEUE = 'transcription_paths'  # Queue for sending transcription file paths

# Optional: Additional configuration settings

