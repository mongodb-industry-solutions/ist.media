import os

openai_api_key = os.environ.get('OPENAI_API_KEY', '')
MONGO_URI      = os.environ.get('MONGODB_IST_MEDIA', '')
NEWS_API_URL   = os.environ.get('NEWS_API_URL', '')
NEWS_API_KEY   = os.environ.get('NEWS_API_KEY', '')

# for clonedb.py script only
MONGO_URI_NEW  = os.environ.get('MONGODB_IST_MEDIA_NEW', '')
