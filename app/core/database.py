from pymongo import MongoClient
from app.core.config import settings

mongo_client = MongoClient(settings.mongodb_url)
db = mongo_client[settings.mongodb_db_name]

# Optional: health check or connection validation
def get_db():
    return db