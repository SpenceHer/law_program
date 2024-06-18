from google.oauth2 import service_account
from google.cloud import vision
import os

def get_vision_client():
    credentials_path = os.environ.get('GOOGLE_CREDENTIALS_PATH')
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
    
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    return client
