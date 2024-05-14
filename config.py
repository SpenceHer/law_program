from google.oauth2 import service_account
from google.cloud import vision
import os

def get_vision_client():
    # Get the path to the credentials JSON file from the environment variable
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
    
    # Use the environment variable to load credentials
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    return client




