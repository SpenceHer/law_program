from google.oauth2 import service_account
from google.cloud import vision

def get_vision_client(credentials_path):
    """ Sets up and returns a Google Vision API client. """
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    return client


