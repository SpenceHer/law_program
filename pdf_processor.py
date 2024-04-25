
from pdf2image import convert_from_path
import io
import config
from google.cloud import vision

class PDFProcessor:
    def __init__(self, pdf_path, credentials_path):
        self.pdf_path = pdf_path
        self.client = config.get_vision_client(credentials_path)

    def convert_pdf_to_images(self):
        return convert_from_path(self.pdf_path)

    def extract_text(self, image):
        """ Extracts text from an image using Google Vision API. """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Google Vision prefers PNG
        img_byte_arr = img_byte_arr.getvalue()

        vision_image = vision.Image(content=img_byte_arr)
        response = self.client.text_detection(image=vision_image)

        texts = response.text_annotations

        return texts[0].description if texts else ''
