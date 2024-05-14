from IPython import embed
from google.oauth2 import service_account
from google.cloud import vision
from pdf2image import convert_from_path

from pdf2image import convert_from_path

import io
import config
from google.cloud import vision
import re
import concurrent.futures
import os
import fitz  # PyMuPDF for PDF handling
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2
from PIL import ImageStat, Image
import pandas as pd
from PIL import ImageDraw
import spacy
from collections import Counter


def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    # Process the text
    doc = nlp(text)
    # Extract entities
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def count_shared_entities(text1, text2):

    # Extract entities from both texts
    entities1 = extract_entities(text1)
    entities2 = extract_entities(text2)
    
    # Count occurrences of each entity
    count1 = Counter(entities1)
    count2 = Counter(entities2)
    
    # Find the shared entities and sum the minimum counts
    shared_entities = set(count1.keys()) & set(count2.keys())
    shared_count = sum(min(count1[entity], count2[entity]) for entity in shared_entities)
    
    # Total shared entities
    total_shared_entities = sum((count1[entity] + count2[entity]) for entity in shared_entities)
    
    return total_shared_entities

def get_vision_client():
    # Get the path to the credentials JSON file from the environment variable
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
    
    # Use the environment variable to load credentials
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    return client

def calculate_average_text_size(bounding_boxes):
    sizes = [get_text_size(box) for box in bounding_boxes if box]
    if sizes:
        return np.mean(sizes)
    else:
        return 0

def get_text_size(bounding_box):
    # This will calculate the average height of the text based on the bounding box
    box_height = abs(bounding_box.vertices[0].y - bounding_box.vertices[2].y)
    return box_height


def detect_lines(bounding_boxes, image):
    # Simplistic line detection based on bounding boxes; real implementation may vary
    draw = ImageDraw.Draw(image)
    lines = []
    for box in bounding_boxes:
        if is_line(box):
            lines.append(box)
            draw.line([box.vertices[0].x, box.vertices[0].y, box.vertices[1].x, box.vertices[1].y], fill=128)

    return len(lines)

def is_line(box):
    # Define logic to detect if a bounding box is a line based on its dimensions
    width = abs(box.vertices[1].x - box.vertices[0].x)
    height = abs(box.vertices[2].y - box.vertices[0].y)
    return width / height > 10  # Example ratio that might represent a line




embed()


df = pd.read_csv("/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/sorted_page_df.csv")


client = get_vision_client()

# new_file_path = filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")], initialdir="/")
# new_file_path = "/Users/spencersmith/Desktop/CODING/Projects/law/PDFs/small/Binder2.pdf" # SMALL FULL
new_file_path = "/Users/spencersmith/Desktop/CODING/Projects/law/PDFs/small/separated/WHITECAP0001000.pdf" # SMALL SINGLE




with fitz.open(new_file_path) as doc:
    total_pages = len(doc)
    print(f"Total Pages: {total_pages}")

    page_number = 0
    page = doc.load_page(page_number)
    image = page.get_pixmap()

    # Converting the pixmap directly to bytes in PNG format
    img_bytes = image.tobytes("png")
    img_byte_arr = io.BytesIO(img_bytes)  # Now this contains the PNG data
    content = img_byte_arr.getvalue()

    # Prepare the image for Google Vision
    vision_image = vision.Image(content=content)
    # response = client.text_detection(image=vision_image)
    response = client.document_text_detection(image=vision_image)

    texts = response.text_annotations

    full_text = texts[0].description if texts else ''
    rows = full_text.split("\n")
    words = [text.description for text in texts[1:]]
    filtered_words = [word for word in words if re.search(r'[a-zA-Z0-9]', word)]


entities = extract_entities(full_text)




# Convert PDF pages to images
print("Converting PDF pages to images...")
images =convert_from_path(new_file_path)




page_data_list = []


page_num = 2
image = images[page_num]

for page_num, image in enumerate(images):

    print(f"Extracting text from page {page_num} of {len(images)}")

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # Google Vision prefers PNG
    img_byte_arr = img_byte_arr.getvalue()

    vision_image = vision.Image(content=img_byte_arr)
    response = client.text_detection(image=vision_image)

    texts = response.text_annotations
    full_text_annotation = response.full_text_annotation


    full_text = texts[0].description if texts else ''
    rows = full_text.split("\n")
    words = [text.description for text in texts[1:]]
    filtered_words = [word for word in words if re.search(r'[a-zA-Z0-9]', word)]


    bounding_boxes = [text.bounding_poly for text in response.text_annotations[1:]]
    average_text_size = calculate_average_text_size(bounding_boxes)


    line_presence = detect_lines(bounding_boxes, image)

    page_data = {
        "page_num": page_num
    }
    page_data.update({
        "full_text": full_text,
        "text_rows": rows,
        "text_words": words,
        "filtered_words": filtered_words,
        "word_count": len(filtered_words),
        "rows_of_text_count": len(rows)
    })
    page_data_list.append(page_data)


page_df = pd.DataFrame(page_data_list)





for text in texts:
    print(text.properties)



# Iterate through page level information
for page in full_text_annotation.pages:
    for block in page.blocks:
        print("\nBlock confidence: ", block.confidence)

        for paragraph in block.paragraphs:
            # Corrected list comprehension for constructing paragraph text
            paragraph_text = ''.join([symbol.text for word in paragraph.words for symbol in word.symbols])
            print("Paragraph text:", paragraph_text)
            print("Paragraph confidence: ", paragraph.confidence)

            for word in paragraph.words:
                # Corrected list comprehension for constructing word text
                word_text = ''.join([symbol.text for symbol in word.symbols])
                print("Word text:", word_text)
                print("Word confidence:", word.confidence)

                for symbol in word.symbols:
                    print("Symbol text:", symbol.text)
                    print("Symbol confidence:", symbol.confidence)






# Entities
prev_page_entities = prev_page_data["entities"].values[0]
page_dict["prev_page_shared_entities"] = len(set(prev_page_entities).intersection(set(page_entities)))

total_count = 0
for tuple_ in entities:
    total_count += prev_page_entities.count(tuple_)
page_dict["prev_page_shared_entities_total"] = total_count











# Example usage
text1 = "Bob went to the store. Bob bought some milk. Susan also went to the store. Susan bought some bread. Susan liked it."
text2 = "Bob was at the park. Susan was at the park too."

shared_entities_count = count_shared_entities(text1, text2)
print(shared_entities_count)










