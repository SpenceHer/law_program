
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
# import numpy as np
# import cv2
# from PIL import ImageStat, Image
import spacy
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import numpy as np




class PageToImageConversion:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def get_pdf_page_count(self):
        with fitz.open(self.pdf_path) as doc:
            return doc.page_count

    def convert_pdf_to_images_parallel(self, num_processes=None):
        if num_processes is None:
            num_processes = os.cpu_count()  # Default to the number of CPUs available

        total_pages = self.get_pdf_page_count()
        pages_per_process = (total_pages + num_processes - 1) // num_processes  # Ceiling division
        page_ranges = [(i * pages_per_process + 1, min((i + 1) * pages_per_process, total_pages)) for i in range(num_processes)]

        images = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Associate each future with its index
            future_to_index = {executor.submit(self.convert_from_page_range, start, end): i for i, (start, end) in enumerate(page_ranges)}
            results = []

            # Collect futures as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                images_results = future.result()
                results.append((index, images_results))

            # Sort results by index and then extend the images list in the correct order
            for _, images_results in sorted(results):
                images.extend(images_results)




        return images

    def convert_from_page_range(self, start_page, end_page):
        return convert_from_path(self.pdf_path, first_page=start_page, last_page=end_page)

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################




class TextExtraction:
    def __init__(self, credentials_path):
        self.client = config.get_vision_client(credentials_path)

    def extract_text_parallel(self, images):
        self.images = images
        total_pages = len(self.images)
        num_processes = os.cpu_count()  # Use all available CPU cores

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(self.extract_and_organize_text, images, range(1, total_pages + 1)))

        page_df = pd.DataFrame(results)
        return page_df

    def text_extraction(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')  # Google Vision prefers PNG
        img_byte_arr = img_byte_arr.getvalue()

        vision_image = vision.Image(content=img_byte_arr)
        response = self.client.text_detection(image=vision_image)

        return response

    def extract_and_organize_text(self, image, page_num):
        print(f"Extracting text from page {page_num} of {len(self.images)}")
        page_data = {"page_num": page_num}
        
        response = self.text_extraction(image)
        
        texts = response.text_annotations

        full_text = texts[0].description if texts else ''
        rows = full_text.split("\n")
        words = [text.description for text in texts[1:]]
        filtered_words = [word for word in words if re.search(r'[a-zA-Z0-9]', word)]

        bounding_boxes = [text.bounding_poly for text in response.text_annotations[1:]]
        average_text_size = self.calculate_average_text_size(bounding_boxes)

        # Assuming text extraction returns bounding boxes for all texts
        text_density = self.calculate_text_density(bounding_boxes, image.size[0], image.size[1])


        page_width = image.size[0]
        page_height = image.size[1]


        first_line_vertical_position = self.get_first_line_vertical_position(texts)

        vertical_margin_ratio, horizontal_margin_ratio = self.calculate_margin_ratios(response, page_width, page_height)


        page_data.update({
            "full_text": full_text,
            "text_rows": rows,
            "text_words": words,
            "filtered_words": filtered_words,
            "word_count": len(filtered_words),
            "rows_of_text_count": len(rows),
            "page_width": page_width,
            "page_height": page_height,
            "average_text_size": average_text_size,
            "text_density": text_density,
            "first_line_vertical_position": first_line_vertical_position,
            "vertical_margin_ratio": vertical_margin_ratio,
            "horizontal_margin_ratio": horizontal_margin_ratio
        })
        
        return page_data

    def calculate_average_text_size(self, bounding_boxes):
        sizes = [self.get_text_size(box) for box in bounding_boxes if box]
        if sizes:
            return np.mean(sizes)
        else:
            return 0

    def get_text_size(self, bounding_box):
        # Calculate the average height of the text based on the bounding box
        box_height = abs(bounding_box.vertices[0].y - bounding_box.vertices[2].y)
        return box_height

    def calculate_text_density(self, bounding_boxes, page_width, page_height):
        total_area = sum(
            (box.vertices[1].x - box.vertices[0].x) * (box.vertices[2].y - box.vertices[0].y)
            for box in bounding_boxes if box
        )
        page_area = page_width * page_height
        return total_area / page_area if page_area > 0 else 0

    def get_first_line_vertical_position(self, texts):
        if not texts:
            return None  # Return None if there are no text annotations

        # Initialize with a very high value
        min_y = float('inf')
        
        # Loop through each text annotation
        for text in texts:
            # Each 'text' has a bounding_poly attribute with vertices (list of vertices)
            for vertex in text.bounding_poly.vertices:
                # Update min_y if a smaller y value is found
                if vertex.y < min_y:
                    min_y = vertex.y

        return min_y if min_y != float('inf') else None


    def calculate_margin_ratios(self, response, page_width, page_height):

        # Initialize min and max coordinates
        min_x, min_y = page_width, page_height
        max_x, max_y = 0, 0

        # Loop through all detected text annotations (excluding the first element which is the full text)
        for text in response.text_annotations[1:]:
            vertices = text.bounding_poly.vertices
            min_x = min(min_x, min(vertex.x for vertex in vertices))
            max_x = max(max_x, max(vertex.x for vertex in vertices))
            min_y = min(min_y, min(vertex.y for vertex in vertices))
            max_y = max(max_y, max(vertex.y for vertex in vertices))

        # Calculate margins
        vertical_margin_top = min_y
        vertical_margin_bottom = page_height - max_y
        horizontal_margin_left = min_x
        horizontal_margin_right = page_width - max_x

        # Calculate margin ratios
        vertical_margin_ratio = (vertical_margin_top + vertical_margin_bottom) / page_height
        horizontal_margin_ratio = (horizontal_margin_left + horizontal_margin_right) / page_width

        return vertical_margin_ratio, horizontal_margin_ratio




################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################





class PageComparison:
    def __init__(self):
        # Load spaCy model for entity recognition
        self.nlp = spacy.load("en_core_web_sm")

        # Load pre-trained BERT model for text embeddings
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # TF-IDF vectorizor
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)



    def page_comparison_parallel(self, page_df, num_processes=None):
        self.page_df = page_df

        # Define the number of workers for parallel processing
        if num_processes is None:
            num_processes = os.cpu_count()

        # Create a process pool executor
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(self.compare_pages, page_num) for page_num in self.page_df['page_num'].unique()]

            # Collect results as they complete
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Combine all the results into a single DataFrame
        result_df = pd.DataFrame(results)
        return result_df


    def compare_pages(self, page_num):
        
        print(f"Analyzing page {page_num} of {len(self.page_df)}")

        page_dict = {}
        page_dict["page_num"] = page_num

        prev_page_data = self.page_df.loc[self.page_df["page_num"] == page_num - 1]
        page_data = self.page_df.loc[self.page_df["page_num"] == page_num]
        next_page_data = self.page_df.loc[self.page_df["page_num"] == page_num + 1]
        

    ################################################

        # Current Page Data
        # page_entities = page_data["entities"].values[0]
        page_header = page_data["text_rows"].values[0][0:3]

        # Email start pattern
        self.page_df.loc[(self.page_df["page_num"] == page_num) & ("message" == page_data["filtered_words"].values[0][0].lower()), "email_start_1"] = 1
        self.page_df.loc[(self.page_df["page_num"] == page_num) & ("message" != page_data["filtered_words"].values[0][0].lower()), "email_start_1"] = 0


    ################################################

        # Previous Page Data
        if len(prev_page_data) > 0:

            # Presence of previous page
            page_dict["prev_page"] = 1

            # Full text similarities
            page_dict["prev_page_bert_text_similarity"] = self.calculate_bert_text_similarity(page_data["full_text"].values[0], prev_page_data["full_text"].values[0])
            page_dict["prev_page_idf_text_similarity"] = self.calculate_idf_text_similarity(page_data["full_text"].values[0], prev_page_data["full_text"].values[0])


            # Entities
            # prev_page_entities = prev_page_data["entities"].values[0]
            # page_dict["prev_page_shared_entities"] = len(set(prev_page_entities).intersection(set(page_entities)))

            # total_count = 0
            # for tuple_ in page_entities:
            #     total_count += prev_page_entities.count(tuple_)
            # page_dict["prev_page_shared_entities_total"] = total_count

            # Margins
            page_dict["prev_vertical_margin_ratio_diff"] = abs(page_data["vertical_margin_ratio"].values[0] - prev_page_data["vertical_margin_ratio"].values[0])
            page_dict["prev_horizontal_margin_ratio_diff"] = abs(page_data["horizontal_margin_ratio"].values[0] - prev_page_data["horizontal_margin_ratio"].values[0])

            # Word Count
            page_dict["prev_page_word_count"] = page_data["word_count"].values[0]

            # Header similarity
            prev_page_header = prev_page_data["text_rows"].values[0][0:3]
            page_dict["prev_page_header_similarity"] = self.calculate_idf_text_similarity(" ".join(page_header), " ".join(prev_page_header))

            # Page height and width
            page_dict["prev_page_width_diff"] = abs(page_data["page_width"].values[0] - prev_page_data["page_width"].values[0])
            page_dict["prev_page_height_diff"] = abs(page_data["page_height"].values[0] - prev_page_data["page_height"].values[0])

            # Text Polarity and Sentiment
            # page_dict["prev_page_polarity_diff"] = abs(page_data["text_polarity"].values[0] - prev_page_data["text_polarity"].values[0])
            # page_dict["prev_page_subjectivity_diff"] = abs(page_data["text_subjectivity"].values[0] - prev_page_data["text_subjectivity"].values[0])

            # Dominant Color Difference
            # page_dict["prev_page_dom_color_diff"] = self.calculate_euclidean_distance(str(page_data["dominant_color"].values[0]), str(prev_page_data["dominant_color"].values[0]))

        else:
            page_dict["prev_page"] = 0
            page_dict["prev_page_bert_text_similarity"] = 0
            page_dict["prev_page_idf_text_similarity"] = 0
            # page_dict["prev_page_shared_entities"] = 0
            # page_dict["prev_page_shared_entities_total"] = 0
            page_dict["prev_vertical_margin_ratio_diff"] = 0
            page_dict["prev_horizontal_margin_ratio_diff"] = 0
            page_dict["prev_page_word_count"] = 0
            page_dict["prev_page_header_similarity"] = 0
            page_dict["prev_page_width_diff"] = 0
            page_dict["prev_page_height_diff"] = 0
            # page_dict["prev_page_polarity_diff"] = 0
            # page_dict["prev_page_subjectivity_diff"] = 0
            # page_dict["prev_page_dom_color_diff"] = 0

    ################################################

        # Next Page Data
        if len(next_page_data) > 0:
            
            # Presence of next page
            page_dict["next_page"] = 1


            # Full text similarities
            page_dict["next_page_bert_text_similarity"] = self.calculate_bert_text_similarity(page_data["full_text"].values[0], next_page_data["full_text"].values[0])
            page_dict["next_page_idf_text_similarity"] = self.calculate_idf_text_similarity(page_data["full_text"].values[0], next_page_data["full_text"].values[0])


            # Entities
            # next_page_entities = next_page_data["entities"].values[0]
            # page_dict["next_page_shared_entities"] = len(set(next_page_entities).intersection(set(page_entities)))

            # total_count = 0
            # for tuple_ in page_entities:
            #     total_count += next_page_entities.count(tuple_)
            # page_dict["next_page_shared_entities_total"] = total_count

            # Margins
            page_dict["next_vertical_margin_ratio_diff"] = abs(page_data["vertical_margin_ratio"].values[0] - next_page_data["vertical_margin_ratio"].values[0])
            page_dict["next_horizontal_margin_ratio_diff"] = abs(page_data["horizontal_margin_ratio"].values[0] - next_page_data["horizontal_margin_ratio"].values[0])

            # Word Count
            page_dict["next_page_word_count"] = page_data["word_count"].values[0]

            # Header similarity
            next_page_header = next_page_data["text_rows"].values[0][0:3]

            page_dict["next_page_header_similarity"] = self.calculate_idf_text_similarity(" ".join(page_header), " ".join(next_page_header))

            # Page height and width
            page_dict["next_page_width_diff"] = abs(page_data["page_width"].values[0] - next_page_data["page_width"].values[0])
            page_dict["next_page_height_diff"] = abs(page_data["page_height"].values[0] - next_page_data["page_height"].values[0])

            # Text Polarity and Sentiment
            # page_dict["next_page_polarity_diff"] = abs(page_data["text_polarity"].values[0] - next_page_data["text_polarity"].values[0])
            # page_dict["next_page_subjectivity_diff"] = abs(page_data["text_subjectivity"].values[0] - next_page_data["text_subjectivity"].values[0])

            # Dominant Color Difference
            # page_dict["next_page_dom_color_diff"] = self.calculate_euclidean_distance(str(page_data["dominant_color"].values[0]), str(next_page_data["dominant_color"].values[0]))

        else:

            page_dict["next_page"] = 0
            page_dict["next_page_bert_text_similarity"] = 0
            page_dict["next_page_idf_text_similarity"] = 0
            # page_dict["next_page_shared_entities"] = 0
            # page_dict["next_page_shared_entities_total"] = 0
            page_dict["next_vertical_margin_ratio_diff"] = 0
            page_dict["next_horizontal_margin_ratio_diff"] = 0
            page_dict["next_page_word_count"] = 0
            page_dict["next_page_header_similarity"] = 0
            page_dict["next_page_width_diff"] = 0
            page_dict["next_page_height_diff"] = 0
            # page_dict["next_page_polarity_diff"] = 0
            # page_dict["next_page_subjectivity_diff"] = 0
            # page_dict["next_page_dom_color_diff"] = 0


        return page_dict




    def get_bert_embeddings(self, text):
        encoded_input = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = self.bert_model(**encoded_input)
        embeddings = output.last_hidden_state.mean(dim=1)
        return embeddings


    def calculate_bert_text_similarity(self, text1, text2):
        emb1 = self.get_bert_embeddings(text1)
        emb2 = self.get_bert_embeddings(text2)
        similarity = cosine_similarity(emb1, emb2)
        return similarity[0][0]


    def calculate_idf_text_similarity(self, text1, text2):
        vector1 = self.vectorizer.transform([text1])
        vector2 = self.vectorizer.transform([text2])
        cosine_similarities = cosine_similarity(vector1, vector2)
        return cosine_similarities[0, 0]


    def calculate_euclidean_distance(self, color_string1, color_string2):
        def convert_color_string(color_string):
            color_values = re.sub(r'[\[\]]', '', color_string).strip().split()
            return [int(value) for value in color_values]
        color1 = np.array(convert_color_string(color_string1))
        color2 = np.array(convert_color_string(color_string2))
        return np.linalg.norm(color1 - color2)
