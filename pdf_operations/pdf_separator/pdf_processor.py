from pdf2image import convert_from_path
import io
from .config import get_vision_client
from google.cloud import vision
import re
import concurrent.futures
import os
import fitz  # PyMuPDF for PDF handling
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import spacy
# from textblob import TextBlob
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
# from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from multiprocessing import Manager, Value, Lock
from collections import Counter
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from PyPDF2 import PdfReader, PdfWriter


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


class ProcessPDF:
    def __init__(self, pdf_path):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.pdf_path = pdf_path
        self.client = get_vision_client()
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()



    def process_pdf(self):
        with fitz.open(self.pdf_path) as doc:
            total_pages = len(doc)
            results = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(self.process_page, doc, i) for i in range(total_pages)]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            return pd.DataFrame(results), total_pages

    def process_page(self, doc, page_number):
        # print(f"Processing page {page_number + 1} of {len(doc)}")

        content, page = self.convert_page(doc, page_number)

        # Prepare the image for Google Vision
        vision_image = vision.Image(content=content)
        response = self.client.text_detection(image=vision_image)

        # Organize text extraction results
        page_data = self.organize_text_extraction(response, page, page_number)

        # Preprocess the full text
        page_data['processed_full_text'] = self.preprocess_text(page_data['full_text'])

        return page_data

    def convert_page(self, doc, page_number):
        page = doc.load_page(page_number)
        image = page.get_pixmap()

        # Converting the pixmap directly to bytes in PNG format
        img_bytes = image.tobytes("png")
        img_byte_arr = io.BytesIO(img_bytes)  # Now this contains the PNG data
        content = img_byte_arr.getvalue()
        
        return content, page

    def organize_text_extraction(self, response, page, page_number):
        # Full text
        texts = response.text_annotations

        if texts:
            full_text = texts[0].description
        else:
            full_text = ""

        # Extract text data and confidence scores
        text_data = []
        for text in texts[1:]:
            text_entry = {
                'description': text.description,
                'confidence': text.confidence if text.confidence else 1.0,  # Assign 1.0 if confidence is not available
                'bounding_poly': text.bounding_poly
            }
            text_data.append(text_entry)

        rows = full_text.split("\n")
        words = [entry['description'] for entry in text_data]
        filtered_words = [word for word in words if re.search(r'[a-zA-Z0-9]', word)]

        # Text size
        bounding_boxes = [entry['bounding_poly'] for entry in text_data]
        average_text_size = self.calculate_average_text_size(bounding_boxes)

        # Page size
        page_size = page.rect  # This is a fitz.Rect object
        page_width = page_size.width
        page_height = page_size.height

        # Calculate text density
        text_density = self.calculate_text_density(bounding_boxes, page_width, page_height)
        entities = self.extract_entities(full_text)
        number_of_entities = len(entities)
        number_of_unique_entities = len(set(entities))

        line_spacing = self.calculate_line_spacing(bounding_boxes)

        page_header = rows[0:3]

        first_line_vertical_position = self.get_first_line_vertical_position(texts)
        vertical_margin_ratio, horizontal_margin_ratio = self.calculate_margin_ratios(response, page_width, page_height)
        avg_x_position, avg_x_first_10, avg_x_last_10 = self.calculate_average_x_position(bounding_boxes)

        # Additional features
        title_candidates = self.detect_title(full_text, bounding_boxes, page_width)
        whitespace_ratio = self.calculate_whitespace_ratio(page, bounding_boxes)
        has_images_or_tables = self.detect_images_or_tables(page)

        has_page_1 = self.check_page_1(full_text)
        has_page_x = self.check_page_x(full_text)
        has_page_x_of_x = self.check_page_x_of_x(full_text)
        has_page_x_of_x_end = self.check_page_x_of_x_end(full_text)



        return {
            "page_num": page_number + 1,
            "full_text": full_text,
            "text_rows": rows,
            "text_words": words,
            "filtered_words": filtered_words,
            "page_header": page_header,
            "word_count": len(filtered_words),
            "rows_of_text_count": len(rows),
            "page_width": page_width,
            "page_height": page_height,
            "average_text_size": average_text_size,
            "text_density": text_density,
            "first_line_vertical_position": first_line_vertical_position,
            "vertical_margin_ratio": vertical_margin_ratio,
            "horizontal_margin_ratio": horizontal_margin_ratio,
            "entities": entities,
            "number_of_entities": number_of_entities,
            "number_of_unique_entities": number_of_unique_entities,
            "title_candidates": title_candidates,
            "whitespace_ratio": whitespace_ratio,
            "has_images_or_tables": has_images_or_tables,
            "average_x_position": avg_x_position,
            "avg_x_first_10": avg_x_first_10,
            "avg_x_last_10": avg_x_last_10,
            "text_confidences": [entry['confidence'] for entry in text_data],
            "line_spacing": line_spacing,
            "has_page_1": has_page_1,
            "has_page_x": has_page_x,
            "has_page_x_of_x": has_page_x_of_x,
            "has_page_x_of_x_end": has_page_x_of_x_end
        }


    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\W+', ' ', text)
        text = ' '.join([word for word in text.split() if word not in self.stop_words])
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])
        return text

    def calculate_line_spacing(self, bounding_boxes):
        # Calculate average line spacing between bounding boxes
        if len(bounding_boxes) < 2:
            return 0

        line_spacings = []
        for i in range(1, len(bounding_boxes)):
            previous_box = bounding_boxes[i - 1]
            current_box = bounding_boxes[i]
            spacing = abs(current_box.vertices[0].y - previous_box.vertices[2].y)
            line_spacings.append(spacing)

        return np.mean(line_spacings) if line_spacings else 0

    def calculate_average_x_position(self, bounding_boxes):
        start_x_positions = [
            box.vertices[0].x for box in bounding_boxes if box and len(box.vertices) > 0
        ]
        if not start_x_positions:
            return 0, 0, 0
        
        total_rows = len(start_x_positions)
        first_10_percent_count = max(1, int(0.1 * total_rows))
        last_10_percent_count = max(1, int(0.1 * total_rows))

        first_10_percent_positions = start_x_positions[:first_10_percent_count]
        last_10_percent_positions = start_x_positions[-last_10_percent_count:]

        avg_x_first_10 = np.mean(first_10_percent_positions)
        avg_x_last_10 = np.mean(last_10_percent_positions)
        avg_x_position = np.mean(start_x_positions)

        return avg_x_position, avg_x_first_10, avg_x_last_10

    def extract_entities(self, text):
        # Process the text
        doc = self.nlp(text)
        # Extract entities
        entities = [(entity.text, entity.label_) for entity in doc.ents]
        return entities

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
            return 0  # Return None if there are no text annotations

        # Initialize with a very high value
        min_y = float('inf')
        
        # Loop through each text annotation
        for text in texts:
            # Each 'text' has a bounding_poly attribute with vertices (list of vertices)
            for vertex in text.bounding_poly.vertices:
                # Update min_y if a smaller y value is found
                if vertex.y < min_y:
                    min_y = vertex.y

        return min_y if min_y != float('inf') else 0



    def calculate_margin_ratios(self, response, page_width, page_height):
        # Define the threshold for header and footer (e.g., 10% of page height)
        header_threshold = 0.1 * page_height
        footer_threshold = 0.9 * page_height

        # Initialize min and max coordinates
        min_x, min_y = page_width, page_height
        max_x, max_y = 0, 0

        # Flag to check if any valid text is found
        valid_text_found = False

        # Loop through all detected text annotations (excluding the first element which is the full text)
        for text in response.text_annotations[1:]:
            vertices = text.bounding_poly.vertices
            text_min_y = min(vertex.y for vertex in vertices)
            text_max_y = max(vertex.y for vertex in vertices)

            # Exclude text in header and footer regions
            if text_max_y < header_threshold or text_min_y > footer_threshold:
                continue

            # Update min and max coordinates
            min_x = min(min_x, min(vertex.x for vertex in vertices))
            max_x = max(max_x, max(vertex.x for vertex in vertices))
            min_y = min(min_y, text_min_y)
            max_y = max(max_y, text_max_y)

            valid_text_found = True

        if not valid_text_found:
            # Handle pages with no valid text (excluding headers and footers)
            # Return default margin ratios (e.g., 0) or any values appropriate for your model
            vertical_margin_ratio = 0
            horizontal_margin_ratio = 0
        else:
            # Calculate margins
            vertical_margin_top = min_y
            vertical_margin_bottom = page_height - max_y
            horizontal_margin_left = min_x
            horizontal_margin_right = page_width - max_x

            # Calculate margin ratios
            vertical_margin_ratio = (vertical_margin_top + vertical_margin_bottom) / page_height
            horizontal_margin_ratio = (horizontal_margin_left + horizontal_margin_right) / page_width

        return vertical_margin_ratio, horizontal_margin_ratio

    def detect_title(self, full_text, bounding_boxes, page_width):
        # This function identifies title candidates based on position and size
        title_candidates = []
        for text, box in zip(full_text.split('\n'), bounding_boxes):
            if self.is_title_candidate(text, box, page_width):
                title_candidates.append(text)
        return title_candidates

    def is_title_candidate(self, text, bounding_box, page_width):
        # Check if the text is a candidate for title based on heuristics
        is_large_font = self.get_text_size(bounding_box) > 20  # Assuming font size > 20 is large
        is_centered = abs((bounding_box.vertices[1].x + bounding_box.vertices[0].x) / 2 - page_width / 2) < page_width * 0.2
        return is_large_font and is_centered

    def calculate_whitespace_ratio(self, page, bounding_boxes):
        # Calculate the ratio of whitespace on the page
        page_area = page.rect.width * page.rect.height
        text_area = sum(
            (box.vertices[1].x - box.vertices[0].x) * (box.vertices[2].y - box.vertices[0].y)
            for box in bounding_boxes if box
        )
        whitespace_area = page_area - text_area
        return whitespace_area / page_area if page_area > 0 else 0

    def detect_images_or_tables(self, page):
        # Detect presence of images or tables
        has_images = bool(page.get_images())
        has_tables = self.detect_tables(page)
        return 1

    def detect_tables(self, page):
        # Implement table detection logic
        text = page.get_text("text")
        table_indicators = ["Table", "TABLE", "table"]
        return any(indicator in text for indicator in table_indicators)



    def check_page_1(self, full_text):
        """Check if the text contains 'page 1'."""
        pattern = r'page\s+1'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x(self, full_text):
        """Check if the text contains 'page x' where x is any number."""
        pattern = r'page\s+\d+'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x_of_x(self, full_text):
        """Check if the text contains 'page x of x' where x is any number."""
        pattern = r'page\s+\d+\s+of\s+\d+'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x_of_x_end(self, full_text):
        """Check if the text contains 'page x of x' and x values are the same indicating the end."""
        pattern = r'page\s+(\d+)\s+of\s+\1'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0













################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


class PageComparison:
    def __init__(self):
        self._bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._bert_model = BertModel.from_pretrained('bert-base-uncased')
        self._vectorizer = TfidfVectorizer(stop_words='english')
        self._embeddings_cache = Manager().dict()



    @property
    def bert_tokenizer(self):
        return self._bert_tokenizer

    @property
    def bert_model(self):
        return self._bert_model

    @property
    def vectorizer(self):
        return self._vectorizer

    def fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)


    # def page_comparison_parallel(self, page_df):
    #     num_processes = 6
    #     page_dict = page_df.set_index('page_num').to_dict('index')

    #     # Batch pages for reduced overhead in process creation
    #     page_nums = list(page_df['page_num'].unique())
    #     batch_size = max(len(page_nums) // num_processes, 1)
    #     page_batches = [page_nums[i:i + batch_size] for i in range(0, len(page_nums), batch_size)]

    #     with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #         futures = [executor.submit(self.process_page_batch, batch, page_dict, len(page_dict), (i == 0)) for i, batch in enumerate(page_batches)]
    #         results = []
    #         for future in as_completed(futures):
    #             results.extend(future.result())

    #     # Transform list of tuples into a dictionary of dictionaries
    #     results_dict = {result[0]: result[1] for result in results}

    #     # Convert the dictionary to a DataFrame
    #     result_df = pd.DataFrame.from_dict(results_dict, orient='index')
    #     return result_df


    def page_comparison_parallel(self, page_df):
        if os.cpu_count() < 6:
            num_processes = os.cpu_count()
        else:
            num_processes = 6

        page_dict = page_df.set_index('page_num').to_dict('index')
        page_nums = list(page_df['page_num'].unique())
        batch_size = max(len(page_nums) // num_processes, 1)
        page_batches = [page_nums[i:i + batch_size] for i in range(0, len(page_nums), batch_size)]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.process_page_batch, batch, page_dict, len(page_dict), (i == 0)) for i, batch in enumerate(page_batches)]
            results = [future.result() for future in as_completed(futures)]

        results_dict = {result[0]: result[1] for result_batch in results for result in result_batch}
        return pd.DataFrame.from_dict(results_dict, orient='index')


    def process_page_batch(self, page_nums, page_dict, total_pages, is_first_batch):
        results = []
        batch_size = len(page_nums)
        for i, page_num in enumerate(page_nums):
            results.append(self.compare_pages(page_num, page_dict, i, batch_size, total_pages, is_first_batch))
        return results

    def compare_pages(self, page_num, page_dict, current_index, batch_size, total_pages, is_first_batch):
        # Only print the progress if this is the first batch
        if is_first_batch:
            percent_done = (current_index + 1) / batch_size * 100
            # print(f"First batch progress: {percent_done:.2f}% ({current_index + 1}/{batch_size} pages processed)")



        page_data = page_dict[page_num]
        prev_page_data = page_dict.get(page_num - 1)
        next_page_data = page_dict.get(page_num + 1)
        
        results = {"page_num": page_num}

        # Handle current page processing
        # results.update(self.process_current_page(page_data))

        # Process previous page data
        if prev_page_data:
            results.update(self.process_adjacent_page(page_data, prev_page_data, 'prev', no_page=False))
        else:
            results.update(self.process_adjacent_page(page_data, prev_page_data, 'prev', no_page=True))

        
        # Process next page data
        if next_page_data:
            results.update(self.process_adjacent_page(page_data, next_page_data, 'next', no_page=False))
        else:
            results.update(self.process_adjacent_page(page_data, next_page_data, 'next', no_page=True))

        return page_num, results


    # def process_current_page(self, page_data):
    #     result = {}
    #     full_text = page_data["full_text"]
        
    #     # Check for page number patterns
    #     result["has_page_1"] = self.check_page_1(full_text)
    #     result["has_page_x"] = self.check_page_x(full_text)
    #     result["has_page_x_of_x"] = self.check_page_x_of_x(full_text)
    #     result["has_page_x_of_x_end"] = self.check_page_x_of_x_end(full_text)

    #     return result


    def process_adjacent_page(self, current_page, adjacent_page, prefix, no_page=False):
        result = {}

        
        if no_page == False:

            # Page dimensions
            result[f"{prefix}_page_width"] = adjacent_page["page_width"]
            result[f"{prefix}_page_height"] = adjacent_page["page_height"]

            result[f"{prefix}_page_width_diff"] = abs(current_page["page_width"] - adjacent_page["page_width"])
            result[f"{prefix}_page_height_diff"] = abs(current_page["page_height"] - adjacent_page["page_height"])


            # Margin ratios
            result[f"{prefix}_page_vertical_margin_ratio"] = adjacent_page["vertical_margin_ratio"]
            result[f"{prefix}_page_horizontal_margin_ratio"] = adjacent_page["horizontal_margin_ratio"]

            result[f"{prefix}_page_vertical_margin_ratio_diff"] = abs(current_page["vertical_margin_ratio"] - adjacent_page["vertical_margin_ratio"])
            result[f"{prefix}_page_horizontal_margin_ratio_diff"] = abs(current_page["horizontal_margin_ratio"] - adjacent_page["horizontal_margin_ratio"])


            # Page layout features
            result[f"{prefix}_page_whitespace_ratio"] = adjacent_page["whitespace_ratio"]
            result[f"{prefix}_page_has_images_or_tables"] = adjacent_page["has_images_or_tables"]
            result[f"{prefix}_page_average_x_position"] = adjacent_page["average_x_position"]
            result[f"{prefix}_page_avg_x_first_10"] = adjacent_page["avg_x_first_10"]
            result[f"{prefix}_page_avg_x_last_10"] = adjacent_page["avg_x_last_10"]

            result[f"{prefix}_page_whitespace_ratio_diff"] = abs(current_page["whitespace_ratio"] - adjacent_page["whitespace_ratio"])
            result[f"{prefix}_page_has_images_or_tables_diff"] = int(current_page["has_images_or_tables"] != adjacent_page["has_images_or_tables"])
            result[f"{prefix}_page_average_x_position_diff"] = abs(current_page["average_x_position"] - adjacent_page["average_x_position"])
            result[f"{prefix}_page_avg_x_first_10_diff"] = abs(current_page["avg_x_first_10"] - adjacent_page["avg_x_first_10"])
            result[f"{prefix}_page_avg_x_last_10_diff"] = abs(current_page["avg_x_last_10"] - adjacent_page["avg_x_last_10"])


            # Page text features
            result[f"{prefix}_page_word_count"] = adjacent_page["word_count"]
            result[f"{prefix}_page_first_line_vertical_position"] = adjacent_page["first_line_vertical_position"]
            result[f"{prefix}_page_text_density"] = adjacent_page["text_density"]
            result[f"{prefix}_page_average_text_size"] = adjacent_page["average_text_size"]
            result[f"{prefix}_page_rows_of_text_count"] = adjacent_page["rows_of_text_count"]

            result[f"{prefix}_page_word_count_diff"] = abs(current_page["word_count"] - adjacent_page["word_count"])
            result[f"{prefix}_page_first_line_vertical_position_diff"] = abs(current_page["first_line_vertical_position"] - adjacent_page["first_line_vertical_position"])
            result[f"{prefix}_page_text_density_diff"] = abs(current_page["text_density"] - adjacent_page["text_density"])
            result[f"{prefix}_page_average_text_size_diff"] = abs(current_page["average_text_size"] - adjacent_page["average_text_size"])
            result[f"{prefix}_page_rows_of_text_count_diff"] = abs(current_page["rows_of_text_count"] - adjacent_page["rows_of_text_count"])


            # Text similarity features
            result[f"{prefix}_page_bert_text_similarity"] = self.calculate_bert_text_similarity(current_page["processed_full_text"], adjacent_page["processed_full_text"])
            result[f"{prefix}_page_idf_text_similarity"] = self.calculate_idf_text_similarity(current_page["processed_full_text"], adjacent_page["processed_full_text"])
            result[f"{prefix}_page_header_similarity"] = self.calculate_idf_text_similarity(" ".join(current_page["page_header"]), " ".join(adjacent_page["page_header"]))


            # Entity features
            result[f"{prefix}_page_number_of_entities"] = adjacent_page["number_of_entities"]
            result[f"{prefix}_page_number_of_unique_entities"] = adjacent_page["number_of_unique_entities"]
            result[f"{prefix}_page_shared_entities"] = len(set(current_page["entities"]).intersection(set(adjacent_page["entities"])))
            result[f"{prefix}_page_shared_entities_total"] = self.count_shared_entities(current_page["entities"], adjacent_page["entities"])

            result[f"{prefix}_page_number_of_entities_diff"] = abs(current_page["number_of_entities"] - adjacent_page["number_of_entities"])
            result[f"{prefix}_page_number_of_unique_entities_diff"] = abs(current_page["number_of_unique_entities"] - adjacent_page["number_of_unique_entities"])
            result[f"{prefix}_page_shared_entities_diff"] = abs(current_page["number_of_entities"] - adjacent_page["number_of_entities"])
            result[f"{prefix}_page_shared_entities_total_diff"] = abs(current_page["number_of_unique_entities"] - adjacent_page["number_of_unique_entities"])

            #Page number features
            result[f"{prefix}_page_has_page_1"] = self.check_page_1(adjacent_page["full_text"])
            result[f"{prefix}_page_has_page_x"] = self.check_page_x(adjacent_page["full_text"])
            result[f"{prefix}_page_has_page_x_of_x"] = self.check_page_x_of_x(adjacent_page["full_text"])
            result[f"{prefix}_page_has_page_x_of_x_end"] = self.check_page_x_of_x_end(adjacent_page["full_text"])





        else:
            result[f"{prefix}_page_width"] = 1000  # Example: Maximum expected page width
            result[f"{prefix}_page_height"] = 2000  # Example: Maximum expected page height
            result[f"{prefix}_page_width_diff"] = 1000  # Example: Full width difference
            result[f"{prefix}_page_height_diff"] = 2000  # Example: Full height difference

            result[f"{prefix}_page_vertical_margin_ratio"] = 1.0  # Example: Full margin
            result[f"{prefix}_page_horizontal_margin_ratio"] = 1.0  # Example: Full margin
            result[f"{prefix}_page_vertical_margin_ratio_diff"] = 1.0  # Example: Full difference
            result[f"{prefix}_page_horizontal_margin_ratio_diff"] = 1.0  # Example: Full difference

            result[f"{prefix}_page_whitespace_ratio"] = 1.0  # Example: All whitespace
            result[f"{prefix}_page_has_images_or_tables"] = 1  # Indicate presence of images/tables
            result[f"{prefix}_page_average_x_position"] = 1000  # Example: Maximum x position
            result[f"{prefix}_page_avg_x_first_10"] = 1000  # Example: Maximum x position
            result[f"{prefix}_page_avg_x_last_10"] = 1000  # Example: Maximum x position
            result[f"{prefix}_page_whitespace_ratio_diff"] = 1.0  # Example: Full difference in whitespace ratio
            result[f"{prefix}_page_has_images_or_tables_diff"] = 1  # Difference in presence of images/tables
            result[f"{prefix}_page_average_x_position_diff"] = 1000  # Example: Full difference in x position
            result[f"{prefix}_page_avg_x_first_10_diff"] = 1000
            result[f"{prefix}_page_avg_x_last_10_diff"] = 1000


            result[f"{prefix}_page_word_count"] = 0  # Example: No words
            result[f"{prefix}_page_first_line_vertical_position"] = 2000  # Example: Maximum position
            result[f"{prefix}_page_text_density"] = 0.0  # Example: No text
            result[f"{prefix}_page_average_text_size"] = 0.0  # Example: No text size
            result[f"{prefix}_page_rows_of_text_count"] = 0  # Example: No rows of text
            result[f"{prefix}_page_word_count_diff"] = 1000  # Example: Maximum difference in word count
            result[f"{prefix}_page_first_line_vertical_position_diff"] = 2000  # Example: Maximum difference in first line position
            result[f"{prefix}_page_text_density_diff"] = 1.0  # Example: Full difference in text density
            result[f"{prefix}_page_average_text_size_diff"] = 1.0  # Example: Full difference in text size
            result[f"{prefix}_page_rows_of_text_count_diff"] = 100  # Example: Full difference in rows of text

            result[f"{prefix}_page_bert_text_similarity"] = 0.0  # No similarity
            result[f"{prefix}_page_idf_text_similarity"] = 0.0  # No similarity
            result[f"{prefix}_page_header_similarity"] = 0.0  # No similarity

            result[f"{prefix}_page_number_of_entities"] = 0  # No shared entities
            result[f"{prefix}_page_number_of_unique_entities"] = 0  # No shared unique entities
            result[f"{prefix}_page_shared_entities"] = 0  # No shared entities
            result[f"{prefix}_page_shared_entities_total"] = 0  # No shared entities total
            result[f"{prefix}_page_number_of_entities_diff"] = 0  # No difference in entities
            result[f"{prefix}_page_number_of_unique_entities_diff"] = 0  # No difference in unique entities
            result[f"{prefix}_page_shared_entities_diff"] = 0  # No difference in shared entities
            result[f"{prefix}_page_shared_entities_total_diff"] = 0  # No difference in shared entities total

            result[f"{prefix}_page_has_page_1"] = 0  # No page 1
            result[f"{prefix}_page_has_page_x"] = 0  # No page x
            result[f"{prefix}_page_has_page_x_of_x"] = 0
            result[f"{prefix}_page_has_page_x_of_x_end"] = 0

        return result


    def get_bert_embeddings(self, text):
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]
        encoded_input = self._bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = self._bert_model(**encoded_input)
        embeddings = output.last_hidden_state.mean(dim=1)
        self._embeddings_cache[text] = embeddings
        return embeddings


    def calculate_bert_text_similarity(self, text1, text2):
        emb1 = self.get_bert_embeddings(text1)
        emb2 = self.get_bert_embeddings(text2)
    
        similarity = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())
        return similarity[0][0]


    def calculate_idf_text_similarity(self, text1, text2):
        vector1 = self.vectorizer.transform([text1])
        vector2 = self.vectorizer.transform([text2])
        cosine_similarities = cosine_similarity(vector1, vector2)
        return cosine_similarities[0, 0]
    

    def count_shared_entities(self, entities1, entities2):   
        # Count occurrences of each entity
        count1 = Counter(entities1)
        count2 = Counter(entities2)
        
        # Find shared entities
        shared_entities = count1.keys() & count2.keys()
        
        # Calculate total shared entities
        total_shared_entities = sum((count1[entity] + count2[entity]) for entity in shared_entities)
        
        return total_shared_entities


    def check_page_1(self, full_text):
        """Check if the text contains 'page 1'."""
        pattern = r'page\s+1'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x(self, full_text):
        """Check if the text contains 'page x' where x is any number."""
        pattern = r'page\s+\d+'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x_of_x(self, full_text):
        """Check if the text contains 'page x of x' where x is any number."""
        pattern = r'page\s+\d+\s+of\s+\d+'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x_of_x_end(self, full_text):
        """Check if the text contains 'page x of x' and x values are the same indicating the end."""
        pattern = r'page\s+(\d+)\s+of\s+\1'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0
    




################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################




class PDFSeparation:
    def __init__(self, pdf_path, page_df):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.pdf_path = pdf_path
        self.page_df = page_df
        self.features = [
            "word_count",
            "rows_of_text_count",
            "page_width",
            "page_height",
            "average_text_size",
            "text_density",
            "first_line_vertical_position",
            "vertical_margin_ratio",
            "horizontal_margin_ratio",
            "number_of_entities",
            "number_of_unique_entities",
            "whitespace_ratio",
            "has_images_or_tables",
            "average_x_position",
            "avg_x_first_10",
            "avg_x_last_10",
            "line_spacing",
            "has_page_1",
            "has_page_x",
            "has_page_x_of_x",
            "has_page_x_of_x_end",


            "next_page_width",
            "next_page_height",
            "next_page_width_diff",
            "next_page_height_diff",
            "next_page_vertical_margin_ratio",
            "next_page_horizontal_margin_ratio",
            "next_page_vertical_margin_ratio_diff",
            "next_page_horizontal_margin_ratio_diff",
            "next_page_whitespace_ratio",
            "next_page_has_images_or_tables",
            "next_page_average_x_position",
            "next_page_avg_x_first_10",
            "next_page_avg_x_last_10",
            "next_page_whitespace_ratio_diff",
            "next_page_has_images_or_tables_diff",
            "next_page_average_x_position_diff",
            "next_page_avg_x_first_10_diff",
            "next_page_avg_x_last_10_diff",
            "next_page_word_count",
            "next_page_first_line_vertical_position",
            "next_page_text_density",
            "next_page_average_text_size",
            "next_page_rows_of_text_count",
            "next_page_word_count_diff",
            "next_page_first_line_vertical_position_diff",
            "next_page_text_density_diff",
            "next_page_average_text_size_diff",
            "next_page_rows_of_text_count_diff",
            "next_page_bert_text_similarity",
            "next_page_idf_text_similarity",
            "next_page_header_similarity",
            "next_page_number_of_entities",
            "next_page_number_of_unique_entities",
            "next_page_shared_entities",
            "next_page_shared_entities_total",
            "next_page_has_page_1",
            "next_page_has_page_x",
            "next_page_has_page_x_of_x",
            "next_page_has_page_x_of_x_end",



            "prev_page_width",
            "prev_page_height",
            "prev_page_width_diff",
            "prev_page_height_diff",
            "prev_page_vertical_margin_ratio",
            "prev_page_horizontal_margin_ratio",
            "prev_page_vertical_margin_ratio_diff",
            "prev_page_horizontal_margin_ratio_diff",
            "prev_page_whitespace_ratio",
            "prev_page_has_images_or_tables",
            "prev_page_average_x_position",
            "prev_page_avg_x_first_10",
            "prev_page_avg_x_last_10",
            "prev_page_whitespace_ratio_diff",
            "prev_page_has_images_or_tables_diff",
            "prev_page_average_x_position_diff",
            "prev_page_avg_x_first_10_diff",
            "prev_page_avg_x_last_10_diff",
            "prev_page_word_count",
            "prev_page_first_line_vertical_position",
            "prev_page_text_density",
            "prev_page_average_text_size",
            "prev_page_rows_of_text_count",
            "prev_page_word_count_diff",
            "prev_page_first_line_vertical_position_diff",
            "prev_page_text_density_diff",
            "prev_page_average_text_size_diff",
            "prev_page_rows_of_text_count_diff",
            "prev_page_bert_text_similarity",
            "prev_page_idf_text_similarity",
            "prev_page_header_similarity",
            "prev_page_number_of_entities",
            "prev_page_number_of_unique_entities",
            "prev_page_shared_entities",
            "prev_page_shared_entities_total",
            "prev_page_has_page_1",
            "prev_page_has_page_x",
            "prev_page_has_page_x_of_x",
            "prev_page_has_page_x_of_x_end"
            ]
        
        self.output_files = self.run_pdf_separation()

    def run_pdf_separation(self):
        self.predict_start_pages()
        return self.separate_pdf()

    def predict_start_pages(self):
        model = joblib.load(os.path.join(self.BASE_DIR, 'pdf_operations', 'pdf_separator', 'models', 'optimized_model_important.pkl'))
        pipeline = joblib.load(os.path.join(self.BASE_DIR, 'pdf_operations', 'pdf_separator', 'models', 'preprocessing_pipeline_important.pkl'))
        important_features_path = os.path.join(self.BASE_DIR, 'pdf_operations', 'pdf_separator', 'models', 'important_features_list.txt')
        with open(important_features_path, "r") as file:
            important_features = [line.strip() for line in file]

        X_new = self.page_df[important_features]

        X_new_transformed = pipeline.transform(X_new)
        predictions = model.predict(X_new_transformed)
        self.page_df['predictions'] = predictions


    def separate_pdf(self):
        reader = PdfReader(self.pdf_path)
        start_pages = self.page_df[self.page_df['predictions'] == 1].index.tolist()

        output_folder = self.create_unique_split_pdfs_folder(self.BASE_DIR)

        def split_pdf(start_pages, reader):
            documents = []
            for i, start_page in enumerate(start_pages):
                writer = PdfWriter()
                end_page = start_pages[i + 1] if i + 1 < len(start_pages) else len(reader.pages)
                for page_num in range(start_page, end_page):
                    writer.add_page(reader.pages[page_num])
                documents.append((writer, start_page, end_page - 1))
            return documents

        documents = split_pdf(start_pages, reader)
        output_files = []
        for i, (writer, start_page, end_page) in enumerate(documents):
            if start_page == end_page:
                output_pdf_path = os.path.join(output_folder, f'document_pages_{start_page + 1}.pdf')
            else:
                output_pdf_path = os.path.join(output_folder, f'document_pages_{start_page + 1}_to_{end_page + 1}.pdf')
            with open(output_pdf_path, 'wb') as output_pdf:
                writer.write(output_pdf)
            output_files.append(output_pdf_path)

        return output_files

    def create_unique_split_pdfs_folder(self, base_dir):
        for number in range(0, 1000):
            output_folder = os.path.join(base_dir, 'static', 'split_files', f'split_pdfs_{number}')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
                return output_folder