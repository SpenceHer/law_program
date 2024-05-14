
from pdf2image import convert_from_path
import io
import config
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
import io
import numpy as np
from multiprocessing import Manager
from collections import Counter


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


class ProcessPDF:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.client = config.get_vision_client()
        self.nlp = spacy.load("en_core_web_sm")



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
        print(f"Processing page {page_number + 1} of {len(doc)}")

        content, page = self.convert_page(doc, page_number)

        # Prepare the image for Google Vision
        vision_image = vision.Image(content=content)
        response = self.client.document_text_detection(image=vision_image)

        # Organize text extraction results
        page_data = self.organize_text_extraction(response, page, page_number)

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
        
        # Text Attributes
        full_text = texts[0].description if texts else ''
        rows = full_text.split("\n")
        words = [text.description for text in texts[1:]]
        filtered_words = [word for word in words if re.search(r'[a-zA-Z0-9]', word)]

        # Text size
        bounding_boxes = [text.bounding_poly for text in response.text_annotations[1:]]
        average_text_size = self.calculate_average_text_size(bounding_boxes)

        # Page size
        page_size = page.rect  # This is a fitz.Rect object
        page_width = page_size.width
        page_height = page_size.height

        # Assuming text extraction returns bounding boxes for all texts
        text_density = self.calculate_text_density(bounding_boxes, page_width, page_height)
        entities = self.extract_entities(full_text)

        page_header = rows[0:3]

        first_line_vertical_position = self.get_first_line_vertical_position(texts)
        # first_line_horizontal_position = self.get_first_line_horizontal_position(texts)
        vertical_margin_ratio, horizontal_margin_ratio = self.calculate_margin_ratios(response, page_width, page_height)

    

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
            "entities": entities
        }


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




################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################




class PageComparison:
    def __init__(self):
        # Initialize tokenizer and model eagerly if they are commonly used and expensive to load each time
        from transformers import BertTokenizer, BertModel
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Load pre-trained BERT model for text embeddings
        self._bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Load TF-IDF vectorizer
        self._vectorizer = TfidfVectorizer(stop_words='english')

        manager = Manager()
        self._embeddings_cache = manager.dict()  # Use Manager's dict



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


    def page_comparison_parallel(self, page_df):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import pandas as pd

        num_processes = 6
        page_dict = page_df.set_index('page_num').to_dict('index')

        # Batch pages for reduced overhead in process creation
        page_nums = list(page_df['page_num'].unique())
        batch_size = max(len(page_nums) // num_processes, 1)
        page_batches = [page_nums[i:i + batch_size] for i in range(0, len(page_nums), batch_size)]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.process_page_batch, batch, page_dict) for batch in page_batches]
            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        # Transform list of tuples into a dictionary of dictionaries
        results_dict = {result[0]: result[1] for result in results}

        # Convert the dictionary to a DataFrame
        result_df = pd.DataFrame.from_dict(results_dict, orient='index')
        return result_df

    def process_page_batch(self, page_nums, page_dict):
        results = []
        for page_num in page_nums:
            results.append(self.compare_pages(page_num, page_dict))
        return results








    def compare_pages(self, page_num, page_dict):
        print(f"Analyzing page {page_num} of {len(page_dict)}")
        page_data = page_dict[page_num]
        prev_page_data = page_dict.get(page_num - 1)
        next_page_data = page_dict.get(page_num + 1)
        
        results = {"page_num": page_num}
        
        # Handle current page processing
        # Here you would process current page's data, e.g., extracting headers, computing specific flags etc.

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


    def process_adjacent_page(self, current_page, adjacent_page, prefix, no_page=False):
        result = {}
        
        if no_page == False:

            # Compute BERT and TF-IDF text similarities
            result[f"{prefix}_page_bert_text_similarity"] = self.calculate_bert_text_similarity(
                current_page["full_text"], adjacent_page["full_text"]
            )
            result[f"{prefix}_page_idf_text_similarity"] = self.calculate_idf_text_similarity(
                current_page["full_text"], adjacent_page["full_text"]
            )

            # Compute differences in page dimensions
            result[f"{prefix}_page_width_diff"] = abs(current_page["page_width"] - adjacent_page["page_width"])
            result[f"{prefix}_page_height_diff"] = abs(current_page["page_height"] - adjacent_page["page_height"])

            result[f"{prefix}_vertical_margin_ratio_diff"] = abs(current_page["vertical_margin_ratio"] - adjacent_page["vertical_margin_ratio"])
            result[f"{prefix}_horizontal_margin_ratio_diff"] = abs(current_page["horizontal_margin_ratio"] - adjacent_page["horizontal_margin_ratio"])

            result[f"{prefix}_page_word_count"] = abs(current_page["word_count"] - adjacent_page["word_count"])

            result[f"{prefix}_page_header_similarity"] = self.calculate_idf_text_similarity(" ".join(current_page["page_header"]), " ".join(adjacent_page["page_header"]))
        
            result[f"{prefix}_page_shared_entities"] = len(set(current_page["entities"]).intersection(set(adjacent_page["entities"])))

            result[f"{prefix}_page_shared_entities_total"] = self.count_shared_entities(current_page["entities"], adjacent_page["entities"])




        else:
            result[f"{prefix}_page_bert_text_similarity"] = 0
            result[f"{prefix}_page_idf_text_similarity"] = 0
            result[f"{prefix}_page_width_diff"] = 0
            result[f"{prefix}_page_height_diff"] = 0
            result[f"{prefix}_vertical_margin_ratio_diff"] = 0
            result[f"{prefix}_horizontal_margin_ratio_diff"] = 0
            result[f"{prefix}_page_word_count"] = 0
            result[f"{prefix}_page_header_similarity"] = 0
            result[f"{prefix}_page_shared_entities"] = 0
            result[f"{prefix}_page_shared_entities_total"] = 0

        return result


    def get_bert_embeddings(self, text):
        # Caching embeddings to avoid recomputing for the same text
        if not hasattr(self, '_embeddings_cache'):
            self._embeddings_cache = {}
        
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]

        encoded_input = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = self.bert_model(**encoded_input)
        embeddings = output.last_hidden_state.mean(dim=1)
        self._embeddings_cache[text] = embeddings
        return embeddings


    def calculate_bert_text_similarity(self, text1, text2):
        emb1 = self.get_bert_embeddings(text1)
        emb2 = self.get_bert_embeddings(text2)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())
        return similarity[0][0]


    def calculate_idf_text_similarity(self, text1, text2):
        vector1 = self.vectorizer.transform([text1])
        vector2 = self.vectorizer.transform([text2])
        from sklearn.metrics.pairwise import cosine_similarity
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

