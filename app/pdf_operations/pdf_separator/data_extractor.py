import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import spacy
from google.cloud import vision

from pdf_operations.pdf_separator.config import get_vision_client

class DataExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.client = get_vision_client()
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = self.nlp.Defaults.stop_words

    def extract_data(self) -> Tuple[pd.DataFrame, int]:
        with fitz.open(self.pdf_path) as doc:
            total_pages = len(doc)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(executor.map(self.process_page, [doc]*total_pages, range(total_pages)))
            return pd.DataFrame(results), total_pages

    def process_page(self, doc: fitz.Document, page_number: int) -> Dict:

        # Print the page number being processed
        print(f"Extracting data from page {page_number + 1}...")

        # Extract text from the page
        img_bytes, page = self.convert_page(doc, page_number)
        vision_image = vision.Image(content=img_bytes)
        response = self.client.text_detection(image=vision_image)

        # Organize the extracted text data
        page_data = self.organize_text_extraction(response, page, page_number)
        page_data['processed_full_text'] = self.preprocess_text(page_data['full_text'])

        return page_data

    def convert_page(self, doc: fitz.Document, page_number: int) -> Tuple[bytes, fitz.Page]:
        page = doc.load_page(page_number)
        image = page.get_pixmap()
        img_bytes = image.tobytes("png")
        return img_bytes, page


    def organize_text_extraction(self, response: vision.AnnotateImageResponse, page: fitz.Page, page_number: int) -> Dict:
        texts = response.text_annotations
        full_text = texts[0].description if texts else ""

        text_data = [
            {
                'description': text.description,
                'confidence': text.confidence or 1.0,
                'bounding_poly': text.bounding_poly
            }
            for text in texts[1:]
        ]

        rows = full_text.split("\n")
        words = [entry['description'] for entry in text_data]
        filtered_words = [word for word in words if re.search(r'[a-zA-Z0-9]', word)]

        entities = self.extract_entities(full_text)

        bounding_boxes = [entry['bounding_poly'] for entry in text_data]

        page_width, page_height = page.rect.width, page.rect.height
        avg_x_position, avg_x_first_10, avg_x_last_10 = self.calculate_average_x_position(bounding_boxes)


        return {
            "page_num": page_number + 1,
            "full_text": full_text,
            "text_rows": rows,
            "text_words": words,
            "filtered_words": filtered_words,
            "page_header": rows[:3],
            "word_count": len(filtered_words),
            "rows_of_text_count": len(rows),
            "page_width": page_width,
            "page_height": page_height,
            "average_text_size": self.calculate_average_text_size(bounding_boxes),
            "text_density": self.calculate_text_density(bounding_boxes, page_width, page_height),
            "first_line_vertical_position": self.get_first_line_vertical_position(texts),
            "vertical_margin_ratio": self.calculate_margin_ratios(response, page_width, page_height)[0],
            "horizontal_margin_ratio": self.calculate_margin_ratios(response, page_width, page_height)[1],
            "entities": entities,
            "number_of_entities": len(entities),
            "number_of_unique_entities": len(set(entities)),
            "title_candidates": self.detect_title(full_text, bounding_boxes, page_width),
            "whitespace_ratio": self.calculate_whitespace_ratio(page, bounding_boxes),
            "has_images_or_tables": self.detect_images_or_tables(page),
            "average_x_position": avg_x_position,
            "avg_x_first_10": avg_x_first_10,
            "avg_x_last_10": avg_x_last_10,
            "text_confidences": [entry['confidence'] for entry in text_data],
            "line_spacing": self.calculate_line_spacing(bounding_boxes),
            "has_page_1": self.check_page_numbering(full_text, r'page\s+1'),
            "has_page_x": self.check_page_numbering(full_text, r'page\s+\d+'),
            "has_page_x_of_x": self.check_page_numbering(full_text, r'page\s+\d+\s+of\s+\d+'),
            "has_page_x_of_x_end": self.check_page_numbering(full_text, r'page\s+(\d+)\s+of\s+\1')
        }

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\W+', ' ', text)
        return ' '.join([token.lemma_ for token in self.nlp(text) if token.text not in self.stop_words])

    def calculate_line_spacing(self, bounding_boxes: List[vision.BoundingPoly]) -> float:
        if len(bounding_boxes) < 2:
            return 0
        line_spacings = [abs(current_box.vertices[0].y - previous_box.vertices[2].y)
                         for previous_box, current_box in zip(bounding_boxes[:-1], bounding_boxes[1:])]
        return np.mean(line_spacings) if line_spacings else 0

    def calculate_average_x_position(self, bounding_boxes: List[vision.BoundingPoly]) -> Tuple[float, float, float]:
        start_x_positions = [box.vertices[0].x for box in bounding_boxes if box and len(box.vertices) > 0]
        if not start_x_positions:
            return 0, 0, 0

        total_rows = len(start_x_positions)
        first_10_percent_count = max(1, int(0.1 * total_rows))
        last_10_percent_count = max(1, int(0.1 * total_rows))

        return (
            np.mean(start_x_positions),
            np.mean(start_x_positions[:first_10_percent_count]),
            np.mean(start_x_positions[-last_10_percent_count:])
        )

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        doc = self.nlp(text)
        return [(entity.text, entity.label_) for entity in doc.ents]

    def calculate_average_text_size(self, bounding_boxes: List[vision.BoundingPoly]) -> float:
        sizes = [self.get_text_size(box) for box in bounding_boxes if box]
        return np.mean(sizes) if sizes else 0

    def get_text_size(self, bounding_box: vision.BoundingPoly) -> float:
        return abs(bounding_box.vertices[0].y - bounding_box.vertices[2].y)

    def calculate_text_density(self, bounding_boxes: List[vision.BoundingPoly], page_width: float, page_height: float) -> float:
        total_area = sum((box.vertices[1].x - box.vertices[0].x) * (box.vertices[2].y - box.vertices[0].y)
                         for box in bounding_boxes if box)
        page_area = page_width * page_height
        return total_area / page_area if page_area > 0 else 0

    def get_first_line_vertical_position(self, texts: List[vision.EntityAnnotation]) -> float:
        if not texts:
            return 0
        return min(vertex.y for text in texts for vertex in text.bounding_poly.vertices)

    def calculate_margin_ratios(self, response: vision.AnnotateImageResponse, page_width: float, page_height: float) -> Tuple[float, float]:
        header_threshold = 0.1 * page_height
        footer_threshold = 0.9 * page_height

        min_x, min_y, max_x, max_y = page_width, page_height, 0, 0
        valid_text_found = False

        for text in response.text_annotations[1:]:
            vertices = text.bounding_poly.vertices
            text_min_y = min(vertex.y for vertex in vertices)
            text_max_y = max(vertex.y for vertex in vertices)

            if text_max_y < header_threshold or text_min_y > footer_threshold:
                continue

            min_x = min(min_x, min(vertex.x for vertex in vertices))
            max_x = max(max_x, max(vertex.x for vertex in vertices))
            min_y = min(min_y, text_min_y)
            max_y = max(max_y, text_max_y)

            valid_text_found = True

        if not valid_text_found:
            return 0, 0

        vertical_margin_ratio = (min_y + (page_height - max_y)) / page_height
        horizontal_margin_ratio = (min_x + (page_width - max_x)) / page_width

        return vertical_margin_ratio, horizontal_margin_ratio

    def detect_title(self, full_text: str, bounding_boxes: List[vision.BoundingPoly], page_width: float) -> List[str]:
        return [text for text, box in zip(full_text.split('\n'), bounding_boxes) if self.is_title_candidate(text, box, page_width)]

    def is_title_candidate(self, text: str, bounding_box: vision.BoundingPoly, page_width: float) -> bool:
        return (
            self.get_text_size(bounding_box) > 20 and
            abs((bounding_box.vertices[1].x + bounding_box.vertices[0].x) / 2 - page_width / 2) < page_width * 0.2
        )

    def calculate_whitespace_ratio(self, page: fitz.Page, bounding_boxes: List[vision.BoundingPoly]) -> float:
        page_area = page.rect.width * page.rect.height
        text_area = sum((box.vertices[1].x - box.vertices[0].x) * (box.vertices[2].y - box.vertices[0].y)
                        for box in bounding_boxes if box)
        whitespace_area = page_area - text_area
        return whitespace_area / page_area if page_area > 0 else 0

    def detect_images_or_tables(self, page: fitz.Page) -> int:
        return int(bool(page.get_images()) or self.detect_tables(page))

    def detect_tables(self, page: fitz.Page) -> bool:
        text = page.get_text("text")
        table_indicators = ["Table", "TABLE", "table"]
        return any(indicator in text for indicator in table_indicators)

    def check_page_numbering(self, full_text: str, pattern: str) -> int:
        return int(bool(re.search(pattern, full_text, re.IGNORECASE)))
