import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProcessPages:
    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words='english')

    @property
    def vectorizer(self):
        return self._vectorizer

    def fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)

    def page_comparison_parallel(self, page_df: pd.DataFrame) -> pd.DataFrame:
        num_processes = min(os.cpu_count(), 6)
        page_dict = page_df.set_index('page_num').to_dict('index')
        page_nums = list(page_df['page_num'].unique())
        batch_size = max(len(page_nums) // num_processes, 1)
        page_batches = [page_nums[i:i + batch_size] for i in range(0, len(page_nums), batch_size)]

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(self.process_page_batch, batch, page_dict, len(page_dict), (i == 0)) for i, batch in enumerate(page_batches)]
            results = [future.result() for future in as_completed(futures)]

        results_dict = {result[0]: result[1] for result_batch in results for result in result_batch}
        return pd.DataFrame.from_dict(results_dict, orient='index')

    def process_page_batch(self, page_nums: List[int], page_dict: Dict, total_pages: int, is_first_batch: bool) -> List[Tuple[int, Dict]]:
        return [self.compare_pages(page_num, page_dict, i, len(page_nums), total_pages, is_first_batch) for i, page_num in enumerate(page_nums)]

    def compare_pages(self, page_num: int, page_dict: Dict, current_index: int, batch_size: int, total_pages: int, is_first_batch: bool) -> Tuple[int, Dict]:
        if is_first_batch:
            percent_done = (current_index + 1) / batch_size * 100
            print(f"First batch progress: {percent_done:.2f}% ({current_index + 1}/{batch_size} pages processed)")

        page_data = page_dict[page_num]
        prev_page_data = page_dict.get(page_num - 1)
        next_page_data = page_dict.get(page_num + 1)

        results = {"page_num": page_num}
        results.update(self.process_adjacent_page(page_data, prev_page_data, 'prev'))
        results.update(self.process_adjacent_page(page_data, next_page_data, 'next'))

        return page_num, results

    def process_adjacent_page(self, current_page: Dict, adjacent_page: Dict, prefix: str) -> Dict:
        result = {}
        if adjacent_page:
            # Page dimensions
            result.update({
                f"{prefix}_page_width": adjacent_page["page_width"],
                f"{prefix}_page_height": adjacent_page["page_height"],
                f"{prefix}_page_width_diff": abs(current_page["page_width"] - adjacent_page["page_width"]),
                f"{prefix}_page_height_diff": abs(current_page["page_height"] - adjacent_page["page_height"]),
                f"{prefix}_page_vertical_margin_ratio": adjacent_page["vertical_margin_ratio"],
                f"{prefix}_page_horizontal_margin_ratio": adjacent_page["horizontal_margin_ratio"],
                f"{prefix}_page_vertical_margin_ratio_diff": abs(current_page["vertical_margin_ratio"] - adjacent_page["vertical_margin_ratio"]),
                f"{prefix}_page_horizontal_margin_ratio_diff": abs(current_page["horizontal_margin_ratio"] - adjacent_page["horizontal_margin_ratio"]),
                f"{prefix}_page_whitespace_ratio": adjacent_page["whitespace_ratio"],
                f"{prefix}_page_has_images_or_tables": adjacent_page["has_images_or_tables"],
                f"{prefix}_page_average_x_position": adjacent_page["average_x_position"],
                f"{prefix}_page_avg_x_first_10": adjacent_page["avg_x_first_10"],
                f"{prefix}_page_avg_x_last_10": adjacent_page["avg_x_last_10"],
                f"{prefix}_page_whitespace_ratio_diff": abs(current_page["whitespace_ratio"] - adjacent_page["whitespace_ratio"]),
                f"{prefix}_page_has_images_or_tables_diff": int(current_page["has_images_or_tables"] != adjacent_page["has_images_or_tables"]),
                f"{prefix}_page_average_x_position_diff": abs(current_page["average_x_position"] - adjacent_page["average_x_position"]),
                f"{prefix}_page_avg_x_first_10_diff": abs(current_page["avg_x_first_10"] - adjacent_page["avg_x_first_10"]),
                f"{prefix}_page_avg_x_last_10_diff": abs(current_page["avg_x_last_10"] - adjacent_page["avg_x_last_10"]),
                f"{prefix}_page_word_count": adjacent_page["word_count"],
                f"{prefix}_page_first_line_vertical_position": adjacent_page["first_line_vertical_position"],
                f"{prefix}_page_text_density": adjacent_page["text_density"],
                f"{prefix}_page_average_text_size": adjacent_page["average_text_size"],
                f"{prefix}_page_rows_of_text_count": adjacent_page["rows_of_text_count"],
                f"{prefix}_page_word_count_diff": abs(current_page["word_count"] - adjacent_page["word_count"]),
                f"{prefix}_page_first_line_vertical_position_diff": abs(current_page["first_line_vertical_position"] - adjacent_page["first_line_vertical_position"]),
                f"{prefix}_page_text_density_diff": abs(current_page["text_density"] - adjacent_page["text_density"]),
                f"{prefix}_page_average_text_size_diff": abs(current_page["average_text_size"] - adjacent_page["average_text_size"]),
                f"{prefix}_page_rows_of_text_count_diff": abs(current_page["rows_of_text_count"] - adjacent_page["rows_of_text_count"]),
                f"{prefix}_page_idf_text_similarity": self.calculate_idf_text_similarity(current_page["processed_full_text"], adjacent_page["processed_full_text"]),
                f"{prefix}_page_header_similarity": self.calculate_idf_text_similarity(" ".join(current_page["page_header"]), " ".join(adjacent_page["page_header"])),
                f"{prefix}_page_number_of_entities": adjacent_page["number_of_entities"],
                f"{prefix}_page_number_of_unique_entities": adjacent_page["number_of_unique_entities"],
                f"{prefix}_page_shared_entities": len(set(current_page["entities"]).intersection(set(adjacent_page["entities"]))),
                f"{prefix}_page_shared_entities_total": self.count_shared_entities(current_page["entities"], adjacent_page["entities"]),
                f"{prefix}_page_number_of_entities_diff": abs(current_page["number_of_entities"] - adjacent_page["number_of_entities"]),
                f"{prefix}_page_number_of_unique_entities_diff": abs(current_page["number_of_unique_entities"] - adjacent_page["number_of_unique_entities"]),
                f"{prefix}_page_shared_entities_diff": abs(current_page["number_of_entities"] - adjacent_page["number_of_entities"]),
                f"{prefix}_page_shared_entities_total_diff": abs(current_page["number_of_unique_entities"] - adjacent_page["number_of_unique_entities"]),
                f"{prefix}_page_has_page_1": self.check_page_1(adjacent_page["full_text"]),
                f"{prefix}_page_has_page_x": self.check_page_x(adjacent_page["full_text"]),
                f"{prefix}_page_has_page_x_of_x": self.check_page_x_of_x(adjacent_page["full_text"]),
                f"{prefix}_page_has_page_x_of_x_end": self.check_page_x_of_x_end(adjacent_page["full_text"])
            })
        else:
            # Handle no adjacent page case
            default_values = {
                "page_width": 1000,
                "page_height": 2000,
                "page_width_diff": 1000,
                "page_height_diff": 2000,
                "page_vertical_margin_ratio": 1.0,
                "page_horizontal_margin_ratio": 1.0,
                "page_vertical_margin_ratio_diff": 1.0,
                "page_horizontal_margin_ratio_diff": 1.0,
                "page_whitespace_ratio": 1.0,
                "page_has_images_or_tables": 1,
                "page_average_x_position": 1000,
                "page_avg_x_first_10": 1000,
                "page_avg_x_last_10": 1000,
                "page_whitespace_ratio_diff": 1.0,
                "page_has_images_or_tables_diff": 1,
                "page_average_x_position_diff": 1000,
                "page_avg_x_first_10_diff": 1000,
                "page_avg_x_last_10_diff": 1000,
                "page_word_count": 0,
                "page_first_line_vertical_position": 2000,
                "page_text_density": 0.0,
                "page_average_text_size": 0.0,
                "page_rows_of_text_count": 0,
                "page_word_count_diff": 1000,
                "page_first_line_vertical_position_diff": 2000,
                "page_text_density_diff": 1.0,
                "page_average_text_size_diff": 1.0,
                "page_rows_of_text_count_diff": 100,
                "page_idf_text_similarity": 0.0,
                "page_header_similarity": 0.0,
                "page_number_of_entities": 0,
                "page_number_of_unique_entities": 0,
                "page_shared_entities": 0,
                "page_shared_entities_total": 0,
                "page_number_of_entities_diff": 0,
                "page_number_of_unique_entities_diff": 0,
                "page_shared_entities_diff": 0,
                "page_shared_entities_total_diff": 0,
                "page_has_page_1": 0,
                "page_has_page_x": 0,
                "page_has_page_x_of_x": 0,
                "page_has_page_x_of_x_end": 0
            }
            result.update({f"{prefix}_{k}": v for k, v in default_values.items()})

        return result

    def calculate_idf_text_similarity(self, text1: str, text2: str) -> float:
        vector1 = self.vectorizer.transform([text1])
        vector2 = self.vectorizer.transform([text2])
        cosine_similarities = cosine_similarity(vector1, vector2)
        return cosine_similarities[0, 0]

    def count_shared_entities(self, entities1: List[Tuple[str, str]], entities2: List[Tuple[str, str]]) -> int:
        count1 = Counter(entities1)
        count2 = Counter(entities2)
        shared_entities = count1.keys() & count2.keys()
        total_shared_entities = sum((count1[entity] + count2[entity]) for entity in shared_entities)
        return total_shared_entities

    def check_page_1(self, full_text: str) -> int:
        pattern = r'page\s+1'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x(self, full_text: str) -> int:
        pattern = r'page\s+\d+'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x_of_x(self, full_text: str) -> int:
        pattern = r'page\s+\d+\s+of\s+\d+'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0

    def check_page_x_of_x_end(self, full_text: str) -> int:
        pattern = r'page\s+(\d+)\s+of\s+\1'
        return 1 if re.search(pattern, full_text, re.IGNORECASE) else 0
