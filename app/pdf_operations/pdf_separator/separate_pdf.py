import os
import joblib
from PyPDF2 import PdfReader, PdfWriter



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
        model = joblib.load(os.path.join(self.BASE_DIR, 'pdf_operations', 'pdf_separator', 'models', 'initial_model.pkl'))
        pipeline = joblib.load(os.path.join(self.BASE_DIR, 'pdf_operations', 'pdf_separator', 'models', 'preprocessing_pipeline.pkl'))
        # important_features_path = os.path.join(self.BASE_DIR, 'pdf_operations', 'pdf_separator', 'models', 'important_features_list.txt')
        # with open(important_features_path, "r") as file:
        #     important_features = [line.strip() for line in file]

        # X_new = self.page_df[important_features]
        X_new = self.page_df[self.features]

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