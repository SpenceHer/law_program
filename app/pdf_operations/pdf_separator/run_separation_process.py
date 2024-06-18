from pdf_operations.pdf_separator import data_extractor, process_pages, separate_pdf
import pandas as pd
import time


class RunSeparationProcess:
    def __init__(self, file_path):
        self.file_path = file_path
        self.output_files = self.start_pdf_separation()

    def initialize_classes(self):
        print("Preparing application...")
        self.data_extractor = data_extractor.DataExtractor(self.file_path)
        self.page_processor = process_pages.ProcessPages()
       

    def start_pdf_separation(self):
        start_overall_time = time.time()
        self.initialize_classes()

        # Extract data from the PDF
        data_extraction_start_time = time.time()
        print("Preparing to extract data...")
        page_df, total_pages = self.data_extractor.extract_data()
        sorted_page_df = page_df.sort_values(by='page_num', ascending=True)
        data_extraction_end_time = time.time()
        data_extraction_elapsed_time = round((data_extraction_end_time - data_extraction_start_time) / 60, 2)
        print(f"Elapsed Extraction time: {data_extraction_elapsed_time} Minutes")

        # Compare the pages
        compare_pages_start_time = time.time()
        print("Comparing pages...")
        self.page_processor.fit_vectorizer(sorted_page_df['processed_full_text'].tolist())
        new_df = self.page_processor.page_comparison_parallel(sorted_page_df)
        sorted_new_df = new_df.sort_values(by='page_num', ascending=True)
        compare_pages_end_time = time.time()
        compare_pages_elapsed_time = round((compare_pages_end_time - compare_pages_start_time) / 60, 2)
        print(f"Elapsed Comparison time: {compare_pages_elapsed_time} Minutes")

        # Merge the two dataframes
        merged_df = pd.merge(sorted_page_df, sorted_new_df, on='page_num')
        
        # merged_df.to_csv(f"/Users/spencersmith/Desktop/CODING/Projects/law/code/app/pdf_operations/pdf_separator/testing/dataframes/dataframe_{total_pages}.csv", index=False)

        # Separate the PDF
        separate_pages_start_time = time.time()
        print("Separating PDF...")
        output_files = separate_pdf.PDFSeparation(self.file_path, merged_df).output_files
        separate_pages_end_time = time.time()
        separate_pages_elapsed_time = round((separate_pages_end_time - separate_pages_start_time) / 60, 2)
        print(f"Elapsed Separation time: {separate_pages_elapsed_time} Minutes")


        end_overall_time = time.time()
        overall_elapsed_time = round((end_overall_time - start_overall_time) / 60, 2)

        print(f"\nElapsed Extraction time: {data_extraction_elapsed_time} Minutes")
        print(f"Elapsed Comparison time: {compare_pages_elapsed_time} Minutes")
        print(f"Overall elapsed time: {overall_elapsed_time} Minutes\n")

        print(f"Data extraction pages per minute: {total_pages / data_extraction_elapsed_time}")
        print(f"Comparison pages per minute: {total_pages / compare_pages_elapsed_time}")
        print(f"Overall pages per minute: {total_pages / overall_elapsed_time}")

        return output_files

