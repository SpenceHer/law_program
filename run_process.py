from pdf_processor import PageComparison, ProcessPDF
import pandas as pd
import time

class RunProcess:
    def __init__(self):
        self.select_file()
        self.choose_save_destination()
        
        self.start_pdf_separation()

    def select_file(self):
        # new_file_path = filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")], initialdir="/")
        new_file_path = "/Users/spencersmith/Desktop/CODING/Projects/law/PDFs/large/Produced Docs.pdf" # SMALL FULL
        # new_file_path = "/Users/spencersmith/Desktop/CODING/Projects/law/PDFs/Test Files - Discovery Software-2/Production 5.23.22 BATES.pdf" # LARGE 4500

        # new_file_path = "/Users/spencersmith/Desktop/CODING/Projects/law/PDFs/small/Binder2.pdf" # SMALL FULL
        # new_file_path = "/Users/spencersmith/Desktop/CODING/Projects/law/PDFs/small/separated/WHITECAP0001000.pdf" # SMALL SINGLE

        if new_file_path:
            self.file_path = new_file_path
            print(f"File Path: {new_file_path}")

    def choose_save_destination(self):
        # directory = filedialog.askdirectory()
        directory = "/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes"
        if directory:
            self.save_destination = directory
            print(f"Save Destination: {directory}")





    def initialize_classes(self):
        print("Preparing application...")
        self.pdf_processor = ProcessPDF(self.file_path)
        # self.page_converter = PageToImageConversion(self.file_path)
        # self.text_extractor = TextExtraction()
        self.page_comparison = PageComparison()



    def start_pdf_separation(self):

        start_overall_time = time.time()
        self.initialize_classes()


        data_extraction_start_time = time.time()
        # Extract text with Google Vision
        print("Preparing to extract data...")
        page_df, total_pages = self.pdf_processor.process_pdf()
        sorted_page_df = page_df.sort_values(by='page_num', ascending=True)
        data_extraction_end_time = time.time()  # End the timer
        data_extraction_elapsed_time = round((data_extraction_end_time - data_extraction_start_time)/60, 2)  # Calculate elapsed time
        print(f"Elapsed Extraction time: {data_extraction_elapsed_time} Minutes")

        # start_time = time.time()
        # # Convert PDF pages to images
        # print("Converting PDF pages to images...")
        # images = self.page_converter.convert_pdf_to_images_parallel()
        # end_time = time.time()  # End the timer
        # elapsed_time = end_time - start_time  # Calculate elapsed time
        # print(f"Elapsed Conversion time: {elapsed_time} seconds")


        # start_time = time.time()
        # # Extract text with Google Vision
        # print("Preparing to extract text...")
        # page_df = self.text_extractor.extract_text_parallel(images)
        # end_time = time.time()  # End the timer
        # elapsed_time = end_time - start_time  # Calculate elapsed time
        # print(f"Elapsed Extraction time: {elapsed_time} seconds")


        # # Compare pages
        compare_pages_start_time = time.time()
        print("Comparing pages...")
        self.page_comparison.fit_vectorizer(sorted_page_df['full_text'].tolist())
        new_df = self.page_comparison.page_comparison_parallel(sorted_page_df)
        sorted_new_df = new_df.sort_values(by='page_num', ascending=True)
        compare_pages_end_time = time.time()  # End the timer
        compare_pages_elapsed_time = round((compare_pages_end_time - compare_pages_start_time)/60, 2)
        print(f"Elapsed Comparison time: {compare_pages_elapsed_time} Minutes")

        
        # # Merge dataframes
        merged_df = pd.merge(sorted_page_df, sorted_new_df, on='page_num')


        # Save dataframes
        merged_df.to_csv(f"{self.save_destination}/law_page_data_{total_pages}.csv", index=False)

        print(merged_df)
                
        end_overall_time = time.time()  # End the overall timer
        overall_elapsed_time = round((end_overall_time - start_overall_time)/60, 2)




        print(f"\nElapsed Extraction time: {data_extraction_elapsed_time} Minutes")
        print(f"Elapsed Comparison time: {compare_pages_elapsed_time} Minutes")
        print(f"Overall elapsed time: {overall_elapsed_time} Minutes\n")

        print(f"Data extraction pages per minute: {total_pages/data_extraction_elapsed_time}")
        print(f"Comparison pages per minute: {total_pages/compare_pages_elapsed_time}")
        print(f"Overall pages per minute: {total_pages/overall_elapsed_time}")



