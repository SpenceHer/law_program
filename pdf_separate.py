import io
import os
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import gensim
from gensim import corpora
from google.cloud import vision
from google.oauth2 import service_account
import numpy as np
import pandas as pd
import pickle
from PIL import Image, ImageStat
import pytesseract
from pdf2image import convert_from_path
import spacy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import styles
from styles import color_dict
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer








class SeparatePdfClass:
    def __init__(self):
        self.setup_credentials()
        self.setup_gui()

    def setup_credentials(self):
        self.nlp = spacy.load("en_core_web_sm")

        # Ensure NLTK resources are available
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Load pre-trained BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.tf_vectorizer = TfidfVectorizer()

        credentials = service_account.Credentials.from_service_account_file(
            '/Users/spencersmith/Desktop/CODING/Projects/law/code/cloud_credentials/lunar-tube-421021-a963c429b915.json'
            )
        self.client = vision.ImageAnnotatorClient(credentials=credentials)

    def setup_gui(self):
        self.main_window = tk.Tk()
        self.configure_window()
        self.create_widgets()
        self.main_window.mainloop()

    def configure_window(self):
        screen_width, screen_height = self.main_window.winfo_screenwidth() // 2, self.main_window.winfo_screenheight() // 2
        self.main_window.title("PDF Processor")
        self.main_window.geometry(f"{screen_width}x{screen_height}+0+0")
        self.style = ttk.Style()
        self.style.theme_use("clam")



    def create_widgets(self):
        self.file_path = None
        self.save_destination = None

        # Main frame for PDF reading functionality
        pdf_reader_frame = tk.Frame(self.main_window, bg=color_dict["frame_background"])
        pdf_reader_frame.pack(side="top", fill=tk.BOTH, expand=True)

        # Style configuration for buttons
        button_style = "action_button.TButton"
        style = ttk.Style()
        style.configure(button_style, background=color_dict["button_background"], foreground=color_dict["button_foreground"],
                        font=("Helvetica", 36), padding=10)

        # File selection frame
        file_selection_frame = tk.Frame(pdf_reader_frame, bg=color_dict["background_color"])
        file_selection_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        file_selection_button = ttk.Button(file_selection_frame, text="Choose PDF File", command=self.select_file, style=button_style)
        file_selection_button.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=20)
        self.current_file_label = tk.Label(file_selection_frame, text="No file selected", font=("Arial", 14), bg=color_dict["background_color"], fg=color_dict["text_color"])
        self.current_file_label.pack(side=tk.TOP, fill=tk.X)

        # Save destination frame
        save_destination_frame = tk.Frame(pdf_reader_frame, bg=color_dict["background_color"])
        save_destination_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        select_destination_button = ttk.Button(save_destination_frame, text="Select Save Destination", command=self.choose_save_destination, style=button_style)
        select_destination_button.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=20)
        self.save_destination_label = tk.Label(save_destination_frame, text="No destination selected", font=("Arial", 14), bg=color_dict["background_color"], fg=color_dict["text_color"])
        self.save_destination_label.pack(side=tk.TOP, fill=tk.X)

        # Separate PDF frame
        separate_pdf_frame = tk.Frame(pdf_reader_frame, bg=color_dict["background_color"])
        separate_pdf_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.separate_pdf_button = ttk.Button(separate_pdf_frame, text="Separate PDF File", command=self.initiate_pdf_separation_thread, state=tk.DISABLED, style=button_style)
        self.separate_pdf_button.pack(side=tk.TOP, fill=tk.BOTH, padx=50, pady=20)
        self.status_label = tk.Label(separate_pdf_frame, text="Ready", font=("Arial", 16), bg=color_dict["background_color"], fg=color_dict["text_color"])
        self.status_label.pack(side=tk.TOP, fill=tk.X)

    def select_file(self):
        new_file_path = filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")], initialdir="/")
        if new_file_path:
            self.file_path = new_file_path
            self.current_file_label.configure(text=f"Current File: {os.path.basename(self.file_path)}")
            self.update_button_state()

    def choose_save_destination(self):
        # directory = filedialog.askdirectory()
        directory = "/Users/spencersmith/Desktop/CODING/Projects/law/code/dictionary_folder"
        if directory:
            self.save_destination = directory
            self.save_destination_label.config(text=directory)
            self.update_button_state()

    def update_button_state(self):
        # Enable the separate button only if both file and destination are selected
        if self.file_path and self.save_destination:
            self.separate_pdf_button.configure(state=tk.ACTIVE)
        else:
            self.separate_pdf_button.configure(state=tk.DISABLED)

        

    def initiate_pdf_separation_thread(self):
        thread = threading.Thread(target=self.separate_pdf)
        thread.start()
        print(f"\n\n\nFILE: {self.file_path}")
 

    def separate_pdf(self):
        # try:
            self.update_status_label("Converting PDF to images...")
            self.images = convert_from_path(self.file_path)
            self.extract_page_information()
            self.analyze_page_information()
        # except Exception as e:
        #     self.update_status_label("Failed to process PDF.")
        #     print(f"Error: {str(e)}")


    def update_status_label(self, status_text):
        def update_label():
            self.status_label.configure(text=status_text)
            self.status_label.update_idletasks()

        # Schedule the update_label function to run on the main GUI thread
        self.main_window.after(0, update_label)

####################################################################

    def extract_page_information(self):
        total_pages = len(self.images)
        self.page_attributes = {}
        self.page_text = {}
        self.page_text_rows = {}
        self.page_text_words = {}

        def process_page(image, page_number):
            # try:
                self.update_status_label(f"Processing page {page_number} of {total_pages}")

                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                vision_image = vision.Image(content=img_byte_arr)
                response = self.client.text_detection(image=vision_image)

                texts = response.text_annotations

                full_text = texts[0].description if texts else ''

                rows = full_text.split('\n')
                words = [text.description for text in texts[1:]]
                filtered_words = [word for word in words if re.search(r'[a-zA-Z0-9]', word)]
                processed_image = self.preprocess_image(np.array(image))
                page_attributes = self.get_page_attributes(processed_image, image, full_text, texts, rows, filtered_words)
                return page_number, page_attributes, full_text, rows, filtered_words
            
            # except Exception as e:
            #     print(f"Failed to process page {page_number}: {str(e)}")
            #     return page_number, {}, '', [], []

        with ThreadPoolExecutor(max_workers=6) as executor:
            results = executor.map(process_page, self.images, range(1, total_pages + 1))

        for result in results:
            page_number, attributes, text, rows, words = result
            self.page_attributes[page_number] = attributes
            self.page_text[page_number] = text
            self.page_text_rows[page_number] = rows
            self.page_text_words[page_number] = words

        self.update_status_label("Done processing")







####################################################################







    def analyze_page_information(self):

        attributes_df = pd.DataFrame.from_dict(self.page_attributes, orient='index').reset_index().rename(columns={"index":"page_num"})
        full_text_df = pd.DataFrame.from_dict(self.page_text, orient='index').reset_index().rename(columns={"index":"page_num"}).rename(columns={0:"page_text"})

        page_text_rows_formatted = [(key, value) for key, value in self.page_text_rows.items()]
        page_text_rows_df = pd.DataFrame(page_text_rows_formatted, columns=['page_num', 'page_rows'])

        page_text_words_formatted = [(key, value) for key, value in self.page_text_words.items()]
        page_text_words_df = pd.DataFrame(page_text_words_formatted, columns=['page_num', 'page_words'])

        # Merging attributes_df and full_text_df
        merged_df = pd.merge(attributes_df, full_text_df, on='page_num')

        # Merging page_text_rows_df to the already merged DataFrame
        merged_df = pd.merge(merged_df, page_text_rows_df, on='page_num')

        # Merging page_text_words_df to the already merged DataFrame
        self.page_df = pd.merge(merged_df, page_text_words_df, on='page_num')
        self.page_df.set_index('page_num', inplace=True, drop=False)


        total_pages = len(self.page_df)

        def analyze_page_text(page_number):
            # try:
                self.update_status_label(f"Analyzing page text {page_number} of {total_pages}")
                page_text = self.page_df.loc[self.page_df['page_num'] == page_number, "page_text"].values[0]

                text_sentiment_polarity, text_sentiment_subjectivity = self.analyze_sentiment(page_text)
                entities = self.extract_entities(page_text)
                prev_page_bert_similarity, prev_page_tf_similarity = self.extract_previous_page_text_diff(page_number)
                next_page_bert_similarity, next_page_tf_similarity = self.extract_next_page_text_diff(page_number)

                return page_number, text_sentiment_polarity, text_sentiment_subjectivity, entities, prev_page_bert_similarity, next_page_bert_similarity, prev_page_tf_similarity, next_page_tf_similarity
        


            # except Exception as e:
            #     print(f"Error updating DataFrame for page {page_number}: {str(e)}")

            

        with ThreadPoolExecutor(max_workers=6) as executor:
            results = executor.map(analyze_page_text, range(1, total_pages + 1))

        for result in results:
            page_number, text_sentiment_polarity, text_sentiment_subjectivity, entities, prev_page_bert_similarity, next_page_bert_similarity, prev_page_tf_similarity, next_page_tf_similarity = result

            self.page_df.loc[page_number, "text_sentiment_polarity"] = text_sentiment_polarity
            self.page_df.loc[page_number, "text_sentiment_subjectivity"] = text_sentiment_subjectivity
            self.page_df.loc[page_number, "entities"] = entities
            self.page_df.loc[page_number, "prev_page_bert_similarity"] = prev_page_bert_similarity
            self.page_df.loc[page_number, "next_page_bert_similarity"] = next_page_bert_similarity
            self.page_df.loc[page_number, "prev_page_tf_similarity"] = prev_page_tf_similarity
            self.page_df.loc[page_number, "next_page_tf_similarity"] = next_page_tf_similarity

        self.update_status_label("Done analyzing text")
        self.page_df.to_csv("/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/page_dataframe.csv")


    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # noise_removed = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # processed_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(thresh)


    def get_page_attributes(self, processed_image, original_image, page_text, texts, rows, filtered_words):
        image_np = np.array(original_image)

        return {
            'brightness': self.calculate_brightness(image_np),
            'contrast': self.calculate_contrast(image_np),
            'edge_density': self.calculate_edge_density(image_np),
            'dominant_color': self.calculate_dominant_color(image_np),
            'text_density': self.calculate_text_density(page_text, original_image),
            'page_size': self.get_page_size(original_image),
            'orientation': self.get_orientation(original_image),
            'image_count': self.count_images(image_np),
            'blurriness':self.calculate_blurriness(image_np),
            'noise':self.calculate_noise(image_np),
            'color_variance':self.calculate_color_variance(image_np),
            'text_blocks':len(texts) - 1,
            'layout_analysis': self.analyze_graphical_layout(image_np),
            'word_count':len(filtered_words),
            'rows_of_text_count':len(rows),
            'unique_colors':self.calculate_unique_color_count(image_np)
        }

    def calculate_unique_color_count(self, image):
        pixels = image.reshape(-1, image.shape[2])
        unique_colors = np.unique(pixels, axis=0)
        return len(unique_colors)

    # Function to encode text into embeddings
    def get_bert_embedding(self, text):
        # Encode text
        encoded_input = self.bert_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.bert_model(**encoded_input)
        # Pool the outputs into a single mean vector
        embeddings = model_output.last_hidden_state.mean(dim=1).squeeze()  # Ensure it's 1-D
        return embeddings

    def extract_previous_page_text_diff(self, page_number):
        if page_number == 1:
            return 0, 0
        else:
            prev_page_text = self.page_df.loc[page_number-1, "page_text"]
            current_page_text = self.page_df.loc[page_number, "page_text"]

            # TF-IDF Analysis
            texts = [current_page_text, prev_page_text]
            self.tf_vectorizer.fit(texts)

            vector1 = self.tf_vectorizer.transform([prev_page_text])
            vector2 = self.tf_vectorizer.transform([current_page_text])

            cosine_similarities = cosine_similarity(vector1, vector2)
            tf_similarity = cosine_similarities[0, 0]


            # BERT Analysis
            embedding1 = self.get_bert_embedding(prev_page_text)
            embedding2 = self.get_bert_embedding(current_page_text)

            bert_similarity = 1 - cosine(embedding1.numpy(), embedding2.numpy())

            return round(bert_similarity, 4), round(tf_similarity, 4)


    def extract_next_page_text_diff(self, page_number):
        if page_number == len(self.page_df):
            return 0, 0
        else:
            next_page_text = self.page_df.loc[page_number+1, "page_text"]
            current_page_text = self.page_df.loc[page_number, "page_text"]

            # TF-IDF Analysis
            texts = [current_page_text, next_page_text]
            self.tf_vectorizer.fit(texts)

            vector1 = self.tf_vectorizer.transform([next_page_text])
            vector2 = self.tf_vectorizer.transform([current_page_text])

            cosine_similarities = cosine_similarity(vector1, vector2)
            tf_similarity = cosine_similarities[0, 0]


            # BERT Analysis
            embedding1 = self.get_bert_embedding(next_page_text)
            embedding2 = self.get_bert_embedding(current_page_text)

            bert_similarity = 1 - cosine(embedding1.numpy(), embedding2.numpy())

            return round(bert_similarity, 4), round(tf_similarity, 4)



    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        string_entities = str(entities)
        return string_entities
        
    def analyze_sentiment(self, text):
        sentiment = TextBlob(text).sentiment
        return sentiment.polarity, sentiment.subjectivity

    def calculate_noise(self, image_np):
        # Convert image to grayscale if it is not already
        if len(image_np.shape) == 3:  # Check if the image is colored (3 channels)
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np  # It's already grayscale

        # Calculate the standard deviation of the grayscale image
        mean, std_dev = cv2.meanStdDev(gray)
        noise_level = std_dev[0][0]

        return noise_level


    def calculate_color_variance(self, image):
        # Calculate variance across all color channels
        variance = np.var(image / 255.0)
        return variance


    def calculate_text_density(self, text, image):
        text_area = len(text) / (image.width * image.height)
        return text_area


    def calculate_blurriness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_estimate = cv2.Laplacian(gray, cv2.CV_64F).var()
        return noise_estimate


    def get_page_size(self, image):
        return image.size  # Returns (width, height)


    def get_orientation(self, image):
        width, height = image.size
        return 'Portrait' if height > width else 'Landscape'


    def count_images(self, image):
        # Simple approach using thresholding to find large enough unique objects
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([cnt for cnt in contours if cv2.contourArea(cnt) > 1000])  # Filter tiny contours


    # Function to calculate brightness
    def calculate_brightness(self, image):
        img = Image.fromarray(image)
        stat = ImageStat.Stat(img)
        return stat.mean[0]  # Average of R or grayscale


    # Function to calculate contrast
    def calculate_contrast(self, image):
        img = Image.fromarray(image)
        stat = ImageStat.Stat(img)
        return stat.stddev[0]  # Standard deviation of R or grayscale


    # Function to calculate edge density using Sobel filter
    def calculate_edge_density(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edge_img = np.hypot(sobelx, sobely)
        edge_mean = np.mean(edge_img)
        return edge_mean

    # Function to calculate dominant color
    def calculate_dominant_color(self, image):
        pixels = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, labels, centroids = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return centroids[0].astype(int)

    # Function to detect graphical layout
    def analyze_graphical_layout(self, img_array):
        # Analyzing margins by finding non-white space around the edges
        vertical_white_space = np.sum(np.all(img_array == 255, axis=(1, 2)))
        horizontal_white_space = np.sum(np.all(img_array == 255, axis=(0, 2)))
        vertical_margin = vertical_white_space / img_array.shape[0]
        horizontal_margin = horizontal_white_space / img_array.shape[1]
        return {'vertical_margin_ratio': vertical_margin, 'horizontal_margin_ratio': horizontal_margin}




if __name__ == "__main__":
    SeparatePdfClass()
