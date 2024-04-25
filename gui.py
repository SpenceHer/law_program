import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pdf_processor import PDFProcessor
from text_analysis import TextAnalyzer
from image_analysis import ImageAnalyzer
import threading
import pandas as pd
import numpy as np

class GUI:
    def __init__(self):
        self.main_window = tk.Tk()
        self.configure_window()
        self.create_widgets()

    def run(self):
        self.main_window.mainloop()

    def configure_window(self):
        screen_width, screen_height = self.main_window.winfo_screenwidth() // 2, self.main_window.winfo_screenheight() // 2
        self.main_window.title("PDF Processor")
        self.main_window.geometry(f"{screen_width}x{screen_height}+0+0")
        self.style = ttk.Style()
        self.style.theme_use("clam")

    def create_widgets(self):
        # Main frame for PDF reading functionality
        pdf_reader_frame = tk.Frame(self.main_window, bg='lightgray')
        pdf_reader_frame.pack(side="top", fill=tk.BOTH, expand=True)

        # Style configuration for buttons
        button_style = "action_button.TButton"
        style = ttk.Style()
        style.configure(button_style, background='darkblue', foreground='white',
                        font=("Helvetica", 36), padding=10)

        # File selection frame
        file_selection_frame = tk.Frame(pdf_reader_frame, bg='white')
        file_selection_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        file_selection_button = ttk.Button(file_selection_frame, text="Choose PDF File", command=self.select_file, style=button_style)
        file_selection_button.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=20)
        self.current_file_label = tk.Label(file_selection_frame, text="No file selected", font=("Arial", 14), bg='white')
        self.current_file_label.pack(side=tk.TOP, fill=tk.X)

        # Save destination frame
        save_destination_frame = tk.Frame(pdf_reader_frame, bg='white')
        save_destination_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        select_destination_button = ttk.Button(save_destination_frame, text="Select Save Destination", command=self.choose_save_destination, style=button_style)
        select_destination_button.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=20)
        self.save_destination_label = tk.Label(save_destination_frame, text="No destination selected", font=("Arial", 14), bg='white')
        self.save_destination_label.pack(side=tk.TOP, fill=tk.X)

        # Separate PDF frame
        separate_pdf_frame = tk.Frame(pdf_reader_frame, bg='white')
        separate_pdf_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        separate_pdf_button = ttk.Button(separate_pdf_frame, text="Separate PDF File", command=self.initiate_pdf_separation, state=tk.DISABLED, style=button_style)
        separate_pdf_button.pack(side=tk.TOP, fill=tk.BOTH, padx=50, pady=20)
        self.status_label = tk.Label(separate_pdf_frame, text="Ready", font=("Arial", 16), bg='white')
        self.status_label.pack(side=tk.TOP, fill=tk.X)

    def select_file(self):
        new_file_path = filedialog.askopenfilename(title="Select a PDF file", filetypes=[("PDF files", "*.pdf")], initialdir="/")
        if new_file_path:
            self.file_path = new_file_path
            self.current_file_label.configure(text=f"Current File: {os.path.basename(self.file_path)}")
            self.update_button_state()

    def choose_save_destination(self):
        directory = filedialog.askdirectory()
        if directory:
            self.save_destination = directory
            self.save_destination_label.config(text=f"Save to: {directory}")
            self.update_button_state()

    def update_button_state(self):
        if self.file_path and self.save_destination:
            self.separate_pdf_button.configure(state=tk.ACTIVE)
        else:
            self.separate_pdf_button.configure(state=tk.DISABLED)

    def initiate_pdf_separation(self):
        if hasattr(self, 'file_path'):
            threading.Thread(target=self.run_analysis).start()
        else:
            messagebox.showerror("Error", "Please load a PDF first.")


    def run_analysis(self):
        self.status_label.configure(text="Preparing application...")
        pdf_processor = PDFProcessor(self.file_path, "test.json")
        text_analyzer = TextAnalyzer()
        image_analyzer = ImageAnalyzer()

        self.status_label.configure(text="Converting PDF pages to images...")
        images = pdf_processor.convert_pdf_to_images()
        results = []

        self.status_label.configure(text="Extracting image information...")
        for i, image in enumerate(images, start=1):
            text = pdf_processor.extract_text(image)
            results.append((i, text))

    def extract_data(self, pdf_processor, page_num, image):
        page_data = {}
        
        # Text information
        text = pdf_processor.extract_text(image)
        page_data["full_text"] = text
        page_data["text_rows"] = text.split("\n")
        









