from pdf2image import convert_from_path
import pytesseract
import re
import PyPDF2
from datetime import datetime

# Path to the PDF file you want to extract text from
input_pdf_file = '/Users/spencersmith/Desktop/coding/python/law/old/Binder2.pdf'
output_path = '/Users/spencersmith/Desktop/coding/python/law/new/test'

class SeparateDocumentClass:
    def __init__(self, input_pdf_file, output_path):
        # Convert PDF pages to images
        self.input_pdf_file = input_pdf_file
        self.output_path = output_path

        self.pdf_document = convert_from_path(self.input_pdf_file)

        # Initialize Tesseract
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
        self.page_texts = {}
        self.new_pdf_start_page_dict = {}
        


        self.transform_pages_to_text()

        self.get_new_pdf_start_pages()

        self.separate_pdf()


    def transform_pages_to_text(self):
        for page_num, page_image in enumerate(self.pdf_document):
            text = pytesseract.image_to_string(page_image, lang='eng')

            self.page_texts[page_num] = text

            # TRACK PAGES REMAINING
            print(page_num)

    def get_new_pdf_start_pages(self):
        
        for page_num, text in self.page_texts.items():
            self.define_page_start_conditions(page_num, text)






    def define_page_start_conditions(self, page_num, text):
        page_start_email_conditions = []
        page_start_document_conditions = []

        page_lines = text.split('\n')
        page_lines = list(filter(lambda item: item != "", page_lines))
        page_lines = [s.lower() for s in page_lines]

        print("\n\n")
        print(page_lines)

        page = text.lower()

        # DEFINE CONDITIONS AND ADD TO LIST OF CONDITIONS THAT DEFINE THE START OF A NEW PDF
        email_condition_1 = 'message' in page_lines[0] and 'from' in page_lines[1] and 'sent' in page_lines[2]
        page_start_email_conditions.append(email_condition_1)

        email_condition_2 = 'from' in page_lines[0] and 'subject' in page_lines[1] and 'date' in page_lines[2]
        page_start_email_conditions.append(email_condition_2)

        document_condition_1 = 'memorandum of understanding/sublease' in page and 'page' in page
        page_start_document_conditions.append(document_condition_1)

        document_condition_2 = 'page 1' in page_lines[0] and 'insurance proposal' in page
        page_start_document_conditions.append(document_condition_2)

        document_condition_3 = 'page 1' in page_lines[0] and 'insurance policy proposal' in page
        page_start_document_conditions.append(document_condition_3)

        document_condition_4 = 'page 1' in page and 'this is a premium estimate.' in page
        page_start_document_conditions.append(document_condition_4)

        if any(page_start_email_conditions):
            print(page_num)
            self.addto_file_name_dict('email', page_lines, page_num)



        elif any(page_start_document_conditions):

            self.addto_file_name_dict('document', page_lines, page_num)




    def addto_file_name_dict(self, pdf_type, page_lines, page_num):

        if pdf_type == 'email':
            
            for line in page_lines[0:9]:
                # try:
                print(1)
                print(line)
                if 'sent' in line or 'date' in line:
                    if 'sent' in line:
                        print(2)
                        pattern = r'\d{1,2}/\d{1,2}/\d{4}'  
                        match = re.search(pattern, line)
                        print(match)
                        if match:
                            date = match.group().replace("/", "_")
                            file_name = f"{self.output_path}/email_{date}.pdf"
                            self.new_pdf_start_page_dict[page_num] = file_name
                            print(3)
                            break
                        else:
                            file_name = f"{self.output_path}/email_unknown_date.pdf"
                            self.new_pdf_start_page_dict[page_num] = file_name

                    else:
                        pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}'
                        match = re.search(pattern, line)
                        if match:
                            date = match.group()
                            date_format = "%B %d, %Y"
                            date_datetime = datetime.strptime(date, date_format)
                            
                            # Format the datetime object as "mm/dd/yyyy"
                            formatted_date = date_datetime.strftime("%m/%d/%Y").replace("/", "_")
                            file_name = f"{self.output_path}/email_{formatted_date}.pdf"
                            self.new_pdf_start_page_dict[page_num] = file_name

                        else:
                            file_name = f"{self.output_path}/email_unknown_date.pdf"
                            self.new_pdf_start_page_dict[page_num] = file_name

                # except:
                #     print(2)
                #     file_name = f"{self.output_path}/email_unknown_date.pdf"
                #     self.new_pdf_start_page_dict[page_num] = file_name

        if pdf_type == 'document':
            file_name = f"{self.output_path}/document.pdf"
            self.new_pdf_start_page_dict[page_num] = file_name




    def separate_pdf(self):
        print(self.new_pdf_start_page_dict)
        with open(self.input_pdf_file, "rb") as original_pdf:
            pdf_reader = PyPDF2.PdfReader(original_pdf)
            length_of_original_pdf = len(pdf_reader.pages)
            keys = list(self.new_pdf_start_page_dict.keys())
            print(length_of_original_pdf)
            # Iterate through the dictionary and create new PDFs
            for idx, (start_page, new_pdf_name) in enumerate(sorted(self.new_pdf_start_page_dict.items())):
                # Create a new PDF writer for the current sub-PDF
                pdf_writer = PyPDF2.PdfWriter()
                print(idx)

                if idx < len(self.new_pdf_start_page_dict):
                    end_key_position = keys.index(start_page) + 1
                    end_page = keys[end_key_position]
                else:
                    end_page = length_of_original_pdf

                for page_num in range(start_page, end_page):
                    pdf_writer.add_page(pdf_reader.pages[page_num])

                with open(new_pdf_name, 'wb') as output_pdf:
                    pdf_writer.write(output_pdf)






SeparateDocumentClass(input_pdf_file, output_path)