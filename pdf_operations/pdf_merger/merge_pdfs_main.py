import os
from PyPDF2 import PdfReader, PdfWriter

def merge_pdfs(pdf_list, output_folder='static/uploads'):
    merger = PdfWriter()
    for pdf in pdf_list:
        reader = PdfReader(pdf)
        for page in reader.pages:
            merger.add_page(page)
    merged_pdf_path = os.path.join(output_folder, 'merged.pdf')
    with open(merged_pdf_path, 'wb') as merged_pdf:
        merger.write(merged_pdf)
    return merged_pdf_path
