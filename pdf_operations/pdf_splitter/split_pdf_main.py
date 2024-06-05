from PyPDF2 import PdfReader, PdfWriter
import os

def split_pdf(file_path, split_points):
    input_pdf = PdfReader(file_path)
    output_files = []
    last_page = 0

    for i, split_point in enumerate(split_points + [len(input_pdf.pages)]):
        output = PdfWriter()
        for j in range(last_page, split_point):
            output.add_page(input_pdf.pages[j])

        output_filename = f'split_part_{i + 1}.pdf'
        output_file_path = os.path.join(os.path.dirname(file_path), output_filename)
        with open(output_file_path, 'wb') as output_file:
            output.write(output_file)
        output_files.append(output_file_path)
        last_page = split_point

    return output_files
