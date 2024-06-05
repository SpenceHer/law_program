import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model and scaler
best_model = joblib.load('best_xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the new data
new_data = pd.read_csv(r"/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/dataframe_475.csv")


# Ensure the new data has the same features
features = [
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
    "next_page_bert_text_similarity",
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
    "prev_page_bert_text_similarity",
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

# Select the features
X_new = new_data[features]

# Scale the new data
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = best_model.predict(X_new_scaled)

# Add predictions to the new data
new_data['predictions'] = predictions

# Save the predictions
new_data.to_csv('predictions.csv', index=False)
























import os
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter

# Load the predictions
predictions_df = pd.read_csv('predictions.csv')

# Load the original PDF
input_pdf_path = "/Users/spencersmith/Desktop/CODING/Projects/law/PDFs/Test Files - Discovery Software-2/Carebridge Compiled Bates.pdf"
reader = PdfReader(input_pdf_path)

# Get the pages with start_page predictions
start_pages = predictions_df[predictions_df['predictions'] == 1].index.tolist()

# Ensure the output folder exists
output_folder = 'split_pdfs'
os.makedirs(output_folder, exist_ok=True)

# Function to split PDF
def split_pdf(start_pages, reader):
    documents = []
    for i, start_page in enumerate(start_pages):
        writer = PdfWriter()
        end_page = start_pages[i + 1] if i + 1 < len(start_pages) else len(reader.pages)
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])
        documents.append(writer)
    return documents

# Split the PDF
documents = split_pdf(start_pages, reader)

# Save the split PDFs into the specified folder
for i, writer in enumerate(documents):
    output_pdf_path = os.path.join(output_folder, f'split_document_{i + 1}.pdf')
    with open(output_pdf_path, 'wb') as output_pdf:
        writer.write(output_pdf)

print("PDFs split and saved successfully.")