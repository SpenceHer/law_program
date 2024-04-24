import pandas as pd
from IPython import embed
import ast


embed()


df = pd.read_csv(r"X:\OHSU Shared\Restricted\SOM\SURG\ORTH\Smith\spencer_stuff\other_coding_projects\law_program\page_df.csv")
df.drop(columns={"page_num.1"}, inplace=True)

# Separate Layout Analysis Values
df['layout_analysis_dict'] = df['layout_analysis'].apply(eval)
df['vertical_margin_ratio'] = df['layout_analysis_dict'].apply(lambda x: x['vertical_margin_ratio'])
df['horizontal_margin_ratio'] = df['layout_analysis_dict'].apply(lambda x: x['horizontal_margin_ratio'])



for page_num in range(1, len(df)+1):
    prev_page_data = df.loc[df["page_num"] == page_num - 1]
    page_data = df.loc[df["page_num"] == page_num]
    next_page_data = df.loc[df["page_num"] == page_num + 1]
    
################################################

    # Previous Page Data
    if len(prev_page_data) > 0:
        df.loc[df["page_num"] == page_num, "prev_page"] = 1

        #Entities
        prev_page_entities = eval(prev_page_data["entities"].values[0])

        # Margins
        df.loc[df["page_num"] == page_num, "prev_vertical_margin_ratio_diff"] = abs(page_data["vertical_margin_ratio"].values[0] - prev_page_data["vertical_margin_ratio"].values[0])
        df.loc[df["page_num"] == page_num, "prev_horizontal_margin_ratio_diff"] = abs(page_data["horizontal_margin_ratio"].values[0] - prev_page_data["horizontal_margin_ratio"].values[0])

        # Word Count
        df.loc[df["page_num"] == page_num, "prev_page_word_count"] = page_data["word_count"].values[0]


    else:
        df.loc[df["page_num"] == page_num, "prev_page"] = 0

        prev_page_entities = []
        df.loc[df["page_num"] == page_num, "prev_vertical_margin_ratio_diff"] = 0
        df.loc[df["page_num"] == page_num, "prev_horizontal_margin_ratio_diff"] = 0

################################################

    # Current Page Data
    page_entities = eval(page_data["entities"].values[0])

################################################

    # Next Page Data
    if len(next_page_data) > 0:
        df.loc[df["page_num"] == page_num, "next_page"] = 1

        next_page_entities = eval(next_page_data["entities"].values[0])

        df.loc[df["page_num"] == page_num, "next_vertical_margin_ratio_diff"] = abs(page_data["vertical_margin_ratio"].values[0] - next_page_data["vertical_margin_ratio"].values[0])
        df.loc[df["page_num"] == page_num, "next_horizontal_margin_ratio_diff"] = abs(page_data["horizontal_margin_ratio"].values[0] - next_page_data["horizontal_margin_ratio"].values[0])

    else:
        df.loc[df["page_num"] == page_num, "next_page"] = 0

        next_page_entities = []
        df.loc[df["page_num"] == page_num, "next_vertical_margin_ratio_diff"] = 0
        df.loc[df["page_num"] == page_num, "next_horizontal_margin_ratio_diff"] = 0

################################################

def unique_entity_count(entities):
    from collections import Counter
    unique_entities = set(entities)  # Set to filter unique entities
    return Counter([entity_type for _, entity_type in unique_entities])

def entity_count(entities):
    from collections import Counter
    return Counter([entity_type for _, entity_type in entities])


eval(prev_page_data["layout_analysis"].values[0])










page_num = 34

prev_page_data = df.loc[df["page_num"] == page_num - 1]
page_data = df.loc[df["page_num"] == page_num]
next_page_data = df.loc[df["page_num"] == page_num + 1]


text1 = page_data["page_text"].values[0]
text2 = next_page_data["page_text"].values[0]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Cosine Similarity: {similarity[0][0]}")






















def find_page_number_format(strings):
    import re
    # Regex pattern to match strings like 'page x of page x', 'x of x', or 'A26 of A27'
    pattern = r'\b[A-Za-z]*\d+\s+of\s+[A-Za-z]*\d+\b'
    
    # Search for matching strings in the list
    for string in strings:
        if re.search(pattern, string):
            return string
    return None  # Return None if no match is found

# List of strings
strings = ['DUGGER + ASSOCIATES, P.A.', 'Architectural Acoustics', 'WED+A', 'acoustics', 'EDWARD', 'Consultants in', 
           '90', '90', '80', '70', '60', '60', 'Sound Pressure Level [dB]', '20', '30', '20', '20', '10', '0', 
           'November 25, 2022', '-', '8:00 p.m. 11:11 p.m.', 'NC-40', 'Threshold of', 'Human Hearing', '16', '31.5', 
           '63', '125', '250', '500', 'NC-70', 'NC-65', 'NC-60', 'NC-55', 'NC-50', 'NC-45', 'NC-40', 'NC-35', 
           'NC-30', 'NC-25', 'NC-20', 'NC-15', '1000', '2000', '4000', '8000', 'Frequency [Hz]', 'Figure A35', 
           '1239 SE Indian Street, Suite 103, Stuart, Florida 34997', 'T: 772-286-8351 www.edplusa.com AA26000667', 
           '4331 North Ocean Drive, Lauderdale-by-the-Sea, Florida', 'A26 of A27', 'WHITECAP0000184']

# Using the function
matching_string = find_page_number_format(strings)
print(f"Matching string: {matching_string}")