import pandas as pd
from IPython import embed
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_euclidean_distance(color_string1, color_string2):
    def convert_color_string(color_string):
        # Remove the brackets and split the string
        color_values = re.sub(r'[\[\]]', '', color_string).strip().split()
        # Convert each part to an integer
        return [int(value) for value in color_values]
    color1 = convert_color_string(color_string1)
    color2 = convert_color_string(color_string2)

    color1 = np.array(color1)
    color2 = np.array(color2)

    distance = abs(np.linalg.norm(color1 - color2))
    return distance


def calculate_text_similarity(list1, list2):

    # Step 1: Convert each list of words into a single string
    doc1 = ' '.join(list1)
    doc2 = ' '.join(list2)

    # Step 2: Create a TF-IDF Vectorizer object and transform the documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

    # Step 3: Calculate cosine similarity between the two documents
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Return the cosine similarity value
    return similarity[0][0]

embed()




df = pd.read_csv(r"/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/page_dataframe.csv")
df.drop(columns={"page_num.1"}, inplace=True)

# Separate Layout Analysis Values
df["layout_analysis_dict"] = df["layout_analysis"].apply(eval)
df["vertical_margin_ratio"] = df["layout_analysis_dict"].apply(lambda x: x["vertical_margin_ratio"])
df["horizontal_margin_ratio"] = df["layout_analysis_dict"].apply(lambda x: x["horizontal_margin_ratio"])

df["page_words"] = df["page_words"].apply(eval)

df["page_rows"] = df["page_rows"].apply(eval)

df["entities"] = df["entities"].apply(eval)



################################################

for page_num in tqdm(range(1, len(df)+1)):
    prev_page_data = df.loc[df["page_num"] == page_num - 1]
    page_data = df.loc[df["page_num"] == page_num]
    next_page_data = df.loc[df["page_num"] == page_num + 1]
    

################################################

    # Current Page Data
    page_entities = page_data["entities"].values[0]
    page_header = page_data["page_rows"].values[0][0:3]
    page_width = eval(page_data["page_size"].values[0])[0]
    page_height = eval(page_data["page_size"].values[0])[1]

    # Email start pattern
    df.loc[(df["page_num"] == page_num) & ("message" == page_data.page_words.values[0][0].lower()), "email_start_1"] = 1
    df.loc[(df["page_num"] == page_num) & ("message" != page_data.page_rows.values[0][0].lower()), "email_start_1"] = 0

################################################

    # Previous Page Data
    if len(prev_page_data) > 0:
        df.loc[df["page_num"] == page_num, "prev_page"] = 1

        # Entities
        prev_page_entities = prev_page_data["entities"].values[0]
        df.loc[df["page_num"] == page_num, "prev_page_shared_entities"] = len(set(prev_page_entities).intersection(set(page_data["entities"].values[0])))

        total_count = 0
        for tuple_ in page_entities:
            total_count += prev_page_entities.count(tuple_)
        df.loc[df["page_num"] == page_num, "prev_page_shared_entities_total"] = total_count

        # Margins
        df.loc[df["page_num"] == page_num, "prev_vertical_margin_ratio_diff"] = abs(page_data["vertical_margin_ratio"].values[0] - prev_page_data["vertical_margin_ratio"].values[0])
        df.loc[df["page_num"] == page_num, "prev_horizontal_margin_ratio_diff"] = abs(page_data["horizontal_margin_ratio"].values[0] - prev_page_data["horizontal_margin_ratio"].values[0])

        # Word Count
        df.loc[df["page_num"] == page_num, "prev_page_word_count"] = page_data["word_count"].values[0]

        # Header similarity
        prev_page_header = prev_page_data["page_rows"].values[0][0:3]
        df.loc[df["page_num"] == page_num, "prev_page_header_similarity"] = calculate_text_similarity(page_header, prev_page_header)

        # Page height and width
        df.loc[df["page_num"] == page_num, "prev_page_width_diff"] = abs(eval(page_data["page_size"].values[0])[0] - eval(prev_page_data["page_size"].values[0])[0])
        df.loc[df["page_num"] == page_num, "prev_page_height_diff"] = abs(eval(page_data["page_size"].values[0])[0] - eval(prev_page_data["page_size"].values[0])[1])

        # Text Polarity and Sentiment
        df.loc[df["page_num"] == page_num, "prev_page_polarity_diff"] = abs(page_data["text_sentiment_polarity"].values[0] - prev_page_data["text_sentiment_polarity"].values[0])
        df.loc[df["page_num"] == page_num, "prev_page_subjectivity_diff"] = abs(page_data["text_sentiment_subjectivity"].values[0] - prev_page_data["text_sentiment_subjectivity"].values[0])

        # Dominant Color Difference
        df.loc[df["page_num"] == page_num, "prev_page_dom_color_diff"] = calculate_euclidean_distance(page_data["dominant_color"].values[0], prev_page_data["dominant_color"].values[0])

    else:
        df.loc[df["page_num"] == page_num, "prev_page"] = 0

        prev_page_entities = []
        df.loc[df["page_num"] == page_num, "prev_page_shared_entities"] = 0
        df.loc[df["page_num"] == page_num, "prev_page_shared_entities_total"] = 0
        df.loc[df["page_num"] == page_num, "prev_vertical_margin_ratio_diff"] = 0
        df.loc[df["page_num"] == page_num, "prev_horizontal_margin_ratio_diff"] = 0
        df.loc[df["page_num"] == page_num, "prev_page_word_count"] = 0
        df.loc[df["page_num"] == page_num, "prev_page_header_similarity"] = 0
        df.loc[df["page_num"] == page_num, "prev_page_width_diff"] = 0
        df.loc[df["page_num"] == page_num, "prev_page_height_diff"] = 0
        df.loc[df["page_num"] == page_num, "prev_page_polarity_diff"] = 0
        df.loc[df["page_num"] == page_num, "prev_page_subjectivity_diff"] = 0
        df.loc[df["page_num"] == page_num, "prev_page_dom_color_diff"] = 0



################################################

    # Next Page Data
    if len(next_page_data) > 0:
        df.loc[df["page_num"] == page_num, "next_page"] = 1

        # Entities
        next_page_entities = next_page_data["entities"].values[0]
        df.loc[df["page_num"] == page_num, "next_page_shared_entities"] = len(set(next_page_entities).intersection(set(page_data["entities"].values[0])))

        total_count = 0
        for tuple_ in page_entities:
            total_count += next_page_entities.count(tuple_)
        df.loc[df["page_num"] == page_num, "next_page_shared_entities_total"] = total_count

        # Margins
        df.loc[df["page_num"] == page_num, "next_vertical_margin_ratio_diff"] = abs(page_data["vertical_margin_ratio"].values[0] - next_page_data["vertical_margin_ratio"].values[0])
        df.loc[df["page_num"] == page_num, "next_horizontal_margin_ratio_diff"] = abs(page_data["horizontal_margin_ratio"].values[0] - next_page_data["horizontal_margin_ratio"].values[0])

        # Word Count
        df.loc[df["page_num"] == page_num, "next_page_word_count"] = page_data["word_count"].values[0]

        # Header similarity
        next_page_header = next_page_data["page_rows"].values[0][0:3]
        df.loc[df["page_num"] == page_num, "next_page_header_similarity"] = calculate_text_similarity(page_header, next_page_header)

        # Page height and width
        df.loc[df["page_num"] == page_num, "next_page_width_diff"] = abs(eval(page_data["page_size"].values[0])[0] - eval(next_page_data["page_size"].values[0])[0])
        df.loc[df["page_num"] == page_num, "next_page_height_diff"] = abs(eval(page_data["page_size"].values[0])[0] - eval(next_page_data["page_size"].values[0])[1])

        # Text Polarity and Sentiment
        df.loc[df["page_num"] == page_num, "next_page_polarity_diff"] = abs(page_data["text_sentiment_polarity"].values[0] - next_page_data["text_sentiment_polarity"].values[0])
        df.loc[df["page_num"] == page_num, "next_page_subjectivity_diff"] = abs(page_data["text_sentiment_subjectivity"].values[0] - next_page_data["text_sentiment_subjectivity"].values[0])

        # Dominant Color Difference
        df.loc[df["page_num"] == page_num, "next_page_dom_color_diff"] = calculate_euclidean_distance(page_data["dominant_color"].values[0], next_page_data["dominant_color"].values[0])


    else:
        df.loc[df["page_num"] == page_num, "next_page"] = 0

        next_page_entities = []
        df.loc[df["page_num"] == page_num, "next_page_shared_entities"] = 0
        df.loc[df["page_num"] == page_num, "next_page_shared_entities_total"] = 0
        df.loc[df["page_num"] == page_num, "next_vertical_margin_ratio_diff"] = 0
        df.loc[df["page_num"] == page_num, "next_horizontal_margin_ratio_diff"] = 0
        df.loc[df["page_num"] == page_num, "next_page_word_count"] = 0
        df.loc[df["page_num"] == page_num, "next_page_header_similarity"] = 0
        df.loc[df["page_num"] == page_num, "next_page_width_diff"] = 0
        df.loc[df["page_num"] == page_num, "next_page_height_diff"] = 0
        df.loc[df["page_num"] == page_num, "next_page_polarity_diff"] = 0
        df.loc[df["page_num"] == page_num, "next_page_subjectivity_diff"] = 0
        df.loc[df["page_num"] == page_num, "next_page_dom_color_diff"] = 0


################################################


page_num = 2

prev_page_data = df.loc[df["page_num"] == page_num - 1]
page_data = df.loc[df["page_num"] == page_num]
next_page_data = df.loc[df["page_num"] == page_num + 1]

for col in page_data.columns:
    print(f"\n\n{col}")
    print(prev_page_data[col].values[0])
    print(page_data[col].values[0])
    print(next_page_data[col].values[0])






















page_num = 34

prev_page_data = df.loc[df["page_num"] == page_num - 1]
page_data = df.loc[df["page_num"] == page_num]
next_page_data = df.loc[df["page_num"] == page_num + 1]


text1 = page_data["page_text"].values[0]
text2 = prev_page_data["page_text"].values[0]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Cosine Similarity: {similarity[0][0]}")











def jaccard_similarity(text1, text2):
    """
    Compute the Jaccard Similarity between two texts.
    """
    # Split the texts into sets of words
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Compute Jaccard Similarity
    return len(intersection) / len(union)

# Example usage
text1 = ""
text2 = ""
similarity = jaccard_similarity(text1, text2)
print(f"Jaccard Similarity: {similarity:.2f}")



