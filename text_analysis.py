# text_analysis.py

import spacy
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial.distance import cosine

class TextAnalyzer:
    def __init__(self):
        # Load spaCy model for entity recognition
        self.nlp = spacy.load("en_core_web_sm")
        # Load pre-trained BERT model for text embeddings
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def analyze_sentiment(self, text):
        # Returns the polarity and subjectivity of the text using TextBlob
        sentiment = TextBlob(text).sentiment
        return sentiment.polarity, sentiment.subjectivity

    def extract_entities(self, text):
        # Uses spaCy to extract named entities from text.
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def get_bert_embedding(self, text):
        # Returns BERT embeddings for the given text
        encoded_input = self.bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.bert_model(**encoded_input)
        embeddings = output.last_hidden_state.mean(dim=1).squeeze()  # Reduce to single vector per input
        return embeddings.numpy()

    def calculate_text_similarity(self, text1, text2):
        # Calculates the cosine similarity between two pieces of text using BERT embeddings.
        embedding1 = self.get_bert_embedding(text1)
        embedding2 = self.get_bert_embedding(text2)
        similarity = 1 - cosine(embedding1, embedding2)
        return similarity

