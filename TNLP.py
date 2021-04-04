import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
stopwordsEn = stopwords.words('english')
from nltk.corpus import wordnet
nltk.download('wordnet')
from nltk.stem import *

METHOD_CHOICE = 1

# Importing the data into dataframes
dev_set = pd.read_csv("dataset/dev.csv")
training_set = pd.read_csv("dataset/test.csv")
test_set = pd.read_csv("dataset/train.csv")

# Returns all words from a list of tokens
def get_all_words(cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token


def get_tweets_for_model(cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)

    # def process_text():


def main():
    if METHOD_CHOICE == 1:
        text_clf = Pipeline([
            ('vect', CountVectorizer(analyzer=baselineTextProcess)),
            ('tfidf', TfidfTransformer(use_idf=True)),

             # Classifier - Stochastic Gradient Descent
            ('clf', SGDClassifier())

        ])
    elif METHOD_CHOICE == 2:
        text_clf = Pipeline([
            ('vect', CountVectorizer(analyzer=custom_text_process)),
            ('tfidf', TfidfTransformer(use_idf=True)),

            # Classifier
            ('clf', SGDClassifier)
        ])



if __name__ == "__main__":
    main()

