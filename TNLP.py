import pandas as pd
import numpy as np
import nltk
import re
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from nltk.corpus import stopwords
from string import punctuation
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
stopwordsEn = stopwords.words('english')
from nltk.corpus import wordnet
nltk.download('wordnet')
from nltk.stem import *

cus_stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])
target_names = ['positive', 'negative', 'neutral']


def main():
    # This code takes the data to a CSV, and then converts it into a dictionary
    training_df = pd.read_csv("dataset/train.csv")
    training_dict = training_df.to_dict('records')

    dev_df = pd.read_csv("dataset/dev.csv")
    dev_dict = dev_df.to_dict('records')

    test_df = pd.read_csv("dataset/test.csv")
    test_dict = test_df.to_dict('records')

    processed_training_tweets = []
    processed_dev_tweets = []
    processed_test_tweets = []

    for tweet in training_dict:
        processed_training_tweets.append((process_tweet(tweet["text"]), tweet["airline_sentiment"]))
    for tweet in dev_dict:
        processed_dev_tweets.append((process_tweet(tweet["text"]), tweet["airline_sentiment"]))
    for tweet in test_dict:
        processed_test_tweets.append((process_tweet(tweet["text"]), tweet["airline_sentiment"]))

    training_tweets = np.array(processed_training_tweets)

    text_clf = Pipeline([
        ('vect', CountVectorizer(analyzer=process_tweet)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier())
    ])

    text_clf.fit(training_tweets[:, 0], training_tweets[:, 1])

    predicted = text_clf.predict(training_tweets[:, 0])

    print("Accuracy:", metrics.accuracy_score(training_tweets[:, 1], predicted))

    print(metrics.classification_report(training_tweets[:, 1], predicted, target_names=target_names))

    print(pd.DataFrame(metrics.confusion_matrix(training_tweets[:, 1], predicted), columns=target_names, index=target_names))


# Text normalisation / cleaning
def process_tweet(tweet):
    for words in tweet:
        tweet = words.lower()
        tweet = words.lower()  # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', words)  # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', words)  # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', words)  # remove the # in #hashtag
        tweet = word_tokenize(tweet)  # remove repeated characters

    return [word for word in tweet if word not in cus_stopwords]


if __name__ == "__main__":
    main()

