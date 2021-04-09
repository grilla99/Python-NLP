import pandas as pd
import numpy as np
import nltk
import re
import gensim
import matplotlib.pyplot as plt
from nltk import word_tokenize
from tensorflow import keras
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from nltk.stem import *
from nltk.corpus import stopwords
from string import punctuation
from sklearn.pipeline import Pipeline

stopwordsEn = stopwords.words('english')
nltk.download('wordnet')

cus_stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])
target_names = ['positive', 'negative', 'neutral']

MORPHOLOGY_CHOICE = 3


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

    print(training_tweets[:,0])

    # :,0 used to get the tweet column from training_tweets
    # get_word2vec(training_tweets[:, 0])

    # text_clf.fit(training_tweets[:, 0], training_tweets[:, 1])
    #
    # predicted = text_clf.predict(training_tweets[:, 0])
    #
    # print("Accuracy:", metrics.accuracy_score(training_tweets[:, 1], predicted))
    #
    # print(metrics.classification_report(training_tweets[:, 1], predicted, target_names=target_names))
    #
    # print(pd.DataFrame(metrics.confusion_matrix(training_tweets[:, 1], predicted), columns=target_names, index=target_names))
    #
    # df_pred = pd.DataFrame({"tweet": training_tweets[:, 0], 'Prediction': predicted, 'true:': training_tweets[:, 1]})


# Text normalisation / cleaning
def process_tweet(tweet):
    tweet = tweet.lower()
    tweet = tweet.lower()  # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
    tweet = word_tokenize(tweet)  # remove repeated characters

    filtered_words = [word for word in tweet if word not in cus_stopwords]

    # Stemming
    if MORPHOLOGY_CHOICE == 1:
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in filtered_words]
        return " ".join(stemmed_words)
    # Lemmatization
    elif MORPHOLOGY_CHOICE == 2:
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
        return " ".join(lemma_words)
    elif MORPHOLOGY_CHOICE == 3:
        return filtered_words

# def pos_tagging():



# Word to Vector Function
def get_word2vec(sentences):
    num_features = 200
    epoch_count = 15
    sentence_count = len(sentences)
    min_count = 1

    word2vec = gensim.models.Word2Vec(sg=1, seed=1, vector_size=num_features, min_count=min_count,
                                      window=5, sample=0)
    print("Building vocab...")

    word2vec.build_vocab(sentences)
    print("Word2Vec Vocab Length: ", len(word2vec.wv.key_to_index))

    print("Training...")
    word2vec.train(sentences, total_examples=sentence_count, epochs=epoch_count)

    print(word2vec.wv.key_to_index)

    return word2vec


if __name__ == "__main__":
    main()

