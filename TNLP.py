import pandas as pd
import numpy as np
import nltk
import re
import gensim
import SentimentAnalysis as sa
import matplotlib.pyplot as plt
from nltk import word_tokenize, pos_tag
from tensorflow import keras
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
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

MORPHOLOGY_CHOICE = 1


def main():
    # Takes the data to a CSV, and then converts it into a dictionary
    training_df = pd.read_csv("dataset/train.csv")
    training_dict = training_df.to_dict('records')

    dev_df = pd.read_csv("dataset/dev.csv")
    dev_dict = dev_df.to_dict('records')

    test_df = pd.read_csv("dataset/test.csv")
    test_dict = test_df.to_dict('records')

    processed_training_tweets = []
    processed_dev_tweets = []
    processed_test_tweets = []

    # Fills array (for column access) with the normalised tweets
    for tweet in training_dict:
        processed_training_tweets.append((process_tweet(tweet["text"]), tweet["airline_sentiment"]))
    for tweet in dev_dict:
        processed_dev_tweets.append((process_tweet(tweet["text"]), tweet["airline_sentiment"]))
    for tweet in test_dict:
        processed_test_tweets.append((process_tweet(tweet["text"]), tweet["airline_sentiment"]))

    training_tweets = np.array(processed_training_tweets)
    dev_tweets = np.array(processed_dev_tweets)
    test_tweets = np.array(processed_test_tweets)


    print("Text processing complete, result: ", training_tweets[3])

    text_clf = Pipeline([
        ('vect', CountVectorizer(analyzer=process_tweet)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier())
    ])

    # tagged_tweets = pos_tagging(training_tweets[:, 0])
    #
    # tagged_tweets = np.array(tagged_tweets)

    # :,0 used to get the tweet column from training_tweets
    # X_train
    word2vec = get_word2vec(training_tweets[:, 0])

    word2vec_clf = Pipeline([
        ('vect', CountVectorizer(analyzer=process_tweet)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', DecisionTreeClassifier())
    ])

    # Training the model using word2vec
    word2vec_clf.fit(training_tweets[:, 0], training_tweets[:, 1])

    # Training the model
    # text_clf.fit(training_tweets[:, 0], training_tweets[:, 1])

    # Calculate and display IDF values
    # df_idf = pd.DataFrame(text_clf['tfidf'].idf_, index=text_clf['vect'].get_feature_names(), columns=["idf_weights"])
    # df_idf.sort_values(by=["idf_weights"])
    # print(df_idf)

    predicted = word2vec_clf.predict(dev_tweets[:, 0])

    print("Accuracy:", metrics.accuracy_score(dev_tweets[:, 1], predicted))
    print(metrics.classification_report(dev_tweets[:, 1], predicted, target_names=target_names))
    print(pd.DataFrame(metrics.confusion_matrix(dev_tweets[:, 1], predicted), columns=target_names, index=target_names))
    df_pred = pd.DataFrame({"tweet": dev_tweets[:, 0], 'Prediction': predicted, 'true:': dev_tweets[:, 1]})
    print(df_pred.head())


# Text normalisation / cleaning
def process_tweet(tweet):
    tweet = tweet.lower()  # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
    tweet = word_tokenize(tweet)  # remove repeated characters

    filtered_words = [word for word in tweet if word not in cus_stopwords]

    # Stemming
    if MORPHOLOGY_CHOICE == 1:
        print("Stemming...")
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in filtered_words]
        return " ".join(stemmed_words)
    # Lemmatization
    elif MORPHOLOGY_CHOICE == 2:
        print("Getting lemma...")
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in filtered_words]
        return " ".join(lemma_words)
    elif MORPHOLOGY_CHOICE == 3:
        return filtered_words


def pos_tagging(tweets):
    print("POS tagging...")
    tagged_tweets = []
    for tweet in tweets:
        print(tweet)
        pos_tag_tweet = pos_tag(tweet)
        print(pos_tag_tweet)
        tagged_tweets.append((pos_tag_tweet[0], pos_tag_tweet[1]))

    print("Work complete. Example: ", tagged_tweets[1])
    return tagged_tweets


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

