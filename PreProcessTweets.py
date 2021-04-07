import re
import pandas as pd
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords

training_set = pd.read_csv("dataset/train.csv")

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def processTweets(self, list_of_tweets):
        processedTweets = []
        for index, tweet in list_of_tweets.iterrows():
            processedTweets.append((self._processTweet(tweet["text"]), tweet["airline_sentiment"]))
        return processedTweets

    def _processTweet(self, tweet):
        tweet = tweet.lower()  # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
        tweet = word_tokenize(tweet)  # remove repeated characters

        return [word for word in tweet if word not in self._stopwords]


tweet_processor = PreProcessTweets()
preprocessed_training_set = tweet_processor.processTweets(training_set)
# preprocessed_test_set = tweet_processor.processTweets(test_set)