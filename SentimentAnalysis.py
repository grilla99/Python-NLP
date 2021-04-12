import os
import csv
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn

def is_adjective(tag):
    if tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
        return True
    else:
        return False


def is_adverb(tag):
    if tag == 'RB' or tag == 'RBR' or tag == 'RBS':
        return True
    else:
        return False


def is_noun(tag):
    if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
        return True
    else:
        return False


def is_verb(tag):
    if tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
        return True
    else:
        return False


def is_valid(token):
    if is_noun(token[1]) or is_adverb(token[1]) or is_verb(token[1]) or is_adjective(token[1]):
        return True
    else:
        return False


def get_sentiment_from_level(i):
    if i == 4 or i == 'positive':
        return 'positive'
    elif i == 2 or i == 'neutral':
        return 'neutral'
    else:
        return 'negative'


def get_first_synset(word):
    synsets = swn.senti_synsets(word)
    if len(synsets) > 0:
        return synsets[0]
    else:
        return None


def filter_tweet(tweet):
    return map(lambda x : x[0], filter(lambda token : is_valid(token), tweet))


def get_synsets(tweet):
    return filter(lambda x: x is not None, map(lambda  x: get_first_synset(x), tweet))


def get_posScore_from_synsets(sentisynsets, tweet):
    scores = list(map(lambda sentisynset: sentisynset.pos_score(), sentisynsets))
    if len(scores) > 0:
        return reduce(lambda a,x: a + x, scores)
    else:
        return 0

def get_negScore_from_synsets(sentisynsets,tweet):
    scores = list(map(lambda sentisynset: sentisynset.neg_score(), sentisynsets))
    if len(scores) > 0:
        return reduce(lambda a,x: a + x, scores)
    else:
        return 0


def get_tweet_sentiment_from_score(posScore, negScore):
    if posScore > negScore:
        return 'positive'
    elif posScore == negScore:
        return 'neutral'
    else:
        return 'negative'

#Deals with empty strings
def pos_tagging(tokens):
    return pos_tag(i for i in tokens if i)


def get_sentiment_from_tweet(tweet):
    tweet = filter_tweet(tweet)
    sentisynsets = get_synsets(tweet)
    posScore = get_posScore_from_synsets(sentisynsets,tweet)
    negScore = get_negScore_from_synsets(sentisynsets,tweet)

    sentiment = get_tweet_sentiment_from_score(posScore, negScore)

    return sentiment
