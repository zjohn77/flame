"""
Created on Fri Jan 08 15:06:47 2016
"""

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import brown
from nltk.classify.api import ClassifierI, MultiClassifierI
from nltk.classify.megam import config_megam, call_megam
from nltk.classify.weka import WekaClassifier, config_weka
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.positivenaivebayes import PositiveNaiveBayesClassifier
from nltk.classify.decisiontree import DecisionTreeClassifier
from nltk.classify.rte_classify import rte_classifier, rte_features, RTEFeatureExtractor
from nltk.classify.util import accuracy, apply_features, log_likelihood
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify.maxent import (MaxentClassifier, BinaryMaxentFeatureEncoding,
                                  TypedMaxentFeatureEncoding,
                                  ConditionalExponentialClassifier)

from nltk.corpus import names
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# dict of word counts
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
all_words.freq
word_features = list(all_words)[:2000]


def document_features(document):
    ''' checks whether each of top 2000 words is present in a given document'''
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

