
# coding: utf-8

# In[25]:

import sklearn as sk
import numpy as np
import pandas as pd
import gensim as gs
from gensim import corpora, models, similarities
import nltk
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[28]:

sentences = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# tokenize
word_list = [word_tokenize(sentence.lower()) for sentence in sentences] 



# In[32]:

# get stop-words corpus from NLTK. There are 153 stop-words.
stop_words = set(stopwords.words("english"))


# In[36]:

filtered_paragraph = []        
for sentence in word_list:    
    filtered_paragraph.append(set(sentence) - stop_words)


# In[37]:

print filtered_paragraph

