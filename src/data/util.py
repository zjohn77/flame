from numpy import array
from spacy import load
from sklearn.datasets import fetch_20newsgroups

def doc2matrix(doc):   
   tokens = load('en_vectors_web_lg')(doc)
   embedding_matrix = [token.vector for token in tokens]
   return array(embedding_matrix)

def fetch_data():
   return (fetch_20newsgroups(subset='train'), 
           fetch_20newsgroups(subset='test')
          )