from numpy import array
from spacy import load
from sklearn.datasets import fetch_20newsgroups

def doc2matrix(doc):   
   '''doc2matrix should tokenize the phrase into 6 tokens, then embed them
   in 300D GloVe vectors.
   '''        
   tokens = load('en_vectors_web_lg')(doc)
   embedding_matrix = [token.vector for token in tokens]
   return array(embedding_matrix)

def fetch_data():
   '''['DESCR', 'data', 'description', 'filenames', 'target', 'target_names']
   '''
   return (fetch_20newsgroups(subset='train'), 
           fetch_20newsgroups(subset='test')
          )

train, test = fetch_data()
print(train.description)
