import numpy as np
from scipy.stats import zscore

from sklearn.datasets import fetch_20newsgroups
from spacy import load

from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader

def fetch_data():
   '''['DESCR', 'data', 'description', 'filenames', 'target', 'target_names']
   '''
   return (fetch_20newsgroups(subset='train'), 
           fetch_20newsgroups(subset='test')
          )

def doc2matrix(doc):   
   '''doc2matrix should tokenize the phrase into 6 tokens, then embed them
   in 300D GloVe vectors.
   '''        
   tokens = load('en_vectors_web_lg')(doc)
   embedding_matrix = [token.vector for token in tokens]
   return np.array(embedding_matrix)

def standardize_dataset(features, targets):
   tensor_features = from_numpy(zscore(features))
   tensor_targets = from_numpy(targets)
   return TensorDataset(tensor_features, tensor_targets)

def mk_dataloader(dataset):
   return DataLoader(dataset = dataset,
                     batch_size = 100, 
                     shuffle = True
                    )