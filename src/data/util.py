from sklearn.datasets import fetch_20newsgroups
import numpy as np
from scipy.stats import zscore
from spacy import load
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader

def fetch_data():
   '''Fetches data files with key attributes: 'data', 'target', 'target_names'.
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

def standardize_dataset(X, y):
   '''Build a TensorDataset object--basically a tuple holding 2 tensors 
   (1 design matrix + 1 response vector)--by standardizing features and then 
   converting from arrays to tensors. 
   '''
   tensor_X = from_numpy(zscore(X))
   tensor_y = from_numpy(y)
   return TensorDataset(tensor_X, tensor_y)

def mk_dataloader(dataset):
   '''Make a DataLoader object--bundling a dataset with its configurations.
   ''' 
   return DataLoader(dataset = dataset,
                     batch_size = 100, 
                     shuffle = True
                    )