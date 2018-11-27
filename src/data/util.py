from sklearn.datasets import fetch_20newsgroups
import numpy as np
from scipy.stats import zscore
from spacy import load
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader
from random import sample

## Loads GloVe embedding into the PRETRAINED_GLOVE function, 
# which returns a list of tokens derived from its document argument.
PRETRAINED_GLOVE = load('en_vectors_web_lg')

def fetch_data():
   '''Fetches data files with key attributes: "data", "target", "filenames".
   '''
   return fetch_20newsgroups(subset = 'all')

def to_hashmap(filenames, data, target):
   '''Reshape the object fetched from sklearn into a new dict keyed by "filenames".  
   '''
   return {f: [d, t] for f, d, t in zip(filenames, data, target)}

def discard_long_docs(docs, N=1000):
   '''Separate documents with 1000+ "words", since extremely long text may warrant a separate model.
   '''        
   ## Indicator function for whether word count of a text document is l.e. 1000.             
   indicator = (lambda text: len(PRETRAINED_GLOVE(text)) 
                <= N
               )
   return {key: val for key, val in docs.items() if indicator(val[0])}
  
def doc2matrix(doc):   
   '''doc2matrix should tokenize the phrase into 6 tokens, then embed them
   in 300D GloVe vectors.
   '''        
   tokens = PRETRAINED_GLOVE(doc)
   return np.array([token.vector for token in tokens])


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