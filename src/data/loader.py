"""
Build DataLoader objects starting from the raw text.
"""
import numpy as np
from torch import tensor, arange, double
from util import fetch_data, doc2matrix, standardize_dataset, mk_dataloader
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import zscore
from torch import from_numpy

def standardize_dataset(X, y):
   '''Build a TensorDataset object--basically a tuple holding 2 tensors 
   (1 design matrix + 1 response vector)--by standardizing features and then 
   converting from arrays to tensors. 
   '''
   tensor_X = from_numpy(zscore(X))
   tensor_y = from_numpy(y)
   return TensorDataset(tensor_X, tensor_y)

def sample_split(X, y):
   X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify = y, 
                                                    test_size = 0.5)

def mk_dataloader(dataset):
   '''Make a DataLoader object--bundling a dataset with its configurations.
   ''' 
   return DataLoader(dataset = dataset,
                     batch_size = 100, 
                     shuffle = True
                    )

# training_batches = mk_dataloader()
# holdout_batches = mk_dataloader()