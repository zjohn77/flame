"""
Build DataLoader objects starting from the raw text.
"""
from scipy.stats import zscore
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader
from numpy import array

def standardize_dataset(X, y: 'numpy arrays'):
   '''Wrap X, y into a TensorDataset object having the attribute TensorDataset.tensors,
   which is a tuple holding 2 tensors (predictors + response).
   '''
   X_tensor = from_numpy(zscore(X))
   
   try:
      y_tensor = from_numpy(y)
   except TypeError:
      y_tensor = from_numpy(array(y))
   
   return TensorDataset(X_tensor.permute(0, 2, 1).float(), 
                        y_tensor.long()
                       )

def mk_dataloader(dataset: 'TensorDataset object'):
   '''Make a DataLoader object--bundling a dataset with its configurations.
   ''' 
   return DataLoader(dataset = dataset,
                     batch_size = 50,
                     shuffle = False
                    )