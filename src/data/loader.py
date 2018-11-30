"""
Build DataLoader objects starting from the raw text.
"""
from scipy.stats import zscore
# from sklearn.model_selection import train_test_split
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader

# def strat_split(X, y):
#    X_trn, X_vld, y_trn, y_vld = train_test_split(X, y,
#                                                  stratify = y, 
#                                                  test_size = .5
#                                                 )
#    return X_trn, X_vld, y_trn, y_vld

def standardize_dataset(X, y: 'numpy arrays'):
   '''Wrap X, y into a TensorDataset object having the attribute TensorDataset.tensors,
   which is a tuple holding 2 tensors (predictors + response).
   '''
   return TensorDataset(from_numpy(zscore(X)), 
                        from_numpy(y)
                       )

def mk_dataloader(dataset: 'TensorDataset object'):
   '''Make a DataLoader object--bundling a dataset with its configurations.
   ''' 
   return DataLoader(dataset = dataset,
                     batch_size = 100,
                     shuffle = False
                    )