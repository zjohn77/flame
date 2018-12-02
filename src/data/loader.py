"""
Build DataLoader objects starting from the raw text.
"""
from scipy.stats import zscore
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader

def standardize_dataset(X, y: 'numpy arrays'):
   '''Wrap X, y into a TensorDataset object having the attribute TensorDataset.tensors,
   which is a tuple holding 2 tensors (predictors + response).
   '''
   Z = zscore(X)
   Ztensor = from_numpy(Z)
   return TensorDataset(Ztensor.permute(0, 2, 1).float(), 
                        from_numpy(y).long()
                       )

def mk_dataloader(dataset: 'TensorDataset object'):
   '''Make a DataLoader object--bundling a dataset with its configurations.
   ''' 
#    print(dataset.tensors[0].shape)
#    torch.Size([3766, 300, 25])

   return DataLoader(dataset = dataset,
                     batch_size = 100,
                     shuffle = False
                    )