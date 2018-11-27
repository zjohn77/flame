"""
Build DataLoader objects starting from the raw text.
"""
import numpy as np
from torch import tensor, arange, double
from util import fetch_data, doc2matrix, standardize_dataset, mk_dataloader

DIM = 300   ## comes from 300D GloVe embedding

## 1. fetch train & test from 20newsgroups in sklearn.datasets 
train, test = fetch_data()

## 2. 


# matrices = np.array([])
# matrices = np.empty((0, DIM))
# for doc in train.data:
#    np.concatenate((matrices, doc2matrix(doc)))
# print(matrices)

# training_batches = mk_dataloader()
# holdout_batches = mk_dataloader()