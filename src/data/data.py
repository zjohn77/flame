import gensim.downloader as api
import torch as t

fasttext = api.load("fasttext-wiki-news-subwords-300")  

_newsgp_training = 0
_newsgp_holdout = 0

## private attributes
_BATCH_SIZE = 100
_MEAN = 0
_STANDARD_DEV = 1



## API          
# train_image_zero, train_target_zero = _mnist_training[0]

# print(_newsgp_training.data[-1:][0])                      
training_batches = t.utils.data.DataLoader(dataset = _newsgp_training,
                                           batch_size = _BATCH_SIZE, 
                                           shuffle = True
                                          )
holdout_batches = t.utils.data.DataLoader(dataset = _newsgp_holdout,
                                          batch_size = _BATCH_SIZE, 
                                          shuffle = True
                                         )