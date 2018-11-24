# import gensim.downloader as api
# fasttext = api.load("fasttext-wiki-news-subwords-300")  

_MEAN = 0
_STANDARD_DEV = 1

# train_image_zero, train_target_zero = _mnist_training[0]
training_batches = t.utils.data.DataLoader(dataset = _newsgp_training,
                                           batch_size = _BATCH_SIZE, 
                                           shuffle = True
                                          )
holdout_batches = t.utils.data.DataLoader(dataset = _newsgp_holdout,
                                          batch_size = _BATCH_SIZE, 
                                          shuffle = True
                                         )