"""
1. Download the MNIST data (both the training and the holdout) from torchvision.
2. Standardize the scale of all features.
3. Create 500-sized batches for fast processing.
4. Make training_batches and holdout_batches available as public API.
"""
from torchvision import datasets, transforms
import torch as t

## private attributes
_BATCH_SIZE = 500
_MEAN = 0.1307
_STANDARD_DEV = 0.3081
_transf = transforms.Compose([transforms.ToTensor(), 
                              transforms.Normalize((_MEAN,), 
                                                   (_STANDARD_DEV,)
                                                  )
                            ])
_mnist_training = datasets.MNIST(root = './src/data', 
                                 train = True, 
                                 download = True,
                                 transform = _transf
                                )
_mnist_holdout = datasets.MNIST(root = './src/data', 
                                train = False, 
                                download = False,
                                transform = _transf
                               )


## API                                
training_batches = t.utils.data.DataLoader(dataset = _mnist_training,
                                           batch_size = _BATCH_SIZE, 
                                           shuffle = True
                                          )
holdout_batches = t.utils.data.DataLoader(dataset = _mnist_holdout,
                                          batch_size = _BATCH_SIZE, 
                                          shuffle = True
                                         )