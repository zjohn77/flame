from torchvision import datasets, transforms
import torch as t

BATCH_SIZE = 500
MEAN = 0.1307
STANDARD_DEV = 0.3081

TRANSF = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((MEAN,), 
                                                    (STANDARD_DEV,)
                                                )
                            ])
TRAIN_DATASET = datasets.MNIST(root = '../data', 
                                train = True, 
                                download = True,
                                transform = TRANSF
                                )
HOLDOUT_DATASET = datasets.MNIST(root = '../data', 
                                train = False, 
                                download = False,
                                transform = TRANSF
                                )
train_loader = t.utils.data.DataLoader(dataset = TRAIN_DATASET,
                                        batch_size = BATCH_SIZE, 
                                        shuffle = True
                                      )
holdout_loader = t.utils.data.DataLoader(dataset = HOLDOUT_DATASET,
                                        batch_size = BATCH_SIZE, 
                                        shuffle = True
                                     )