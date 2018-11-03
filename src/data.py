from torchvision import datasets, transforms
import torch as t

# Declare constants.
BATCH_SIZE = 100
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
TEST_DATASET = datasets.MNIST(root = '../data', 
                                train = False, 
                                download = False,
                                transform = TRANSF
                                )
train_loader = t.utils.data.DataLoader(dataset = TRAIN_DATASET,
                                        batch_size = BATCH_SIZE, 
                                        shuffle = True
                                      )
test_loader = t.utils.data.DataLoader(dataset = TEST_DATASET,
                                        batch_size = BATCH_SIZE, 
                                        shuffle = True
                                     )