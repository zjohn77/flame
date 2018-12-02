"""
Set hyperparameters; then, fit a ConvNet to training sample; finally, output its accuracy on holdout sample.
Hyperparameters: 
   * KERNEL_SIZE = 3, 
   * stride = 1, 
   * padding--constrained to fix spatial size during convolution,
   * optimizer = Adam, 
   * learn rate = 0.001, 
   * loss_func = CrossEntropyLoss(),
   * n_epochs = 2.
"""
from models import ConvNet
from train import fit
from validate import Accuracy
from data import training_batches, validati_batches
from torch.nn import CrossEntropyLoss
import torch.optim as optim

## 1. make a convnet object configured to hyperparameters
KERNEL_SIZE = 3
convnet = ConvNet(input_height = 800,
                  kernel_size = KERNEL_SIZE,
                  stride = 1,
                  padding = int((KERNEL_SIZE - 1) / 2)
               )

## 2. fit model
fit(model = convnet, 
    training = training_batches, 
    optimizer = optim.Adam(convnet.parameters(), 
                           lr = 0.001   ## lr: the learn rate
                          ),
    loss_func = CrossEntropyLoss(),
    n_epochs = 20
   )

## 3. check accuracy
accuracy = Accuracy(convnet, validati_batches)
print(f'{accuracy}')   