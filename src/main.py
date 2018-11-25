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
from util import mk_dataloader
from torch.nn import CrossEntropyLoss
import torch as t

## make a convnet object configured to hyperparameters
KERNEL_SIZE = 3
convnet = ConvNet(input_dim = 28,
                  kernel_size = KERNEL_SIZE,
                  stride = 1,
                  padding = int((KERNEL_SIZE - 1) / 2)
               )

## fit model
fit(model = convnet, 
    training = training_batches, 
    optimizer = t.optim.Adam(convnet.parameters(), 
                             lr = 0.001   ## lr: the learn rate
                            ),   
    loss_func = CrossEntropyLoss(),
    n_epochs = 2
   )

## check accuracy
accuracy = Accuracy(convnet, holdout_batches)
print(f'{accuracy}')   