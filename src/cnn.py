"""
A function that abstracts the process of writing a 2-dim list to a csv.
"""
from data import train_loader, holdout_loader
from design import ConvNet
import torch as t
from torch import no_grad, max
from torch.nn import CrossEntropyLoss

def fit(model, training, optimizer, loss_func, n_epochs):   
   '''trains a network when given features (x) and labels (y).'''
   for e in range(n_epochs):      
      for x, y in training: ## loop over batches
         optimizer.zero_grad()
         loss = loss_func(model(x), y) 
         loss.backward()
         optimizer.step()
   print('Finished Training')    
   
class Accuracy:
   '''Generate the model's holdout accuracy rate--by first looping over each holdout batch
   to accumulate the # of correct predictions, and then dividing by sample size.'''
   def __init__(self, model, holdout):
      self.model = model
      self.holdout = holdout
      with no_grad():
         self.correct_tally = 0
         for x, y in holdout:
            _, yhat = max(model(x).data, 1)
            self.correct_tally += (yhat == y).sum().item()
   def __str__(self):      
      return ('The out-of-sample accuracy of the ConvNet:' +
              f' {self.correct_tally / len(self.holdout.dataset) * 100}%')

if __name__ == '__main__':
    KERNEL_SIZE = 3
    convnet = ConvNet(input_dim = 28
                      kernel_size = KERNEL_SIZE,
                      stride = 1,
                      padding = int((KERNEL_SIZE - 1) / 2)
                     )
    fit(model = convnet, 
        training = train_loader, 
        optimizer = t.optim.Adam(convnet.parameters(), 
                                 lr = 0.001
                                ),   ## lr: the learn rate
        loss_func = CrossEntropyLoss(),
        n_epochs = 2
       )
    accuracy = Accuracy(convnet, holdout_loader)
    print(f'{accuracy}')   