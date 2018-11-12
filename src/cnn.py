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
    ## hyperparameters
    KERNEL_SIZE = 5
    STRIDE = 1
    PADDING = int((KERNEL_SIZE - 1) / 2)
    convnet = ConvNet(KERNEL_SIZE, STRIDE, PADDING)

    LEARN_RATE = 0.001
    OPTIMIZER = t.optim.Adam(convnet.parameters(), 
                            lr = LEARN_RATE)   
    LOSS_FUNC = CrossEntropyLoss()
    N_EPOCHS = 2
    
    fit(convnet, 
        train_loader, 
        OPTIMIZER, 
        LOSS_FUNC,
        N_EPOCHS
    )
    accuracy = Accuracy(convnet, holdout_loader)
    print(f'{accuracy}')   