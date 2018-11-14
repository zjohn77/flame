from data import train_loader, holdout_loader
from models import ConvNet
import torch as t
from torch.nn import CrossEntropyLoss

def fit(model, training, optimizer, loss_func, n_epochs):   
   '''trains a network when given features (x) and labels (y).
   '''
   for e in range(n_epochs):      
      for x, y in training: ## loop over batches
         optimizer.zero_grad()
         loss = loss_func(model(x), y) 
         loss.backward()
         optimizer.step()
   print('Finished Training')