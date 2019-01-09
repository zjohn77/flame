"""
Use the trained model to predict holdout classes; then compute the % of the holdout that were correctly classified.
"""
from torch import no_grad, max

class Accuracy:
   '''Generate the model's holdout accuracy rate--by first looping over each holdout batch
   to accumulate the # of correct predictions, and then dividing by sample size.
   '''
   def __init__(self, model, holdout):
      self.holdout = holdout
      with no_grad():
         self.correct_tally = 0
         for x, y in holdout:
            _, yhat = max(model(x).data, 1)
            self.correct_tally += (yhat == y).sum().item()
   
   def __str__(self):      
      return ('The out-of-sample accuracy of the ConvNet:' +
              f' {self.correct_tally / len(self.holdout.dataset) * 100}%')