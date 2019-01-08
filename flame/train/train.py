"""
Trains a deep neural network when passed these 5 arguments:
      * a model instance, 
      * training sample,
      * optimizer (commonly Adam or SGD),
      * loss function
      * number of epochs
"""
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