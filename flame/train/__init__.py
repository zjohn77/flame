import torch.optim as optim
from torch.nn import CrossEntropyLoss
from .train import fit
from .validate import Accuracy

def train_model(training_batches, validati_batches, model, learn_rate, n_epochs):
    '''The "fit" function here updates the "model" variable in place.
    ''' 
    fit(model = model, 
        training = training_batches, 
        optimizer = optim.Adam(model.parameters(), 
                            lr = learn_rate   ## lr: the learn rate
                            ),
        loss_func = CrossEntropyLoss(),
        n_epochs = n_epochs
    )

    ## 3. check accuracy
    accuracy = Accuracy(model, validati_batches)
    print(f'{accuracy}')   

    return model