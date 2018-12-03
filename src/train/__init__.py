import torch.optim as optim
from torch.nn import CrossEntropyLoss
from .train import fit
from .validate import Accuracy

def train_model(training_batches, validati_batches, model):
    '''The "fit" function here updates the "model" variable in place.
    ''' 
    fit(model = model, 
        training = training_batches, 
        optimizer = optim.Adam(model.parameters(), 
                            lr = 0.001   ## lr: the learn rate
                            ),
        loss_func = CrossEntropyLoss(),
        n_epochs = 20
    )

    ## 3. check accuracy
    accuracy = Accuracy(model, validati_batches)
    print(f'{accuracy}')   

    return model