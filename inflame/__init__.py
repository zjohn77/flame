"""
Fit a ConvNet to training sample and output its accuracy on holdout sample.
"""
from .data import data_pipeline
from .models.news import ConvNet    # .news customizable to another model
from .train import train_model

def build_model(data, target, config):
   '''API that takes raw data in lists and returns a pytorch model object.
   '''
   ## Shape data into pytorch Datasets (batch data according to loader.py to reducing RAM use).
   training_batches, validati_batches = data_pipeline(data, 
                                                      target
                                                     )
   ## Call generic train_model wrapper with the ConvNet model type.                                                   
   model = train_model(training_batches,
                       validati_batches,
                       ConvNet(input_length = 600, # similar to pixels in vision
                               channels = 50,
                               kernel_size = config['kernel_size'],
                               stride = config['stride'],
                               padding = int((config['kernel_size'] - 1) / 2), # constrained to fix spatial size during convolution
                               hidden_layer_nodes = config['hidden_layer_nodes']
                               ),
                       config['learn_rate'],
                       config['n_epochs']
                      )
   return model