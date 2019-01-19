"""
Defines the build_model function, which processes raw data, builds model, and prints 
the out-of-sample performance of the model. How each of those actions is implemented is specified
in the 'data', 'models', and 'train' modules respectively.
"""
# .bbcnews customizable to another model.
from .models.bbcnews import ConvNet    

# .data can be plugged with any custom data pipeline as long as 
# it maps (data, target) to (training_batches, validati_batches) like the api below.
from .data import data_pipeline     

# .train can be swapped with another module to customize: the optimizer (default: Adam),
# the loss function (default: CrossEntropy), or the accuracy measure (defined by the Accuracy class).
from .train import train_model

def build_model(data, target, config):
   '''API that takes raw data in lists and returns a pytorch model object that can
   be used to predict the target for new tensors.
   '''
   ## 1. Shape data into pytorch Datasets (batch data according to loader.py to reducing RAM use).
   training_batches, validati_batches = data_pipeline(data, 
                                                      target
                                                     )
   
   ## 2. Call generic train_model wrapper with the ConvNet model type.                                                 
   model = train_model(training_batches,
                       validati_batches,
                       ConvNet(input_length = config['input_length'], # similar to pixels in vision
                               channels = config['channels'],
                               kernel_size = config['kernel_size'],
                               stride = config['stride'],
                               padding = int((config['kernel_size'] - 1) / 2), # constrained to fix spatial size during convolution
                               hidden_layer_nodes = config['hidden_layer_nodes'],
                               output_layer_nodes = config['output_layer_nodes']
                              ),
                       config['learn_rate'],
                       config['n_epochs']
                      )
   
   return model