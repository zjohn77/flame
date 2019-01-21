"""
Defines the main function, which processes raw data, builds model, and prints 
the out-of-sample performance of the model. How each of those actions is implemented is specified
in the 'data', 'models', 'params', and 'train' modules.
"""
from argparse import ArgumentParser  # Pick the corpus to load at the command line.
from corpus4classify import getdata  # Gets the dataset corresponding to its argument: name of the corpus. 

# The 'data' can be plugged with any custom data pipeline as long as 
# it maps (data, target) to (training_batches, validati_batches) like the api below.
from inflame.data import data_pipeline

# The 'train' can be swapped with another module to customize: the optimizer (default: Adam),
# the loss function (default: CrossEntropy), or the accuracy measure (defined by the Accuracy class).
from inflame.train import train_model

## Enables the use of a command line switch to choose the corpus to load. 
parser = ArgumentParser()
parser.add_argument('--corpus', action='store', dest='corpus')
CORPUS_NAME = parser.parse_args().corpus

# The 'bbcnews' or 'newsgrp' are customizable to another model. The params module holds
# the hyperparameters related to the specific corpus.
if CORPUS_NAME == 'bbcnews':
   from inflame.models.bbcnews import ConvNet
   import inflame.params.bbcnews as params
elif CORPUS_NAME == 'newsgrp':
   from inflame.models.newsgrp import ConvNet
   import inflame.params.newsgrp as params
else:
   raise Exception('A valid corpus name was not entered via command line switch.')

def main():
   '''API that processes raw data and returns a pytorch model object that can
   be used to predict the target for new tensors.
   '''
   ## 1. Using CORPUS_NAME passed from CLI, loads data & targe, and shapes data into pytorch Datasets 
   # (batch data according to loader.py to reducing RAM use).
   training_batches, validati_batches = data_pipeline(*getdata(CORPUS_NAME)
                                                     )
   
   ## 2. Calls generic train_model wrapper with the ConvNet model type.                                                 
   return train_model(training_batches,
                      validati_batches,
                      ConvNet(input_length = params.input_length, # similar to pixels in vision
                              channels = params.channels,
                              kernel_size = params.kernel_size,
                              stride = params.stride,
                              padding = int((params.kernel_size - 1) / 2), # constrained to fix spatial size during convolution
                              hidden_layer_nodes = params.hidden_layer_nodes,
                              output_layer_nodes = params.output_layer_nodes
                             ),
                      params.learn_rate,
                      params.n_epochs
                     )

if __name__ == "__main__":
   main()