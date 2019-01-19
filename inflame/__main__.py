"""
Defines the main function, which processes raw data, builds model, and prints 
the out-of-sample performance of the model. How each of those actions is implemented is specified
in the 'data', 'models', and 'train' modules respectively.
"""
from pathlib import Path
import sys; sys.path.insert(0, Path(__file__).parent)

# .data can be plugged with any custom data pipeline as long as 
# it maps (data, target) to (training_batches, validati_batches) like the api below.
from data import data_pipeline   

# .bbcnews customizable to another model.
from models.bbcnews import ConvNet  

# .train can be swapped with another module to customize: the optimizer (default: Adam),
# the loss function (default: CrossEntropy), or the accuracy measure (defined by the Accuracy class).
from train import train_model

import configs as cf
from corpus4classify import getdata
from argparse import ArgumentParser

# Pick the corpus to load at the command line by supplying the --corpus flag.
parser = ArgumentParser()
parser.add_argument('--corpus', action='store', dest='corpus')

CORPUS_NAME = parser.parse_args().corpus
CONFIG = cf.CORPUS_NAME  # Get the hyperparameters related to this corpus.

def main():
   '''API that takes raw data in lists and returns a pytorch model object that can
   be used to predict the target for new tensors.
   '''
   ## 1. Load data & target from the corpus flagged by the command line argument.
   ## 1. Shape data into pytorch Datasets (batch data according to loader.py to reducing RAM use).
   training_batches, validati_batches = data_pipeline(*getdata(CORPUS_NAME)
                                                     )
   
   ## 2. Call generic train_model wrapper with the ConvNet model type.                                                 
   return train_model(training_batches,
                       validati_batches,
                       ConvNet(input_length = CONFIG['input_length'], # similar to pixels in vision
                               channels = CONFIG['channels'],
                               kernel_size = CONFIG['kernel_size'],
                               stride = CONFIG['stride'],
                               padding = int((CONFIG['kernel_size'] - 1) / 2), # constrained to fix spatial size during convolution
                               hidden_layer_nodes = CONFIG['hidden_layer_nodes'],
                               output_layer_nodes = CONFIG['output_layer_nodes']
                              ),
                       CONFIG['learn_rate'],
                       CONFIG['n_epochs']
                      )

if __name__ == "__main__":
   main()