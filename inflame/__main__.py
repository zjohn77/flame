"""
Defines the main function, which processes raw data, builds model, and prints 
the out-of-sample performance of the model. How each of those actions is implemented is specified
in the 'data', 'models', and 'train' modules respectively.
"""
# from pathlib import Path
# import sys; sys.path.insert(0, Path(__file__).absolute().parent)
# import os
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Gets the dataset corresponding to its argument: name of the corpus. 
from corpus4classify import getdata  

# The 'data' can be plugged with any custom data pipeline as long as 
# it maps (data, target) to (training_batches, validati_batches) like the api below.
from inflame.data import data_pipeline   

# The 'train' can be swapped with another module to customize: the optimizer (default: Adam),
# the loss function (default: CrossEntropy), or the accuracy measure (defined by the Accuracy class).
from inflame.train import train_model

# The 'bbcnews' or 'newsgrp' are customizable to another model.
from inflame.models.newsgrp import ConvNet
import inflame.params.newsgrp as params  # Get the hyperparameters related to this corpus.

# Pick the corpus to load at the command line.
from argparse import ArgumentParser

def main():
   '''API that takes raw data in lists and returns a pytorch model object that can
   be used to predict the target for new tensors.
   '''
   ## 1. Enables the use of a command line switch to choose the corpus to load. 
   parser = ArgumentParser()
   parser.add_argument('--corpus', action='store', dest='corpus')
   
   ## 2. Load data & target , 
   # and shape data into pytorch Datasets (batch data according to loader.py to reducing RAM use).
   training_batches, validati_batches = data_pipeline(*getdata(parser.parse_args().corpus)
                                                     )
   
   ## 3. Call generic train_model wrapper with the ConvNet model type.                                                 
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