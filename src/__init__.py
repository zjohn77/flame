"""
Fit a ConvNet to training sample and output its accuracy on holdout sample.
"""
from yaml import load
from .data import data_pipeline
from .train import train_model
from .models.newsgrp import ConvNet

def build_model(data, target):
   '''API that takes raw data in lists and returns a pytorch model object.
   '''
   ## Set the dict CONFIG to the hyperparameters loaded from the newsgrp section of "config.yaml".
   CONFIG = load(open('config.yaml'))['newsgrp']

   ## Shape data into pytorch Datasets (batch data according to loader.py to reducing RAM use).
   training_batches, validati_batches = data_pipeline(data, 
                                                      target
                                                     )
   ## Call generic train_model wrapper with the ConvNet model type.                                                   
   model = train_model(training_batches,
                       validati_batches,
                       ConvNet(input_height = 800,
                               kernel_size = CONFIG['kernel_size'],
                               stride = CONFIG['stride'],
                               padding = int((CONFIG['kernel_size'] - 1) / 2),   # constrained to fix spatial size during convolution
                               hidden_layer_nodes = CONFIG['hidden_layer_nodes']
                               ),
                       CONFIG['learn_rate'],
                       CONFIG['n_epochs']
                      )
   return model