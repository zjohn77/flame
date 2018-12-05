"""
Set hyperparameters; then, fit a ConvNet to training sample; finally, output its accuracy on holdout sample.
Hyperparameters: 
   * padding--constrained to fix spatial size during convolution,
"""
from sklearn.datasets import fetch_20newsgroups
from data import data_pipeline
from models.newsgrp import ConvNet
from train import train_model
from yaml import load
from util import extract_data

CONFIG = load(open('config.yaml'))['newsgrp']
BBC = extract_data('data/bbc')

training_batches, validati_batches = data_pipeline(NEWSGROUPS.data, 
                                                   NEWSGROUPS.target
                                                  )
model = train_model(training_batches,
                    validati_batches,
                    ConvNet(input_height = 800,
                            kernel_size = CONFIG['kernel_size'],
                            stride = CONFIG['stride'],
                            padding = int((CONFIG['kernel_size'] - 1) / 2),
                            hidden_layer_nodes = CONFIG['hidden_layer_nodes']
                           ),
                    CONFIG['learn_rate'],
                    CONFIG['n_epochs']
                   )