"""
Set hyperparameters; then, fit a ConvNet to training sample; finally, output its accuracy on holdout sample.
Hyperparameters: 
   * KERNEL_SIZE = 3, 
   * stride = 1, 
   * padding--constrained to fix spatial size during convolution,
   * optimizer = Adam, 
   * learn rate = 0.001, 
   * loss_func = CrossEntropyLoss(),
   * n_epochs = 20.
"""
from sklearn.datasets import fetch_20newsgroups
from data import data_pipeline
from models.newsgrp import ConvNet
from train import train_model
from yaml import load

CONFIG = load(open('config.yaml'))['newsgrp']
NEWSGROUPS = fetch_20newsgroups(subset='all')

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