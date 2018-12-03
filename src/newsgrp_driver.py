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

KERNEL_SIZE = 3
NEWSGROUPS = fetch_20newsgroups(subset='all')

training_batches, validati_batches = data_pipeline(NEWSGROUPS.data, 
                                                   NEWSGROUPS.target
                                                  )

model = train_model(training_batches, 
                     validati_batches,
                     ConvNet(input_height = 800,
                              kernel_size = KERNEL_SIZE,
                              stride = 1,
                              padding = int((KERNEL_SIZE - 1) / 2)
                           )
                  )                 