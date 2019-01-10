"""
Entire model building process:
   1. train model: deep learn on training cases; evaluate OOS prediction accuracy.
   2. return the trained model object for further analysis.
"""
## import the function to build a model
import sys
from pathlib import Path
module_path = Path(__file__).resolve().parents[2] # cd ../..
sys.path.insert(0, str(module_path))
from flame import main

from sklearn.datasets import fetch_20newsgroups
from yaml import safe_load

NEWSGROUPS = fetch_20newsgroups(subset='test')
CONFIG_FILE = module_path / 'usage' / 'config.yaml'
CONFIG_SECTION = 'newsgrp'

def build_model():
   '''main function'''
   return main(NEWSGROUPS.data, NEWSGROUPS.target, 
               safe_load(open(CONFIG_FILE))[CONFIG_SECTION] # Load hyperparameters from the newsgrp section 
                                                       # of "config.yaml".
              )

build_model()