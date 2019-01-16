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
from inflame import main

from sklearn.datasets import fetch_20newsgroups

NEWSGROUPS = fetch_20newsgroups(subset='test')

def runner(config):
   '''main function'''
   return build_model(NEWSGROUPS.data, NEWSGROUPS.target, 
               config
              )