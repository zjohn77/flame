"""
Entire model building process:
   1. train model: deep learn on training cases; evaluate OOS prediction accuracy.
   2. return the trained model object for further analysis.
"""
from sklearn.datasets import fetch_20newsgroups

NEWSGROUPS = fetch_20newsgroups(subset='all')


### The API ###
data, target = NEWSGROUPS.data, NEWSGROUPS.target