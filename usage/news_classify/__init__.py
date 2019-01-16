"""
Entire model building process:
   1. handle input: read all files into a dict by pointing extract_data to the data location
   2. reshape data: transform dict into 2 lists -- target (type of news) & features (news content)
   3. train model: deep learn on training cases; evaluate OOS prediction accuracy.
   4. return the trained pytorch model object for further analysis.
"""
from .extract_data import extract_data
from .transform_data import reshape


### The API ###
data, target = reshape(extract_data('data/'))