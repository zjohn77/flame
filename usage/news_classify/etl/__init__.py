from .extract_bbc import extract_data
from .transform_bbc import reshape

def bbc_data_pipeline():
   return reshape(extract_data('data/bbc')
                 )