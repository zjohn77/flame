import sys
from pathlib import Path
module_path = Path(__file__).resolve().parents[2] ## cd ../..
sys.path.insert(0, str(module_path))
from flame import main
from extract_data import extract_data
from transform_data import reshape
from yaml import load

def build_model():
   '''Entire model building process:
      1. point extract_data to the data location
      2.   
   '''
   data, target = reshape(extract_data('data/'))
   return main(data, target, 
               load(open('config.yaml'))['news'] # Load hyperparameters from the news section 
                                                 # of "config.yaml".
              )

if __name__ == '__main__':
   build_model()