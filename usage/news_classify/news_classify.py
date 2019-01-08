import sys
from pathlib import Path
module_path = Path(__file__).resolve().parents[2] ## cd ../..
sys.path.insert(0, str(module_path))

from etl import extract_data, reshape
from flame import main

def bbc_data_pipeline():
   return reshape(extract_data('data/')
                 )

if __name__ == '__main__':
   data, target = bbc_data_pipeline()
   model = main(data, target)