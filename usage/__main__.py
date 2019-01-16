import sys
from pathlib import Path
module_path = Path(__file__).resolve().parents[1] # cd ../..
sys.path.insert(0, str(module_path))
from inflame import build_model
from yaml import safe_load

EXAMPLE_NAME = sys.argv[1]
CONFIG_FILE = module_path / 'usage' / 'config.yaml'

if EXAMPLE_NAME == 'news':
   CONFIG_SECTION = 'news'
   config = safe_load(open(CONFIG_FILE))[CONFIG_SECTION] # Load hyperparameters from the news section 
   import news_classify
   news_classify.runner(config)

elif EXAMPLE_NAME == 'newsgrp':
   CONFIG_SECTION = 'newsgrp'
   config = safe_load(open(CONFIG_FILE))[CONFIG_SECTION] # Load hyperparameters from the news section 
   import newsgrp_classify
   newsgrp_classify.runner(config)
   
else:
   raise Exception('example not found')