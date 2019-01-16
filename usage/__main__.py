import sys
from pathlib import Path
module_path = Path(__file__).resolve().parents[1] # cd ../..
sys.path.insert(0, str(module_path))
from inflame import build_model
from yaml import safe_load

CONFIG_FILE = module_path / 'usage' / 'config.yaml'
EXAMPLE = sys.argv[1]
config = safe_load(open(CONFIG_FILE))[EXAMPLE] # Load hyperparameters from the news section 

if EXAMPLE == 'news':
   import news_classify as nc 
   build_model(nc.data, nc.target, config)
elif EXAMPLE == 'newsgrp':
   import newsgrp_classify as ngc 
   build_model(ngc.data, ngc.target, config)
else:
   raise Exception('example not found')