import sys
from pathlib import Path
module_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(module_path))
from inflame import build_model
from yaml import safe_load

def main():
   # Load hyperparameters from the news section.
   CONFIG_FILE = module_path / 'inflame' / 'config.yaml'
   try:
      EXAMPLE = sys.argv[1]   
      config = safe_load(open(CONFIG_FILE))[EXAMPLE] 
   except:
      print("ERROR: Proper example name argument wasn't found behind the run script command.")

   # Pick example.
   if EXAMPLE == 'news':
      import usage.news_classify as nc 
      build_model(nc.data, nc.target, config)
   elif EXAMPLE == 'newsgrp':
      import usage.newsgrp_classify as ngc 
      build_model(ngc.data, ngc.target, config)
   else:
      raise Exception("Example not found.")

if __name__ == "__main__":
   main()