from pathlib import Path
import os; os.chdir(str(Path(__file__).resolve().parents[0]))

# module_path = Path(__file__).resolve().parents[0]
# import sys; sys.path.insert(0, str(module_path))

from .__init__ import build_model
from yaml import safe_load
from argparse import ArgumentParser
from corpus4classify import controller

# Pick the corpus to load at the command line by supplying the --corpus flag.
parser = ArgumentParser()
parser.add_argument('--corpus', action='store', dest='corpus')
CORPUS = parser.parse_args().corpus

def main():
   ## 1. Load data & target from the corpus flagged by the command line argument.
   data, target = controller(CORPUS)

   ## 2. Load hyperparameters from the section named CORPUS.
   config = safe_load(open('config.yaml'))[CORPUS] 

   return build_model(data, target, config)

if __name__ == "__main__":
   main()