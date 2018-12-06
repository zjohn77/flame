# from os.path import abspath
from pathlib import Path

def __navigate(locator):
   '''Navigate to the dir holding the files via relative path fashion, starting from __file__. 
   '''
   # root = abspath(__file__)
   return Path(__file__).parent.parent / locator

def __files2list(files):
   '''Iterates over a files generator. Reads each file as a string. And then append to list.
   '''
   return [file.read_text(errors='ignore') for file in files]

def extract_data(holding_dir):
   '''Go to the dir holding all the data; index the sub-dir names; pack all files in each sub-dir
   into a list. Finally, put the lists in a dict keyed by the sub-dir names.
   '''
   folders = __navigate(holding_dir).iterdir()
   return {folder.stem: __files2list(folder.iterdir()) for folder in folders}