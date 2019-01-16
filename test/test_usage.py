"""
Test the usage module's functions: extract_data, reshape. 
"""
from unittest import TestCase

from pathlib import Path
import sys
module_path = Path(__file__).resolve().parents[1] ## cd ..
sys.path.insert(0, str(module_path))
from usage import extract_data, reshape

class Usage(TestCase):
   @classmethod
   def setUpClass(cls):
      cls.EXTRACT = extract_data('data/')
   
   def test_extract_data(self):
      '''Should traverse the directory supplied as argument (the "data/" here), and
      load all files into a dict with 5 keys corresponding to the 5 folders under "data/".
      '''
      self.assertEqual(set(Usage.EXTRACT.keys()),
                       {'tech', 'sport', 'politics', 'business', 'entertainment'}
                      )

   def test_reshape(self):
      '''Should convert a wide dataframe to a narrow dataframe through the use of an
      indicator column (the 2nd column here).
      '''
      DATA = {'arts': [4],
              'tech': [6, 9]
             }
      self.assertEqual(reshape(DATA),
                       ([4, 6, 9], [0, 1, 1])
                      )