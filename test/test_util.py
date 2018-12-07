"""
Unit test src.util module's functions: reshape. 
"""
from unittest import TestCase
from pathlib import Path
from sys import path
module_path = Path(__file__).resolve().parents[1] ## cd ..
path.insert(0, str(module_path))
from src.util as *

class Util(TestCase):   
    @classmethod
    def setUpClass(cls):
        cls.data = {'arts': [4],
                    'tech': [6, 9]
                   }
        cls.extracted_data = extract_data('data/bbc')
    
    def test_extract_data(self):
        self.assertTrue(len(Util.extracted_data) == 5)

    def test_reshape(self):
        self.assertEqual(reshape(Util.data),
                         ([4, 6, 9], [0, 1, 1])
                        )