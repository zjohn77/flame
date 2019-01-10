"""
Test the usage module's functions: extract_data, reshape. 
"""
from unittest import TestCase

from pathlib import Path
import sys
module_path = Path(__file__).resolve().parents[1] ## cd ..
sys.path.insert(0, str(module_path))
from usage.news_classify import *

class Usage(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extracted_data = extract_data('data/')
        cls.data = {'arts': [4],
                    'tech': [6, 9]
                   }
    def test_extract_data(self):
        self.assertTrue(len(Usage.extracted_data) == 5)

    def test_reshape(self):
        self.assertEqual(reshape(Usage.data),
                         ([4, 6, 9], [0, 1, 1])
                        )