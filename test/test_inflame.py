"""
Test the inflame module's functions: embed. 
"""
from unittest import TestCase

from pathlib import Path
import sys
module_path = Path(__file__).resolve().parents[1] ## cd ..
sys.path.insert(0, str(module_path))
from inflame.data import *

class Usage(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = "The most powerful three-letter word in human history--'why'."

    def test_embed(self):
        self.assertEqual(embed(cls.text),
                         ['the', 'most', 'powerful', 'three', 'letter', 'word', 'in', 'human', 'history', 'why']
                        )