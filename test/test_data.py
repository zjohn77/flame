"""
Test the inflame.data module's NLP functions for embedding a list of text documents
into a word tensor.
"""
from unittest import TestCase
from pathlib import Path
module_path = Path(__file__).resolve().parents[1] ## cd ..
import sys; sys.path.insert(0, str(module_path))
from inflame.data import NLP

class Data(TestCase):
   @classmethod
   def setUpClass(cls):
      DOCS = ["The most potent three-letter word in history--'why'.",
              "FHA-insured H.E.C.M. first came into being in 1988."
             ]
      cls.embedding = NLP(DOCS).embed()

   def test_NLP(self):
      '''Should embed a list of strings into word vectors.'''
      self.assertTrue(Data.embedding.shape == (2, 600, 50)
                     )
      self.assertTrue(Data.embedding.sum().round(4) == 19.7118
                     )