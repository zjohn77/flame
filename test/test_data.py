import unittest
from numpy import array

## prepend the parent directory of this file to path
import sys
from os.path import abspath, join, dirname, pardir
sys.path.insert(0, abspath(join(dirname(__file__), 
                                pardir
                               )
                          )
               )

## expose all public functions from src.data.util
from src.data.util import doc2matrix


class Data(unittest.TestCase):   
    '''test all functions in the data module'''
    def test_doc2matrix(self):
        self.assertEqual(doc2matrix('U.K. startup worth $1 billion').sum(),
                         7.882058
                        )                        