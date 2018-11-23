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
# from src.data.util import doc2matrix
from src.data import doc2matrix, fetch_data

class Data(unittest.TestCase):   
    '''test all functions in the data module'''
    def test_doc2matrix(self):
        '''doc2matrix should tokenize the phrase into 6 tokens, then embed them
        in 300D GloVe vectors. Check that the shape and total are as expected.
        '''
        embedding_matrix = doc2matrix('U.K. startup worth $1 billion')
        self.assertEqual(embedding_matrix.shape, (6, 300))
        self.assertAlmostEqual(embedding_matrix.sum(),
                               7.882058,
                               places = 5
                              )            

    def test_fetch_data(self):
        train, test = fetch_data()
        self.assertTrue(train.data.__class__ == list)
        self.assertEqual(len(train.data), 11314)
        self.assertTrue(train.data[0].__class__ == str)

                                                 