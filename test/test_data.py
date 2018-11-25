import unittest
import numpy as np
from torch import tensor, arange, double

## prepend the parent directory of this file to path
import sys
from os.path import abspath, join, dirname, pardir
sys.path.insert(0, abspath(join(dirname(__file__), 
                                pardir
                               )
                          )
               )
from src.data import *  ## expose all public functions from src.data

class Data(unittest.TestCase):   
    '''test all functions in the data module'''
    
    def test_fetch_data(self):
        '''should pull the right number of records and right type of records
        from the 20newsgroups dataset as provided by sklearn
        '''
        train, _ = fetch_data()
        self.assertTrue(train.data.__class__ == list)
        self.assertEqual(len(train.data), 11314)
        self.assertTrue(train.data[0].__class__ == str)
    
    def test_doc2matrix(self):
        '''doc2matrix should tokenize the phrase into 6 tokens, then embed them
        in 300D GloVe vectors. Check that the shape and total are as expected.
        '''
        embedding_matrix = doc2matrix('U.K. startup worth $1 billion')
        self.assertTrue(embedding_matrix.shape == (6, 300))
        self.assertAlmostEqual(embedding_matrix.sum(),
                               7.882058,
                               places = 5
                              )            

    def test_standardize_dataset(self):
        X = np.array([[1, 8],
                      [2, 4],
                      [3, 3]]
                    )
        y = np.arange(len(X))
        dataset = standardize_dataset(X, y)

        ## test for correct object dimensions
        self.assertTrue(len(dataset) == len(X))
        self.assertTrue(len(dataset.tensors) == 2)

        ## test for correct tensor elements
        Z = tensor(zscore(X))
        self.assertEqual((Z - dataset.tensors[0]).max(),
                         0
                        )
        self.assertEqual((arange(3) - dataset.tensors[1]).max(), 
                         0
                        )
