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
    @classmethod
    def setUpClass(cls):
        cls.HASHMAP = {'row1': ['foo bar', 0],
                       'row2': ['baz', 1]
                      }

    def test_fetch_data(self):
        '''should pull the right number of records and right type of records
        from the 20newsgroups dataset as provided by sklearn
        '''
        train = fetch_data()
        self.assertTrue(train.data.__class__ == list)
        self.assertTrue(len(train.data) == 18846)
        self.assertTrue(train.data[0].__class__ == str)
    
    def test_to_hashmap(self):
        '''Should reshape 3 lists into 1 dict.
        '''
        BUILT_HASHMAP = to_hashmap(['row1', 'row2'], 
                                   ['foo bar', 'baz'],
                                   range(2)
                                  )
        self.assertTrue(BUILT_HASHMAP == self.HASHMAP)

    # def test_discard_long_docs(self):
    #     '''Should return a list containing only docs that have <= 5 tokens. 
    #     '''
    #     DOCS = ['U.S. unicorn', 'U.K. startup worth $1 billion']
    #     self.assertTrue(discard_long_docs(DOCS, 1) == ['U.S. unicorn'])

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
        '''1. Define an example of X & y, and use these to create "dataset".
           2. Feed "dataset" into assertions about the feature scaling and about the
           tensor dataset that was created.
        '''
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

    def test_mk_dataloader(self):
        pass
