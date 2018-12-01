import unittest
import numpy as np
from torch import tensor, arange

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
        cls.X = np.array([[[1, 8],
                           [2, 4]],

                          [[3, 7],
                           [5, 1]],
                           
                          [[10, 0],
                           [-9, 6]]]
                        )
        cls.y = np.arange(len(cls.X))

    def test_standardize_dataset(self):
        '''1. Define an example of X & y, and use these to create "dataset".
           2. Feed "dataset" into assertions about the feature scaling and about the
           tensor dataset that was created.
        '''        
        dataset = standardize_dataset(Data.X, Data.y)

        ## test for correct object dimensions
        self.assertTrue(len(dataset) == len(Data.X))
        self.assertTrue(len(dataset.tensors) == 2)

        ## test for correct tensor elements
        Z = tensor(zscore(Data.X))
        self.assertEqual((Z - dataset.tensors[0]).max(),
                         0
                        )
        self.assertEqual((arange(2) - dataset.tensors[1]).max(), 
                         0
                        )

    def test_mk_dataloader(self):
        pass
