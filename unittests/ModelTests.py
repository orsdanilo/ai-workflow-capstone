#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from model import *




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        data_dir = os.path.join(".","data","cs-train")
        model_train(data_dir,test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test-all-0_1.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        data_dir = os.path.join(".","data","cs-production")
        all_data, all_models  = model_load(data_dir=data_dir, training=False)
        
        self.assertTrue('predict' in dir(all_models['all']))
        self.assertTrue('fit' in dir(all_models['all']))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        data_dir = os.path.join(".","data","cs-production")
        all_data, all_models  = model_load(data_dir=data_dir, training=False)

    
        ## ensure that a query can be passed
        query = {'country': 'all',
                 'year': '2019',
                 'month': '10',
                 'day': '8'
        }

        result = model_predict(**query, test=True)
        y_pred = result['y_pred']
        self.assertTrue(isinstance(y_pred[0], float))

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
