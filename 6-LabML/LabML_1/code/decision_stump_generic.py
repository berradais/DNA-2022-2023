import numpy as np
import utils



class DecisionStumpEqualityGeneric:
    '''
    The goal of this class is to implement a generic version de the stump equality in terms of:
    - Objective hyperparameter : any user defined method for computing the error
    - Scoring function : reporting directly the score of validation
     
    Inheritance from DecisionStumpEquality class can be also a good idea
    '''

    def __init__(self, loss=utils.loss_l0):
        self._minError = None
        """ YOUR CODE HERE """    
        raise NotImplementedError

    
    def fit(self, X, y):
        """ YOUR CODE HERE """
        raise NotImplementedError

    
    def predict(self, X):
        """ YOUR CODE HERE """
        raise NotImplementedError
    
    def fit_predict(self,X,y):
        """ YOUR CODE HERE """
        raise NotImplementedError
            
    def score(self, X, y):
        """ YOUR CODE HERE """        
        raise NotImplementedError


