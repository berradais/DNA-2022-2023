import numpy as np
import utils


class DecisionStumpEquality:

    def __init__(self):
        '''
        Local parameters of the algorithm
        '''
        self._minError = None
        self._splitVariable = None
        self._splitValue = None
        self._splitSat = None
        self._splitNot =None


    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 

        self._splitSat = y_mode

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        self._minError = np.sum(y != y_mode)/N

        # Loop over features looking for the best split
        X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] == value])
                y_not = utils.mode(y[X[:,d] != value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] != value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)/N

                # Compare to minimum error so far
                if errors < self._minError:
                    # This is the lowest error, store this value
                    self._minError = errors
                    self._splitVariable = d
                    self._splitValue = value
                    self._splitSat = y_sat
                    self._splitNot = y_not

    
    def predict(self, X):
        N, D = X.shape
        X = np.round(X)

        if self._splitVariable is None:
            return self._splitSat * np.ones(N)

        yhat = np.zeros(N)

        for m in range(N):
            if X[m, self._splitVariable] == self._splitValue:
                yhat[m] = self._splitSat
            else:
                yhat[m] = self._splitNot

        return yhat


