from utils import *
import numpy as np
from glob import glob
import os
import subprocess
import random
import sys

from sklearn import svm, linear_model, cross_validation

class RankSVM(svm.LinearSVC):
    #def __init__(self, coef):
    #    super(RankSVM, self).__init__(C=coef)
    def getFeatures(self, dataA, datasetB, featureFunction):
        X_trans = []
        y_trans = []
        for i in range(len(dataA) / 2):
            try:
                basename = os.path.basename(dataA[2 * i])
                x1_feat = featureFunction(dataA[2 * i])

                x2_path = datasetB + basename
                if not os.path.isfile(x2_path):
                    continue
                x2_feat = featureFunction(x2_path)
                X_trans.append(np.array(x1_feat) - np.array(x2_feat))
                y_trans.append(1.)

                basename = os.path.basename(dataA[2 * i + 1])
                x1_feat = featureFunction(dataA[2 * i + 1])

                x2_path = datasetB + basename
                if not os.path.isfile(x2_path):
                    continue
                x2_feat = featureFunction(x2_path)
                X_trans.append(np.array(x2_feat) - np.array(x1_feat))
                y_trans.append(-1.)
            except:
                continue

        X_trans = np.array(X_trans)
        y_trans = np.array(y_trans)
        return X_trans, y_trans

    def fit(self, datasetA, datasetB, featureFunction):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        dataA = glob(os.path.join(datasetA, "*.jpg"))
        X_trans, y_trans = self.getFeatures(dataA, datasetB, featureFunction)
        r = np.arange(X_trans.shape[0])
        np.random.shuffle(r)
        print(X_trans.shape)
        n_validate_examples = 0.2*X_trans.shape[0]
        super(RankSVM, self).fit(X_trans[r[:-n_validate_examples]], y_trans[r[:-n_validate_examples]])
        err,len = self.validate(X_trans[r[-n_validate_examples:]], y_trans[r[-n_validate_examples:]])
        print(float(err) / len)
        return self

    def validate(self,  X_trans, y_trans):
        #dataA = glob(os.path.join(datasetA, "*.jpg"))
        #dataA = dataA[len(dataA) / 2:]
        #X_trans, y_trans = self.getFeatures(dataA, datasetB, featureFunction)
        error = 0.
        for i in range(X_trans.shape[0]):
            #print (np.dot(X_trans[i], self.coef_.ravel()))
            #print(np.dot(X_trans[i], self.coef_.ravel()), y_trans[i])
            if np.dot(X_trans[i], self.coef_.ravel()) * y_trans[i] < 0:
                error += 1
        return error, X_trans.shape[0]

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        The item is given such that items ranked on top have are
        predicted a higher ordering (i.e. 0 means is the last item
        and n_samples would be the item ranked on top).
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.ravel()))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def save(self, path):
        np.savetxt(path, self.coef_.ravel())

if __name__ == '__main__':
    step = int(sys.argv[1])
    data_dir = sys.argv[2]
    r = RankSVM()
    r.fit(data_dir + "test_AtoB_"+str(step) + "/",
           data_dir + "test_BtoA_" + str(step)+"/", Gist)
    np.savetxt(data_dir + str(step) + ".txt", r.coef_.ravel())



