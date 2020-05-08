import numpy as np
from abc import abstractmethod
from sklearn.svm import SVC


class GeneralClassifier:
    """
    Implementation of a half-space classifier using the perceptron algorithm.
    TODO: add description of algorithm
    """

    class Model:
        def

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        """
        Given a training set as X in R^(d x m) and y in {-1, 1}, this method learns the
        parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model.
        :param X:   input in R^(d x m)
        :param y:   ground truth tag in {-1, 1}
        :return:    None
        """
        self.model = self._train(X, y)

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    @abstractmethod
    def _train(self, X, y):
        pass

