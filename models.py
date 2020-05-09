import numpy as np
from abc import abstractmethod, ABC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ================ in-class models ================
class Model(ABC):
    """
    base class
    """
    def __init__(self):
        self._weights = None

    def score(self, X: np.ndarray, y: np.ndarray):
        """
        estimates model score
        :param X:   unlabeled test set X in R^(d x m) where d = #features, m = #samples
        :param y:   ground truth labels
        :return:    dictionary with the following fields:
                    - num samples: number of samples in the test set
                    - error: error (misclassification) rate
                    - accuracy: accuracy
                    - FPR: false positive rate
                    - TPR: true positive rate
                    - precision: precision
                    - recall: recall
        """
        # init output dict
        output = dict()

        # get predictions
        predictions = self.predict(X).flatten()
        y = y.flatten()

        # estimate error types - asserting all y in {-1,1}
        FN = np.where(predictions - y == 2)[0].shape[0]
        TP = np.where(predictions + y == 2)[0].shape[0]
        TN = np.where(predictions + y == -2)[0].shape[0]
        FP = np.where(predictions - y == -2)[0].shape[0]
        P = np.where(y == 1).shape[0]
        N = np.where(y == -1).shape[0]

        # update dictionary
        output["num_samples"] = predictions.shape[0]
        output["error"] = (FP + FN) / (P + N)
        output["accuracy"] = (TP + TN) / (P + N)
        output["FPR"] = FP / N
        output["TPR"] = TP / P
        output["precision"] = TP / (TP + FP)
        output["recall"] = TP / P

        return output

    def get_weights(self):
        return self._weights

    @abstractmethod
    def predict(self, X: np.ndarray):
        """
        return predicted class given X samples
        :param X:   unlabeled test set X in R^(d x m) where d = #features, m = #samples
        :return:    predicted labels vector of length m, matching the given samples
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Given a training set as X in R^(d x m) and y in {-1, 1}, this method learns the
        parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model.
        :param X:   input in R^(d x m)
        :param y:   ground truth tag in {-1, 1}
        :return:    None
        """
        pass

    # -- class utils
    @staticmethod
    def to_col_vec(y):
        y = np.asarray(y).flatten()
        return y.reshape((y.shape[0], 1))


class ModelWrapper:
    """
    wrapper class to restrict API on external library model
    """
    def __init__(self, model):
        self._model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Given a training set as X in R^(d x m) and y in {-1, 1}, this method learns the
        parameters of the model and stores the trained model (namely, the variables that
        define hypothesis chosen) in self.model.
        :param X:   input in R^(d x m)
        :param y:   ground truth tag in {-1, 1}
        :return:    None
        """
        return self._model.fit(X, y)

    def predict(self, X: np.ndarray):
        """
        :param X:   unlabeled test set X in R^(d x m) where d = #features, m = #samples
        :return:    predicted labels vector of length m, matching the given samples
        """
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        """
        estimates model score
        :param X:   unlabeled test set X in R^(d x m) where d = #features, m = #samples
        :param y:   ground truth labels
        :return:    dictionary with the following fields:
                    - num samples: number of samples in the test set
                    - error: error (misclassification) rate
                    - accuracy: accuracy
                    - FPR: false positive rate
                    - TPR: true positive rate
                    - precision: precision
                    - recall: recall
        """
        return self._model.score(X, y)

    def get_weights(self):
        weights = Model.to_col_vec(self._model.coef_)
        return weights


# ================ Classifiers ================
class Perceptron:
    """
    Implementation of a half-space classifier using the perceptron algorithm.
    The Algorithm:
    1. init w = (0,...,0)^T
    2. for t = 1,2,... :
        2.1. if for some i : y[i] * (w^T @ x[i]) <= 0:
            w = w + y[i]x[i]
        2.2. else:
            return w
    """
    def __init__(self):
        self.model = self.PerceptronModel()

    class PerceptronModel(Model, ABC):
        def predict(self, X: np.ndarray):
            """
            :param X:   unlabeled test set X in R^(d x m) where d = #features, m = #samples
            :return:    predicted labels vector of length m, matching the given samples
            """
            # get weights
            w = self._weights
            assert w is not None, "There are no fitted weights for this model\n" \
                                  "Please use fit method to estimates weights"

            # return prediction
            res = np.matmul(X.T, w).flatten()
            idx = np.where(res == 0)[0]
            if idx.shape[0] != 0:
                res[idx] = np.random.rand(idx.shape[0])
            return np.sign(res)

        def fit(self, X: np.ndarray, y: np.ndarray):
            """
            Given a training set as X in R^(d x m) and y in {-1, 1}, this method learns the
            parameters of the model and stores the trained model (namely, the variables that
            define hypothesis chosen) in self.model.
            :param X:   input in R^(d x m)
            :param y:   ground truth tag in {-1, 1}
            :return:    None
            """
            y = y.flatten()

            # init coefficients vector
            w = np.zeros((X.shape[0], 1))

            # estimation loop
            j = 0
            while True:
                # enforce endless loop
                assert j < 1e6, "exceeded 10^6 iterations"
                j += 1

                tmp = (Model.to_col_vec(y) * np.matmul(X.T, w)).flatten()
                idx = np.where(tmp <= 0)[0]

                # exit term
                if idx.shape[0] == 0:
                    break

                # else - update weights
                w = w + y[idx[0]] * Model.to_col_vec(X[:, idx[0]])

            # update coefficients
            self._weights = w
            return


class LDA:
    """
    Implementation of LDA classifier
    """

    def __init__(self):
        self.model = self.LDAModel()

    class LDAModel(Model, ABC):
        def __init__(self):
            super().__init__()
            self._mu = None
            self._sigma_i = None
            self._pr_y1 = None

        def predict(self, X: np.ndarray):
            """
            :param X:   unlabeled test set X in R^(d x m) where d = #features, m = #samples
            :return:    predicted labels vector of length m, matching the given samples
            """
            # check init of parameters: mu, sigma_i, pr_y1
            assert self._mu is not None and self._sigma_i is not None and self._pr_y1 is not None, \
                "parameters are not initiated\n" \
                "Please use fit method."

            # get parameters
            mu_p, mu_n = self._mu[0], self._mu[1]

            # estimate labels
            delta = self._delta(X, mu_n, self._sigma_i, 1 - self._pr_y1) - self._delta(X, mu_p, self._sigma_i, self._pr_y1)

            # return estimation
            return np.array(delta / np.abs(delta), dtype=int).reshape(delta.shape)

        def fit(self, X: np.ndarray, y: np.ndarray):
            """
            Given a training set as X in R^(d x m) and y in {-1, 1}, this method learns the
            parameters of the model and stores the trained model (namely, the variables that
            define hypothesis chosen) in self.model.
            :param X:   input in R^(d x m)
            :param y:   ground truth tag in {-1, 1}
            :return:    None
            """
            # get needed numerical specs
            num_1_tag = np.where(y == 1)[0].shape[0]
            num_samples = X.shape[1]

            # calculate Pr(y = 1)
            self._pr_y1 = num_1_tag / num_samples

            # calculate mu
            mu = []
            for i in [-1, 1]:
                idx = np.where(y == i)[0]

                # add mean vector - calculated coordinate-wise
                mu.append(Model.to_col_vec(np.sum(X[:, idx], axis=1) / num_samples))
            self._mu = mu

            # calculate common sigma inverse
            X_yp = X[:, np.where(y == 1)[0]]  # all samples with tag: y = 1
            X_yn = X[:, np.where(y == -1)[0]]  # all samples with tag: y = -1
            d = X_yp.shape[0]
            sigma = np.zeros((d, d))
            for i in range(X_yp.shape[1]):  # sum (x_i-μ_y )@(x_i-μ_y )^T
                vec = Model.to_col_vec(X_yp[:, i] - mu[1].flatten())
                sigma = sigma + np.matmul(vec, vec.T)
            for i in range(X_yn.shape[1]):  # sum (x_i-μ_y )@(x_i-μ_y )^T
                vec = Model.to_col_vec(X_yn[:, i] - mu[0].flatten())
                sigma = sigma + np.matmul(vec, vec.T)
            sigma = sigma / (y.shape[0] - 2)

            # store the inverse matrix
            self._sigma_i = np.linalg.inv(sigma)
            return

        @staticmethod
        def _delta(X: np.ndarray, mu: np.ndarray, sigma_i: np.ndarray, pr_y: float):
            """
            calculates delta function matching: x^T @ Σ^(-1) @ μ - 0.5 * μ^T@Σ^(-1)@μ+ ln(Pr(y))
            :param X: input matrix
            :param mu: mean vector
            :param sigma_i: inverse covariance matrix
            :return: delta function value
            """
            # force col vectors
            mu = Model.to_col_vec(mu)

            # return delta function value
            A = np.matmul(X.T, np.matmul(sigma_i, mu))
            B = 0.5 * np.matmul(mu.T, np.matmul(sigma_i, mu)).flatten()[0]
            C = np.log(pr_y)
            return A - B + C


class SVM:
    """
    Implementation of SVM classifier (wrapper) using sklearn SVC library
    """
    def __init__(self, C=1e10, kernel='linear'):
        self.model = ModelWrapper(SVC(C, kernel))


class Logistics:
    """
    Implementation of LogisticRegression classifier (wrapper) using sklearn
    """
    def __init__(self, solver='liblinear'):
        self.model = ModelWrapper(LogisticRegression(solver))


class DesicionTree:
    """
    Implementation of Desicion Tree classifier (wrapper) using sklearn
    """
    def __init__(self, max_depth=1):
        self.model = ModelWrapper(DecisionTreeClassifier(max_depth=max_depth))
