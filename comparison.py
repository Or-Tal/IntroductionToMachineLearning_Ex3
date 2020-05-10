import matplotlib.pyplot as plt
from models import *
from ex3_utils import *


def draw_points(m: int, pad=False):
    """
    generates m samples of 2D normal distribution ~ N(0,I_(2x2))
    :param m: num of samples in output
    :return: X = (2, m) matrix representing samples drawn,
             y = ground truth labels (m,), given by f(x) = sign([0.3, -0.5].T @ x + 0.1)
    """
    # create X samples
    X_T = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=m)

    # evaluate labels by given function
    y = np.sign(np.matmul(X_T, [[0.3], [-0.5]]) + 0.1).flatten()

    X_T = X_T.T
    if pad:
        X_T = np.concatenate([np.ones((1, X_T.shape[1])), X_T])

    if abs(np.sum(y)) == y.shape[0]:  # case where all samples has the same tag
        return draw_points(m, pad)
    return X_T, y


def Q9():
    """
    Answer to Q9 as described in Ex. sheet
    """
    for m in [5, 10, 15, 25, 70]:
        X, y = draw_points(m)
        X_p = X[:, np.where(y == 1)[0]]
        X_n = X[:, np.where(y == -1)[0]]

        # get models
        perceptron_model = Perceptron().model
        svm_model = SVM().model

        # pad X with ones -> non homogeneous case
        X = np.concatenate([np.ones((1, X.shape[1])), X])

        # fit
        perceptron_model.fit(X, y)
        svm_model.fit(X.T, y.flatten())

        # get weights
        perc_weights = perceptron_model.get_weights()
        svm_weights = svm_model.get_weights()

        # estimate hyperplane
        true_func = gen_hyperplane_func('true')
        perc_func = gen_hyperplane_func('perceptron')
        svm_func = gen_hyperplane_func('svm')
        x_pts = np.linspace(-3, 3, 50)
        y_true = true_func(x_pts)
        y_perc = perc_func(x_pts, perc_weights)
        y_svm = svm_func(x_pts, svm_weights)

        # draw lines
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.add_line(plt.Line2D(x_pts, y_true, alpha=0.7, color='m', linestyle=':'))
        ax.add_line(plt.Line2D(x_pts, y_perc, alpha=0.7, color='green', linestyle=':'))
        ax.add_line(plt.Line2D(x_pts, y_svm, alpha=0.7, color='red', linestyle=':'))

        # plot
        plt.grid()
        ax.scatter(X_p[0, :], X_p[1, :], color="blue")
        ax.scatter(X_n[0, :], X_n[1, :], color="orange")
        plt.legend(["true", "perceptron", "svm", "positive tag", "negative tag"])
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("hyperplanes defined by classifiers, m = {}".format(m))
        plt.show()


def Q10():
    """
    Answer to Q10 as described in Ex. sheet
    """
    estimate_models(models=[Perceptron().model, LDA().model, SVM().model],
                    M=[5, 10, 15, 25, 70], repeat=500, legend=["perceptron", "LDA", "SVM"],
                    Q10=True)
