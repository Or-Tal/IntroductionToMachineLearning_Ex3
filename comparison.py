import matplotlib.pyplot as plt
from models import *


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
        true_func = _gen_hyperplane_func('true')
        perc_func = _gen_hyperplane_func('perceptron')
        svm_func = _gen_hyperplane_func('svm')
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


def _gen_hyperplane_func(hypothesis='true'):
    """
    this returns a function for estimating the 2nd coordinate
    for drawing the hyperplane line purposes
    """
    if hypothesis == 'true':
        def func(X: np.ndarray):
            X = X.flatten()
            return 0.6 * X + 0.2

        return func
    elif hypothesis == 'perceptron':
        def func(X: np.ndarray, w: np.ndarray):
            X = X.flatten()
            w = w.flatten()
            if w.shape[0] == 3:  # non homogeneous case
                return -w[0] / w[2] - X * w[1] / w[2]
            return - X * w[0] / w[1]

        return func
    elif hypothesis == "svm":
        def func(X: np.ndarray, w: np.ndarray, b=0):
            X = X.flatten()
            w = w.flatten()
            if w.shape[0] == 3:  # non homogeneous case
                return - b - w[0] / w[2] - X * w[1] / w[2]
            return - b - X * w[0] / w[1]

        return func
    return


def Q10():
    """
    Answer to Q10 as described in Ex. sheet
    """
    perceptron_acc = []
    svm_acc = []
    lda_acc = []
    k = 1000
    M = [5, 10, 15, 25, 70]
    for m in M:
        tmp_perc_acc, tmp_svm_acc, tmp_lda_acc = [], [], []
        for i in range(500):
            # draw train and test sets,
            # Note: Implementation of draw_points enforces existence of {-1,1} tags
            train_X, train_y = draw_points(m, pad=True)
            test_X, test_y = draw_points(k, pad=True)

            # get models
            perceptron_model = Perceptron().model
            lda_model = LDA().model
            svm_model = SVM().model

            # fit
            perceptron_model.fit(train_X, train_y)
            lda_model.fit(train_X, train_y)
            svm_model.fit(train_X.T, train_y)

            # predict
            perceptron_predictions = perceptron_model.predict(test_X).flatten()
            lda_predictions = lda_model.predict(test_X).flatten()
            svm_predictions = svm_model.predict(test_X.T).flatten()

            # estimate accuracy
            perceptron_correct = np.where(test_y - perceptron_predictions == 0)[0].shape[0]
            lda_correct = np.where(test_y - lda_predictions == 0)[0].shape[0]
            svm_correct = np.where(test_y - svm_predictions == 0)[0].shape[0]
            tmp_perc_acc.append(perceptron_correct / k)
            tmp_lda_acc.append(lda_correct / k)
            tmp_svm_acc.append(svm_correct / k)

        # update mean accuracy
        perceptron_acc.append(np.mean(tmp_perc_acc))
        lda_acc.append(np.mean(tmp_lda_acc))
        svm_acc.append(np.mean(tmp_svm_acc))

    #plot
    plt.grid()
    plt.plot(M, perceptron_acc, c="red")
    plt.plot(M, lda_acc, c="blue")
    plt.plot(M, svm_acc, c="m")
    plt.xlabel("m")
    plt.ylabel("accuracy")
    plt.legend(["perceptron", "LDA", "SVM"])
    plt.title("accuracy as a function of m")
    plt.show()


if __name__ == "__main__":
    Q10()
