import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from mnist_data import rearrange_data
from comparison import draw_points

# ------- mnist preset given in Ex. sheet -------
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]
# ------
# ------ other constants ------
COLORS = ["red", "blue", "m", "green", "black"]


def gen_hyperplane_func(hypothesis='true'):
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


def draw_samples(m, x: np.ndarray, y: np.ndarray):
    """
    draw m images uniformly from the test set
    :param m: num of samples to draw
    :param x: train set
    :param y: test set
    :return: formatted_samples (m, 28^2), tags = their tags
    """
    # uniformly choose m indexes
    idx = np.random.choice(np.arange(x.shape[0]), size=m, replace=False)

    # format samples and tags
    formatted_samples = rearrange_data(x[idx])
    tags = y[idx]

    return formatted_samples.T, tags


def estimate_models(models: list, M: list, repeat: int, legend: list, Q10=True):
    """
    holds model estimation functionality for Q10, Q14
    :param models: models to use in estimation
    :param M: list of number of wanted training samples in iteration
    :param repeat: how many inner loop repetitions
    :param legend: legend for plot
    :param Q10: boolean flag, indicates Q10 = True, Q14 = False
    :return: None
    """
    acc = []
    rtime = []
    for i in range(len(models)):
        acc.append([])
        rtime.append([])
    for m in M:
        # init tmp lists
        tmp_acc = []
        tmp_rtime = []
        for i in range(len(models)):
            tmp_acc.append([])
            tmp_rtime.append([])

        for i in range(repeat):
            # draw train, test sets
            if Q10:
                # Note: Implementation of draw_points enforces existence of {-1,1} tags
                train_X, train_y = draw_points(m, pad=True)
                test_X, test_y = draw_points(1000, pad=True)
            else:
                train_X, train_y = draw_samples(m, x_train, y_train)
                test_X, test_y = draw_samples(m, x_test, y_test)  # evaluate over the whole test set

            for j, model in enumerate(models):
                t = time.time()
                # fit
                model.fit(train_X, train_y)

                # predict
                model_predictions = model.predict(test_X).flatten()

                # estimate accuracy and runtime
                model_correct = np.where(test_y - model_predictions == 0)[0].shape[0]
                tmp_acc[j].append(model_correct / test_y.shape[0])
                tmp_rtime[j].append(time.time()-t)

        # update mean accuracy
        for j in range(len(tmp_acc)):
            acc[j].append(np.average(tmp_acc[j]))
            rtime[j].append(np.average(tmp_rtime[j]))

    # plot
    if Q10:
        plt.grid()
        for i in range(len(acc)):
            plt.plot(M, acc[i], c=COLORS[i])
        plt.xlabel("m")
        plt.ylabel("accuracy")
        plt.legend(legend)
        plt.title("accuracy as a function of m")
        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].grid()
        axs[1].grid()
        for i in range(len(acc)):
            axs[0].plot(M, acc[i], color=COLORS[i])
            axs[1].plot(M, rtime[i], color=COLORS[i])
        axs[0].set(xlabel="m", ylabel="accuracy")
        axs[1].set(xlabel="m", ylabel="run_time [sec]")
        axs[0].legend(legend)
        axs[1].legend(legend)
        axs[0].set_title("accuracy as a function of m")
        axs[1].set_title("runtime as a function of m")
        plt.show()
