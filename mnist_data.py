from models import *
from ex3_utils import *


def Q12():
    """
    draw 3 images of label '0', and 3 images of label '1'
    """
    # choose only relevant images
    img_0 = x_train[np.where(y_train == 0)[0]]
    img_1 = x_train[np.where(y_train == 1)[0]]

    # plot images
    # -- label 0
    for i in range(3):
        plt.imshow(img_0[i], cmap="gray")
        plt.show()

    # -- label 1
    for i in range(3):
        plt.imshow(img_1[i], cmap="gray")
        plt.show()


def rearrange_data(X: np.ndarray):
    """
    :param X: array to flatten of shape (m, 28, 28)
    :return: array of shape (m, 28^2)
    """
    assert len(X.shape) == 3
    # flatten dims 1-2
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def Q14():
    """
    ans to Question 14 as described in Ex. sheet
    """
    estimate_models(models=[Logistics().model, SVM(C=10).model, DesicionTree().model, Neighbours(5).model],
                    M=[50, 100, 300, 500], repeat=50,
                    legend=["Logistics", "Soft-SVM", "DecisionTree", "5-Nearest neighbours"], Q10=False)
