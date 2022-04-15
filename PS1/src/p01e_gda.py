import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel
from p01b_logreg import LogisticRegression

def plot(x, y, theta_1, legend_1=None, theta_2=None, legend_2=None, title=None, correction=1.0):
    # plot dataset 
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # plot decision boundary (found by solving for theta_1^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta_1[0] / theta_1[2] * correction + theta_1[1] / theta_1[2] * x1)
    plt.plot(x1, x2, c='red', label=legend_1, linewidth=2)

    # plot decision boundary (found by solving for theta_2^T x = 0)
    if theta_2 is not None:
        x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
        x2 = -(theta_2[0] / theta_2[2] * correction + theta_2[1] / theta_2[2] * x1)
        plt.plot(x1, x2, c='black', label=legend_2, linewidth=2)
    
    # add labels, legend and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    if legend_1 is not None or legend_2 is not None:
        plt.legend(loc='upper left')
    if title is not None:
        plt.suptitle(title, fontsize=12)


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    # *** START CODE HERE ***

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    gda = GDA()
    gda.fit(x_train, y_train)
    res = gda.predict(x_eval)

    accuarcy = np.sum((res > 0.5) == y_eval)

    print('accuracy = {}%'.format(accuarcy))
    util.plot(x_train, y_train, gda.all_theta, 'output/p01e_{}.png'.format(pred_path[-5]))

    plot(x_eval, y_eval, theta_1=log_reg.theta, legend_1='logistic regression', theta_2=gda.all_theta, legend_2='GDA', title='Validation Set')
    plt.savefig('output/constract_{}.png'.format(pred_path[-5]))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.p = sum(y == 1) / m
        u0 = np.zeros(n)
        cnt = 0
        for i in range(m):
            u0 += x[i] if y[i] == 0 else 0
            cnt += 1 if y[i] == 0 else 0 
        self.u0 = u0 / cnt 
        cnt = 0
        u1 = np.zeros(n)
        for i in range(m):
            u1 += x[i] if y[i] == 1 else 0
            cnt += 1 if y[i] == 1 else 0
        self.u1 = u1 / cnt
        t = np.copy(x)
        for i in range(m):
            t[i] -= self.u0 if y[i] == 0 else self.u1 
        self.matrix_variance = t.T.dot(t) / m 
        assert self.matrix_variance.shape == (n, n)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        ni_matrix = np.linalg.inv(self.matrix_variance)
        theta = ni_matrix.dot(self.u1 - self.u0)
        assert theta.shape == (x.shape[1], )
        theta_0 = 1 / 2 * (self.u0 + self.u1).dot(ni_matrix).dot(self.u0 - self.u1) - np.log((1 - self.p) / self.p)
        self.all_theta = np.append(theta_0, theta)
        return 1 / (1 + np.exp(-(x.dot(theta) + theta_0)))
        # *** END CODE HERE
