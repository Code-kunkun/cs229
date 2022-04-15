import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    res = clf.predict(x_eval)
    
    accuracy = np.sum((res > 0.5) == y_eval)
    print("accuracy = {}%".format(accuracy))
    np.savetxt(pred_path, res > 0.5, fmt="%d")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***

        m, n = x.shape
        self.theta = np.zeros(n)
        
        # Newton's method
        while True:
            theta_old = np.copy(self.theta)

            h_x = self.sigmoid(x.dot(self.theta))
            H = (x.T * h_x * (1 - h_x)).dot(x) / m 
            gradient_J_theta = x.T.dot(h_x - y) / m 

            # update theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta) # np.linalg.inv: 矩阵求逆

            # End training
            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps: # 以第一范式为判定原则
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(x.dot(self.theta))
        # *** END CODE HERE ***
