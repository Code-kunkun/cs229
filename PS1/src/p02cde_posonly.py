from re import U
from cv2 import add
import numpy as np
from sklearn.metrics import accuracy_score
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    _, t_train = util.load_dataset(train_path, label_col='t')
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    _, t_valid = util.load_dataset(valid_path, label_col='t')
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    _, t_test = util.load_dataset(test_path, label_col='t')
    n = x_test.shape[0]
    LGR_c = LogisticRegression()
    LGR_c.fit(x_train, t_train)
    res = LGR_c.predict(x_test)

    accuracy = np.mean((res > 0.5) == t_test) * 100
    print('accuracy = {:.2f}%'.format(accuracy))
    np.savetxt(pred_path_c, res > 0.5, fmt="%d")


    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    LGR_d = LogisticRegression()
    LGR_d.fit(x_train, y_train)

    res = LGR_d.predict(x_test)

    accuracy = np.mean((res > 0.5) == t_test) * 100
    print('accuracy = {:.2f}%'.format(accuracy))
    np.savetxt(pred_path_d, res > 0.5, fmt="%d")
    

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    x_valid = x_valid[y_valid == 1]
    alpha = np.mean(LGR_d.predict(x_valid))

    res = LGR_d.predict(x_test) / alpha

    accuracy = np.mean((res > 0.5) == t_test) * 100
    print('accuracy = {:.2f}%'.format(accuracy))
    # *** END CODER HERE
