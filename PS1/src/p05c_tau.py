from re import L
import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    model = LocallyWeightedLinearRegression(tau=0.5)
    model.fit(x_train, y_train)

    mse_list = []
    for tau in tau_values:
        model.tau = tau 
        y_pred = model.predict(x_valid)

        mse = np.mean((y_pred - y_valid) ** 2)
        mse_list.append(mse)
        print("Valid set: tau = {}, mse = {}".format(tau, mse))

        plt.figure()
        plt.title('tau = {}'.format(tau))
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_valid, y_pred, 'ro', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('output/p05c_tau_{}.png'.format(tau))

    tau_opt = tau_values[np.argmin(mse_list)]
    print(f"valid set: lowest MSE = {min(mse_list)}, tau = {tau_opt}")
    model.tau = tau_opt 

    y_pred = model.predict(x_test)
    np.savetxt(pred_path, y_pred)
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
