"""
Method on uses vectorized numpy for loss and grad, scipy L-BFGS for opt
"""

import numpy as np
from scipy.optimize import minimize

def bt_loss_and_grad(ratings, matchups):
    pass


def method_1(matchups):
    num_models = np.unique(matchups).shape[0]

    pi = np.zeros()



if __name__ == "__main__":
    matchups = np.load("data/matchups.npz")["matchups"]

    print(matchups.shape)
