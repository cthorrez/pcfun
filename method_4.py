"""
Method on uses scikit-learn LogisticRegression to fit the Bradley Terry model
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

def preprocess(matchups):
    num_models = np.unique(matchups).shape[0]
    unique_matchups, counts = np.unique(matchups, return_counts=True, axis=0)
    num_unique_matchups = unique_matchups.shape[0]

    X = np.zeros(shape=(num_unique_matchups, num_models))
    y = np.ones(shape=(num_unique_matchups))
    weights = counts.astype(np.float32)

    X[np.arange(num_unique_matchups), unique_matchups[:,0]] = 1.0
    X[np.arange(num_unique_matchups), unique_matchups[:,1]] = -1.0

    mask = (np.arange(num_unique_matchups) % 2) == 1
    X[mask] *= -1
    y[mask] = 0.0

    return X, y, weights



def bt_loss_and_grad(ratings, matchups):
    d = ratings.shape[0]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(d)[None,:])
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = expit(rating_diffs)
    loss = -np.log(probs).mean()

    row_level_grad = (1.0 - probs)[:,None]
    row_level_grad = np.repeat(row_level_grad,  axis=1, repeats=2)
    row_level_grad[:,0] *= -1.0
    grad = (row_level_grad[:,:,None] * schedule_mask).sum(axis=(0,1)) / matchups.shape[0]
    return loss, grad

def accuracy(ratings, matchups):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = expit(rating_diffs)
    return np.mean(probs > 0.5)


def method_4(matchups):

    start_time = time.time()
    X, y, weights = preprocess(matchups)
    model = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-12)

    # start_time = time.time()
    model.fit(X=X, y=y, sample_weight=weights)
    duration = time.time() - start_time

    ratings = model.coef_[0]

    acc = accuracy(ratings, matchups)
    loss, _ = bt_loss_and_grad(ratings, matchups)
    print(f'log loss: {loss}')
    print(f'accuracy: {acc}')
    print(f'duration (s): {duration:.4f}')





if __name__ == "__main__":
    matchups = np.load("data/matchups.npz")["matchups"]

    method_4(matchups)