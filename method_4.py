"""
Method on uses scikit-learn LogisticRegression to fit the Bradley Terry model
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit



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


def method_1(matchups):
    start_time = time.time()
    num_models = np.unique(matchups).shape[0]
    ratings = np.zeros(num_models)
    ratings = minimize(
        fun=bt_loss_and_grad,
        x0=ratings,
        args = (matchups,),
        method='L-BFGS-B',
        jac=True,
        options={'disp' : True}
    )['x']
    duration = time.time() - start_time
    print(f"accuracy: {accuracy(ratings, matchups)}")
    print(f"duration (s) {duration:.4f}")



if __name__ == "__main__":
    matchups = np.load("data/matchups.npz")["matchups"]
    method_1(matchups)