"""
Method on uses torch for auto diff and newton's method for optimization
"""
import time
import numpy as np
import torch
from torch.func import hessian, grad_and_value
from torch.nn.functional import sigmoid


# @torch.jit.script
def bt_loss(ratings, matchups):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = sigmoid(rating_diffs)
    return -torch.log(probs).mean()

def accuracy(ratings, matchups):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = sigmoid(rating_diffs)
    return (probs > 0.5).to(torch.float32).mean()  


def method_3(matchups, device="cpu"):
    num_models = torch.unique(matchups).shape[0]
    ratings = torch.zeros(num_models, dtype=torch.float32, device=device)
    loss_and_grad_fn = grad_and_value(bt_loss)
    hess_fn = hessian(bt_loss)

    tol = 1e-16

    done = False
    prev_loss = torch.inf
    i = 0

    start_time = time.time()
    while not done:
        gradient, loss = loss_and_grad_fn(ratings, matchups)
        print(f"iter {i+1} loss: {loss}")
        hess = hess_fn(ratings, matchups)
        update = torch.linalg.solve(hess, gradient)
        ratings = ratings - update
        if torch.abs(prev_loss - loss) <= tol:
            done = True
        prev_loss = loss
        i += 1
    duration = time.time() - start_time
    acc = accuracy(ratings, matchups)
    print(f"accuracy: {acc}")
    print(f"duration (s): {duration:.4f}")



if __name__ == "__main__":
    matchups = np.load("data/matchups.npz")["matchups"]
    device="cuda:0"
    device="cpu"


    matchups = torch.tensor(matchups, dtype=torch.int32, device=device)
    method_3(matchups, device=device)
