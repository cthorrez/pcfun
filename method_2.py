"""
Method on uses jax for auto diff and newton's method for optimization
"""
import time
import numpy as np
from jax import jit, value_and_grad, hessian
from jax.scipy.linalg import solve
import jax.numpy as jnp

@jit
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-1.0 * x))

@jit
def bt_loss(ratings, matchups):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = sigmoid(rating_diffs)
    return -jnp.log(probs).mean()

def accuracy(ratings, matchups):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = sigmoid(rating_diffs)
    return (probs > 0.5).mean()  


def method_2(matchups):
    num_models = np.unique(matchups).shape[0]

    ratings = jnp.zeros(num_models)
    loss_and_grad_fn = value_and_grad(bt_loss)
    hess_fn = hessian(bt_loss)

    tol = 1e-8

    done = False
    prev_loss = jnp.inf
    i = 0

    start_time = time.time()
    while not done:
        loss, gradient = loss_and_grad_fn(ratings, matchups)
        print(f"iter {i+1} loss: {loss}")
        hess = hess_fn(ratings, matchups)
        update = solve(hess, gradient, assume_a="sym")
        # update = np_solve(hess, gradient
        # update = jnp.linalg.inv(hess).dot(gradient)
        ratings = ratings - update
        if jnp.abs(prev_loss - loss) <= tol:
            done = True
        prev_loss = loss
        i += 1
    duration = time.time() - start_time
    acc = accuracy(ratings, matchups)
    print(f"accuracy: {acc}")
    print(f"duration (s): {duration:.4f}")


    

if __name__ == "__main__":
    matchups = np.load("data/matchups.npz")["matchups"]

    matchups - jnp.array(matchups)
    method_2(matchups)
