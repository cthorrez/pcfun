"""
Method on uses jax for auto diff and newton's method for optimization on the deduplicated dataset
"""
import time
import numpy as np
import jax
from jax import jit, value_and_grad, hessian
from jax.scipy.linalg import solve
import jax.numpy as jnp

def preprocess(matchups):
    unq_matchups, counts = jnp.unique(matchups, return_counts=True, axis=0)
    weights = counts.astype(jnp.float32) / matchups.shape[0]
    return unq_matchups, weights


@jit
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-1.0 * x))

@jit
def bt_loss(ratings, unq_matchups, weights):
    matchup_ratings = ratings[unq_matchups]
    rating_diffs = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = sigmoid(rating_diffs)
    return -jnp.sum(weights * jnp.log(probs))

def accuracy(ratings, matchups):
    matchup_ratings = ratings[matchups]
    rating_diffs = matchup_ratings[:,0] - matchup_ratings[:,1]
    probs = sigmoid(rating_diffs)
    return (probs > 0.5).mean()  


def method_5(matchups):
    num_models = np.unique(matchups).shape[0]

    unq_matchups, weights = preprocess(matchups)

    ratings = jnp.zeros(num_models)
    loss_and_grad_fn = value_and_grad(bt_loss)
    hess_fn = hessian(bt_loss)

    tol = 1e-12

    done = False
    prev_loss = jnp.inf
    i = 0

    start_time = time.time()
    while not done:
        loss, gradient = loss_and_grad_fn(ratings, unq_matchups, weights)
        print(f"iter {i+1} loss: {loss}")
        hess = hess_fn(ratings, unq_matchups, weights)
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
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)
    matchups = jnp.array(matchups)
    method_5(matchups)
