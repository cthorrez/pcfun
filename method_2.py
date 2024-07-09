"""
Method on uses jax for auto diff and newton's method for optimization
"""
import numpy as np
from jax import jit, grad, jacfwd, jacrev
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
    grad_fn = grad(bt_loss)
    hess_fn = jacfwd(grad_fn)

    num_iter = 200
    tol = 1e-6

    prev_loss = jnp.inf
    for i in range(num_iter):
        loss = bt_loss(ratings, matchups)
        print(f"iter {i} loss: {loss}")
        gradient = grad_fn(ratings, matchups)
        hessian = hess_fn(ratings, matchups)
        update = jnp.linalg.inv(hessian).dot(gradient)
        ratings = ratings - update
        if jnp.abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    acc = accuracy(ratings, matchups)
    print(f"accuracy: {acc}")

    

if __name__ == "__main__":
    matchups = np.load("data/matchups.npz")["matchups"]

    matchups - jnp.array(matchups)
    method_2(matchups)
