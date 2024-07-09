# pcfun: Fun with Paired Comparisons

a lil repo to play around with models, algorithms, and technologies for paired comparisons
stuff like testing the speed of different implementations of Bradley-Terry and Rao Kupper models

## Terminology
For binary paired comparison models data is presented as `matchups (np.int32, [n,2])` containing winner and loser indices.
For ternary data it's `matchups (np.int32, [n,2])` (order arbitrary) and `outcomes (np.float32, [n,])` taking values 1.0, 0.5, 0.0 representing win, draw, and loss for the first indexed competitor

