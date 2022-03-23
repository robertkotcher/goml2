# TODO

- [x] implement max depth in decision trees

# gradient boosting for regression

We have a series of h's that together make a good prediction. Assume we already have a function F that is better than random. Then we want to find F(x1) + h(x1) = y1, where y1 is the actual target.

h(x1) = y1 - F(x1)
h(x2) = y2 - F(x2)

Fit a regression tree to data:

(x1, y1 - F(x1)), (x2, y2 - F(x2)), ..

## how is this related to gradient descent?

dJ/dF(xi) = ... = F(xi) - yi - dJ/dF

^^ i.e. derivative of loss function wrt the predicted value:
   i.e. derivative of .5(observed - predicted)^2

residual <=> negative gradient
fit h to residual <=> fit h to negative gradient

**** called gradient boosting because:

at each step, we're finding a gamma that minimizes the SSR. At each leaf, we're minimizing a loss function. because this is a minimization problem, we take derivative and set to 0 to find the minimum. we could use gradient descent here but since math is pretty easy, we just solve. it ends up being just the average at each leaf for this loss function.

# statquest

## intuition

first learner predicts the average label, i.e., it's a tree with just a root leaf.

next trees predict the _residual_. By summing the average with the output of this tree (scaled by some alpha), we get predictions that are a bit closer to the actual target label.
