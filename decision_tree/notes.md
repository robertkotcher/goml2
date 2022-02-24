-- References

https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/trees.pdf

https://www.youtube.com/watch?v=D0efHEJsfHo&t=776s

http://mlwiki.org/index.php/Cost-Complexity_Pruning

-- k-fold cross validation

    Returns a model trained on _all_ of the training data

    pass all data, and iteratively build trees with k-1/k of the
    training data. Then for each value of k, determine the average
    Tree Score across all data splits.

-- pruning

    bottom-up
    * reduced-error pruning
    * minimum cost complexity pruning

        Tree Score = SSR + alpha * |T|  (num leaves)
            (higher the alpha, higher the penalty for having leaves)

        Note this is also dependent on train/test data split, which we
        average using k-fold cross validation.

    * minimum error pruning

    top-down
    * pessimistic error pruning