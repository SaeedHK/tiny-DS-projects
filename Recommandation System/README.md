### Matrix Factorization

This is an autonomous project on doing a Matrix Factorization of a evaluation matrix for doing a recommendation system. The story is as follows:

Suppose there are `m` users evaluating `n` items. The matrix of evluation `M` is hence is in form `m*n`. The matrix has
missing values and our goal os to predict the missing values in a resonable way. One idea is to write `M = UV` where U and V are `m*k` and `k*n` matrices. This is not always possible to find these matrices but our goes is find `U,V` such that minimises the L2 norm of the differences. This is the code I developped in [file](https://github.com/saeedhadikhanloo/MyProjectsCodes/blob/master/Recommandation%20System/Rec1.ipynb).
