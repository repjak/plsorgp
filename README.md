Introduction
------------

A MATLAB implementation of Probabilistic Least Squares Ordinal Regression [[1]](#srijith).
Two covariance functions are provided, namely *isotropic squared exponential* and
*squared exponential with ARD*.
The implementation at hand is also compatible with GPML [[2]](#gpml) covariances
if [GPML Toolbox](http://www.gaussianprocess.org/gpml/code/matlab/doc/) is in path.

Usage
-----

A simple example on random data:

    % generate training samples
    Xtr = linspace(0, 20, 21)';
    Ytr = [ones(2, 1); 2*ones(3, 1); 3*ones(10,1); 4*ones(5,1); 5];

    % train a PLSOR model with default settings
    ordgp = OrdRegressionGP(Xtr, Ytr);

    % make prediction
    Xte = linspace(0, 20, 50)';
    [Ypred, ~, mu] = ordgp.predict(Xte);

    % plot results
    plot(Xte, mu);          % latent GP mean
    hold on;
    plot(Xtr, Ytr, '+');    % training data
    plot(Xte, Ypred, 'o');  % predicted ordinal labels

For more advanced examples on data from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/),
see script `benchmark.m`.

MEX Files
---------

The following functions for operations on a positive definite matrix are implemented also in C,
using LAPACK and MATLAB C API:

- `CHOLINV` for inverting a positive definite matrix given its Cholesky factor
- `CHOLSOLVE` for solving a system of linear equations AX = B for X with positive definite A
  given by its Cholesky factor

Both functions can be compiled and linked against libraries in a MATLAB path by MEX.
A Makefile is provided for convenience:

    make cholinv cholsolve

MATLAB implementation is provided as well, so if any of the binaries is not present at call time,
the corresponding MATLAB function will be called instead silently.

Dependencies
------------

- MATLAB Optimization Toolbox
- MATLAB Statistics and Machine Learning Toolbox (for running tests)
- [GPML Toolbox](http://www.gaussianprocess.org/gpml/code/matlab/doc/) (optional)

References
----------
<a name="srijith">[1]
Srijith, P. K.; Shevade, S. & Sundararajan, S.
A probabilistic least squares approach to ordinal regression
*Proceedings of the 25th Australasian Joint Conference on Advances in Artificial Intelligence, Springer-Verlag*, **2012**, 683-694
</a>
<a name="gpml">[2]
Rasmussen, C. E. & Nickisch, H.
Gaussian processes for machine learning (GPML) toolbox *J. Mach. Learn. Res., JMLR.org*, **2010**, *11*, 3011-3015
</a>
