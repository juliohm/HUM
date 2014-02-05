## -*- coding: utf8 -*-
## Copyright (c) 2014 Júlio Hoffimann Mendes
##
## This file is part of HUM.
##
## HUM is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## HUM is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with HUM.  If not, see <http://www.gnu.org/licenses/>.
##
## Created: 09 Jan 2014
## Author: Júlio Hoffimann Mendes

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

class Nonparametric(object):
    """
    Wrapper class over scikit-learn KDE with built-in cross-validation

    Parameters
    ----------
    X: ndarray or matrix
        Data matrix (nsamples x nfeatures)

    References
    ----------
      PARZEN, E., 1962. On Estimation of a Probability Density Function and Mode.

      TERREL, G. R.; SCOTT D. W., 1992. Variable Kernel Density Estimation.
    """
    def __init__(self, X):
        self.kde = self._best_estimator(X)


    def pdf(self, x):
        return np.exp(self.logpdf(x))


    def logpdf(self, x):
        return self.kde.score_samples(x)


    def sample(self, *args, **kwargs):
        return self.kde.sample(*args, **kwargs)


    def _best_estimator(self, X):
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth' : np.linspace(0.1, 1, 30)},
                            cv=10) # 10-fold cross-validation
        grid.fit(X)
        return grid.best_estimator_
