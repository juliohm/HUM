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
from scipy.stats import gaussian_kde

# machine epsilon
eps = np.finfo(float).eps

class Nonparametric(gaussian_kde):
    """
    Wrapper class over SciPy Gaussian KDE
    """
    def pdf(self, x):
        return self.evaluate(x)

    def logpdf(self, x):
        p = self.evaluate(x)
        return np.log(p) if p > eps else -np.inf
