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
## Created: 16 Feb 2014
## Author: Júlio Hoffimann Mendes

import numpy as np
import pylab as pl
import numpy.ma as ma
from pyhum.plotting import *

# load results
prior      = np.loadtxt("lnprob0001.dat")
posterior  = np.loadtxt("lnprob1000.dat")
acceptance = np.loadtxt("acceptance1000.dat")

# purge outliers
prior      = ma.masked_less(prior, -2000).compressed()
posterior  = ma.masked_less(posterior, -2000).compressed()

# prior vs. posterior log-probabilities
fig = plot_lnprob((prior, posterior))
pl.show()
fig.savefig("lnprob.pdf")

# acceptance fraction for each walker
fig = plot_acceptance(acceptance)
pl.show()
fig.savefig("acceptance.pdf")
