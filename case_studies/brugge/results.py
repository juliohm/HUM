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
## Created: 16 Mar 2014
## Author: Júlio Hoffimann Mendes

import logging
import numpy as np
import pylab as pl
from pyhum.plotting import *
from pyhum.decomposition import KernelPCA

# make sure results are reproducible
np.random.seed(2014)

# logging settings
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(asctime)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger()

#-----------------------------------------------------------

logger.info("Plotting prior and posterior log-probabilities...")

prior      = np.loadtxt("lnprob001-001.dat")
posterior  = np.loadtxt("lnprob010-100.dat")

fig = plot_lnprob((prior, posterior))
pl.show()
fig.savefig("lnprob.pdf", bbox_inches="tight")

#-----------------------------------------------------------

logger.info("Plotting acceptance fraction for each walker...")

acceptance = np.loadtxt("acceptance010-100.dat")
fig = plot_acceptance(acceptance)
pl.show()
fig.savefig("acceptance.pdf", bbox_inches="tight")
