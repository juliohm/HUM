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

prior     = np.loadtxt("lnprob001-001.dat")
posterior = np.loadtxt("lnprob003-010.dat")
fig = plot_lnprob((prior, posterior))
pl.show(); fig.savefig("lnprob.pdf", bbox_inches="tight")

#-----------------------------------------------------------

logger.info("Plotting acceptance fraction for each walker...")

acceptance = np.loadtxt("acceptance003-010.dat")
fig = plot_acceptance(acceptance)
pl.show(); fig.savefig("acceptance.pdf", bbox_inches="tight")

#-----------------------------------------------------------

logger.info("Plotting history for prior and posterior ensemble...")

idx = np.argsort(posterior)[::-1]

# list of wells to plot (up to 8 wells)
wells  = np.arange(1, 9) - 1 # BR-P-1 = 0, BR-P-2 = 1, ...

from utils import history
CSI = np.loadtxt("ensemble003-010.dat")
nsteps, nwells = history.shape
nsamples, ncomps = CSI.shape
Dprior = np.loadtxt("Dprior.dat")
Dpost  = np.loadtxt("Dpost.dat")
dobs   = history
dmap   = Dpost[:,idx[0]].reshape(nsteps, nwells)
for name, D in [("prior",Dprior),("posterior",Dpost)]:
    fig = pl.figure()
    for plotid, w in enumerate(wells, 1):
        pl.subplot(2,4,plotid)
        for i in xrange(nsamples):
            d = D[:,i].reshape(nsteps, nwells)
            pl.plot(d[:,w], color="gray", linewidth=0.1)
        pl.plot(dobs[:,w], color="red", linewidth=1, label="BR-P-%i" % (w+1))
        if name == "posterior":
            pl.plot(dmap[:,w], color="yellow", linewidth=1, label="MAP")
        pl.gca().set_xlim(0, nsteps)
        pl.legend(loc="upper right", fontsize=8)
    fig.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.92, wspace=0.24, hspace=0.2)
    fig.suptitle("history for "+name+" ensemble")
    fig.text(0.5, 0.02, "timestep", ha="center", va="center")
    fig.text(0.015, 0.5, u"production rate [m³/d]", ha="center", va="center", rotation="vertical")
    pl.show(); fig.savefig("history_"+name+".pdf", bbox_inches="tight")
