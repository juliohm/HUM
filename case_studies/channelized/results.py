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
from pyhum.decomposition import KernelPCA

# prior and posterior log-probabilities
prior      = np.loadtxt("lnprob0001.dat")
posterior  = np.loadtxt("lnprob1000.dat")

# purge outliers
prior      = ma.masked_less(prior, -2000).compressed()
posterior  = ma.masked_less(posterior, -2000).compressed()

fig = plot_lnprob((prior, posterior))
pl.show()
fig.savefig("lnprob.pdf", bbox_inches="tight")

# acceptance fraction for each walker
acceptance = np.loadtxt("acceptance1000.dat")
fig = plot_acceptance(acceptance)
pl.show()
fig.savefig("acceptance.pdf", bbox_inches="tight")

# prior and posterior ensemble
CSI = np.loadtxt("ensemble1000.dat")
nsamples, ncomps = CSI.shape
Xprior = np.loadtxt("ensemble.csv", delimiter=",", skiprows=1, usecols=xrange(nsamples))
kpca = KernelPCA(degree=4)
kpca.train(Xprior, ncomps=ncomps)
Xpost = kpca.predict(CSI.T)

for name, X in {"prior":Xprior, "posterior":Xpost}.items():
    fig = pl.figure()
    for i in xrange(25):
        pl.subplot(5,5,i)
        pl.imshow(X[:,i].reshape(250,250), cmap="PuBu")
        pl.axis("off")
    fig.subplots_adjust(left=0.1, bottom=0.0, right=0.9, top=0.92, wspace=0.2, hspace=0.2)
    pl.suptitle(name+" ensemble")
    pl.show()
    fig.savefig(name+".pdf", bbox_inches="tight")
