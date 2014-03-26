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
## Created: 22 Mar 2014
## Author: Júlio Hoffimann Mendes

import numpy as np
import pylab as pl
from pyhum.decomposition import KernelPCA

# make sure results are reproducible
np.random.seed(2014)

# load ensemble from disk (nfeatures x nsamples)
X = np.loadtxt("ensemble.csv", delimiter=",", skiprows=1, usecols=xrange(100))

# parametrization
kpca = KernelPCA(degree=4)
ncomps = kpca.train(X, ncomps=50)

# featurize and reconstruct
CSI  = kpca.featurize(X)
Xnew = kpca.predict(CSI)
Xnew = kpca.denoise(Xnew)

fig = pl.figure()

for i in xrange(10):
    pl.subplot(5,4,2*i+1)
    pl.imshow(X[:,i].reshape(250,250), cmap="PuBu")
    if i < 2: pl.title("original", fontdict={"fontsize":10})
    pl.axis("off")

    pl.subplot(5,4,2*(i+1))
    pl.imshow(Xnew[:,i].reshape(250,250), cmap="PuBu")
    if i < 2: pl.title("reconstruction", fontdict={"fontsize":10})
    pl.axis("off")

fig.subplots_adjust(left=0.1, bottom=0.0, right=0.9, top=0.9, wspace=0.0, hspace=0.2)
fig.suptitle("Kernel PCA reconstruction")
pl.show()
fig.savefig("kpca_reconstruction.pdf", bbox_inches="tight")
