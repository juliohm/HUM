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
## Created: 08 Jan 2014
## Author: Júlio Hoffimann Mendes

import numpy as np
import pylab as pl
from pyhum.decomposition import KernelPCA

# load ensemble from disk (nfeatures x nsamples)
X = np.loadtxt("ensemble.csv", delimiter=",", skiprows=1, usecols=xrange(100))

fig = pl.figure()

for d in xrange(1,5):
    kpca = KernelPCA(degree=d)
    ncomps = kpca.train(X, ncomps=50)
    img = kpca.predict(np.ones(ncomps))
    denoised = kpca.denoise(img)

    pl.subplot(2,4,d)
    pl.title("d = %i" % d)
    pl.imshow(img.reshape(250,250), cmap="PuBu")
    pl.axis("off")

    pl.subplot(2,4,d+4)
    pl.imshow(denoised.reshape(250,250), cmap="PuBu")
    pl.axis("off")

fig.tight_layout(h_pad=0.5, w_pad=0.5)
bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="white", ec="b", lw=2)
fig.text(0.15, 0.5, (
         "Kernel PCA for increasing degrees: "
         "reconstruction above and denoised version below"),
         bbox=bbox_props)

pl.show()
fig.savefig("kpca_validation.pdf", bbox_inches="tight")
