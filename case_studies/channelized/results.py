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

import logging
import numpy as np
import pylab as pl
import numpy.ma as ma
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

prior      = np.loadtxt("lnprob0001.dat")
posterior  = np.loadtxt("lnprob1000.dat")

fig = plot_lnprob((prior, posterior))
pl.show()
fig.savefig("lnprob.pdf", bbox_inches="tight")

#-----------------------------------------------------------

logger.info("Plotting acceptance fraction for each walker...")

acceptance = np.loadtxt("acceptance1000.dat")
fig = plot_acceptance(acceptance)
pl.show()
fig.savefig("acceptance.pdf", bbox_inches="tight")

#-----------------------------------------------------------

logger.info("Plotting prior and posterior ensemble... (be patient)")

CSI = np.loadtxt("ensemble1000.dat")
nsamples, ncomps = CSI.shape
Xprior = np.loadtxt("ensemble.csv", delimiter=",", skiprows=1, usecols=xrange(nsamples))
kpca = KernelPCA(degree=4)
kpca.train(Xprior, ncomps=ncomps)
Xpost = kpca.predict(CSI.T)
Xpost = kpca.denoise(Xpost)
idx = np.argsort(posterior)[::-1]
for name, X in [("prior",Xprior),("posterior",Xpost)]:
    fig = pl.figure()
    for i in xrange(25):
        pl.subplot(5,5,i+1)
        pl.imshow(X[:,idx[i]].reshape(250,250), cmap="PuBu")
        pl.axis("off")
    fig.subplots_adjust(left=0.1, bottom=0.0, right=0.9, top=0.92, wspace=0.2, hspace=0.2)
    fig.suptitle(name+" ensemble")
    pl.show()
    fig.savefig(name+".pdf", bbox_inches="tight")

#-----------------------------------------------------------

logger.info("Plotting history for prior and posterior ensemble...")

nsteps, nwells = 20, 8
Dprior = np.loadtxt("Dprior.dat")
Dpost  = np.loadtxt("Dpost.dat")
dobs   = np.loadtxt("dobs.dat").reshape(nsteps, nwells)
dmap   = Dpost[:,idx[0]].reshape(nsteps, nwells)
for name, D in [("prior",Dprior),("posterior",Dpost)]:
    fig = pl.figure()
    for w in xrange(nwells):
        pl.subplot(2,4,w+1)
        for i in xrange(nsamples):
            d = D[:,i].reshape(nsteps, nwells)
            pl.plot(d[:,w], color="gray", linewidth=0.1)
        pl.plot(dobs[:,w], color="red", linewidth=1, label="well %i" % (w+1))
        if name == "posterior":
            pl.plot(dmap[:,w], color="yellow", linewidth=1, label="MAP")
        pl.gca().set_ylim(235, 265)
        pl.legend(loc="upper right", fontsize=8)
    fig.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.92, wspace=0.24, hspace=0.2)
    fig.suptitle("history for "+name+" ensemble")
    fig.text(0.5, 0.02, "timestep", ha="center", va="center")
    fig.text(0.015, 0.5, u"production rate [m³/d]", ha="center", va="center", rotation="vertical")
    pl.show()
    fig.savefig("history_"+name+".pdf", bbox_inches="tight")

#-----------------------------------------------------------

logger.info("Plotting maximum a posteriori estimate...")

mtrue = np.loadtxt("mtrue.dat", skiprows=22).reshape(250,250)
mmap  = Xpost[:,idx[0]].reshape(250,250)

fig = pl.figure()
pl.subplot(121)
pl.imshow(mtrue, cmap="PuBu")
pl.axis("off")
pl.title("true reservoir")
pl.subplot(122)
pl.imshow(mmap, cmap="PuBu")
pl.axis("off")
pl.title("MAP estimate")
pl.show()
fig.savefig("MAP.pdf", bbox_inches="tight")

#-----------------------------------------------------------

try:
    import pandas as pd
    from skimage.measure import structural_similarity

    logger.info("Computing structural similarity statistics...")

    ssim = np.empty(25)
    for name, X in [("prior",Xprior),("posterior",Xpost)]:
        for i in xrange(25):
            ssim[i] = structural_similarity(mtrue, X[:,idx[i]].reshape(250,250), win_size=7)
        ssim = pd.Series(ssim)
        print "==> " + name + " SSIM statistics:", ssim.describe().to_string()
        print "interquartile range:", ssim.quantile(0.75) - ssim.quantile(0.25)
    print "SSIM index for MAP estimate:", structural_similarity(mtrue, mmap, win_size=7)
except ImportError:
    print "Consider installing scikit-image and pandas for SSIM statistics."
