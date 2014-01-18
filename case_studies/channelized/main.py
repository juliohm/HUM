## -*- coding: utf8 -*-
## Copyright (c) 2013 Júlio Hoffimann Mendes
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
## Created: 26 Dec 2013
## Author: Júlio Hoffimann Mendes

import sys
import subprocess
import numpy as np
from scipy.stats import multivariate_normal
import emcee
from pyhum.decomposition import KernelPCA
from pyhum.distribution import Nonparametric

# initialize the MPI-based pool
pool = emcee.utils.MPIPool()

def G(m):
    # dump input to file
    basename = "rank%i" % pool.rank
    infile = basename+".dat"
    np.savetxt(infile, m, header="250x250 permeability field")

    # call external simulator
    with open("run%i.log" % pool.rank, "w") as log:
        subprocess.check_call(["./simulator", "-f", infile], stdout=log)

    # load output back
    outfile = basename+".out"
    d = np.loadtxt(outfile, skiprows=1, usecols=xrange(8)) # 8 producer wells

    return d.flatten()

# mtrue is unknown, only used here to generate dobs
mtrue = np.loadtxt("mtrue.dat")
dobs = G(mtrue)

dprior = multivariate_normal(mean=dobs, cov=1)

# initial ensemble from disk (nfeatures x nsamples)
X = np.loadtxt("ensemble.csv", delimiter=",", skiprows=1, usecols=xrange(100))

# ensemble in feature space (ncomps << nfeatures)
ncomps, nsamples = 50, 100
CSI = np.random.randn(ncomps, nsamples)

# connect the two spaces CSI --> X
kpca = KernelPCA(degree=4)
kpca.train(X, ncomps=ncomps)

mprior = Nonparametric(CSI)

# If perfect forwarding is assumed to hold true, the posterior
# density is just the product sigma_m(m) ~ rho_d(G(m)) * rho_m(m).
# We further map m = m(csi) to make this application possible with
# today's computers.
def lnprob(csi):
    m = kpca.predict(csi)
    return mprior.logpdf(csi) + dprior.logpdf(G(m))

# wait for instructions from the master process
if not pool.is_master():
    pool.wait()
    sys.exit()

sampler = emcee.EnsembleSampler(nsamples, ncomps, lnprob, pool=pool, live_dangerously=True)

i=1
for ensemble, logp, state in sampler.sample(CSI.T, iterations=10, storechain=False):
    np.savetxt("ensemble%i.dat" % i, ensemble, header="Ensemble at iteration %i" % i)
    np.savetxt("lnprob%i.dat" % i, logp, header="Log-probability at iteration %i" % i)
    i += 1

pool.close()
