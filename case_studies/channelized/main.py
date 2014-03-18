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
import emcee
import numpy as np
from scipy.stats import multivariate_normal
from pyhum.decomposition import KernelPCA
from pyhum.distribution import Nonparametric
from utils import filtersim, OPMSimulator

# make sure results are reproducible
np.random.seed(2014)

# initialize the MPI-based pool
pool = emcee.utils.MPIPool()

# forward operator d = G(m)
G = OPMSimulator

# mtrue is unknown, only used here to generate dobs
mtrue = np.loadtxt("mtrue.dat", skiprows=22)
dobs = G(mtrue)

dprior = multivariate_normal(mean=dobs, cov=1)

# save synthetic observations into disk
if pool.is_master():
    np.savetxt("dobs.dat", dobs)

# tuning parameters
ncomps, nsamples = 50, 100

# initial ensemble from disk (nfeatures x nsamples)
X = np.loadtxt("ensemble.csv", delimiter=",", skiprows=1, usecols=xrange(nsamples))

# evaluate forward operator on prior ensemble and save results
D = np.array(pool.map(G, [m for m in X.T])).T
if pool.is_master():
    np.savetxt("Dprior.dat", D)

# tell slaves to proceed
pool.close()

# ensemble in feature space (ncomps << nfeatures)
kpca = KernelPCA(degree=4)
kpca.train(X, ncomps=ncomps)
CSI = kpca.featurize(X)

mprior = Nonparametric(CSI.T)

# If perfect forwarding is assumed to hold true, the posterior
# density is just the product sigma_m(m) ~ rho_d(G(m)) * rho_m(m).
# We further map m = m(csi) to make this application possible with
# today's computers.
def lnprob(csi):
    m = kpca.predict(csi)
    return mprior.logpdf(csi) + dprior.logpdf(G(m))

# likelihood
def lnlike(csi):
    m = kpca.predict(csi)
    return dprior.logpdf(G(m))

# KDE-based proposal
def kde_proposal(CSI):
    return mprior.sample(n_samples=nsamples)

# Filtersim-based proposal
def filtersim_proposal(CSI):
    X = filtersim(nsamples)
    return kpca.featurize(X).T

# wait for instructions from the master process
if not pool.is_master():
    pool.wait()
    sys.exit()

### There are three possible configurations:

# a) (symmetric) stretch move
#sampler = emcee.EnsembleSampler(nsamples, ncomps, lnprob, pool=pool, live_dangerously=True)
#mcmc = sampler.sample(CSI.T, iterations=1000, storechain=False)

# b) KDE move
sampler = emcee.EnsembleSampler(nsamples, ncomps, lnlike, pool=pool, live_dangerously=True)
mcmc = sampler.sample(CSI.T, iterations=1000, storechain=False, mh_proposal=kde_proposal)

# c) Filtersim move
#sampler = emcee.EnsembleSampler(nsamples, ncomps, lnlike, pool=pool, live_dangerously=True)
#mcmc = sampler.sample(CSI.T, iterations=1000, storechain=False, mh_proposal=filtersim_proposal)

for i, (ensemble, logp, state) in enumerate(mcmc, 1):
    np.savetxt("ensemble{0:04d}.dat".format(i), ensemble)
    np.savetxt("lnprob{0:04d}.dat".format(i), logp)
    np.savetxt("acceptance{0:04d}.dat".format(i), sampler.acceptance_fraction)

# G* = (G o m)(csi)
def G_star(csi):
    m = kpca.predict(csi)
    return G(m)

# evaluate forward operator on posterior ensemble and save results
D = np.array(pool.map(G_star, [csi for csi in ensemble])).T
if pool.is_master():
    np.savetxt("Dpost.dat", D)

pool.close()
