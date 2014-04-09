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
## Created: 10 Feb 2014
## Author: Júlio Hoffimann Mendes

import emcee
import numpy as np
from mpi4py import MPI
from scipy.stats import multivariate_normal
from pyhum.decomposition import KernelPCA
from pyhum.distribution import Nonparametric
from utils import IMEX

# make sure results are reproducible
np.random.seed(2014)

# initialize the MPI-based pool
pool = emcee.utils.MPIPool()

# forward operator d = G(m)
G = IMEX

# tuning parameters
ncomps, nsamples = 50, 100

# initial ensemble from disk (nfeatures x nsamples)
X = np.empty([60048, nsamples])
for j in xrange(nsamples):
    X[:,j] = np.loadtxt("prior/POR_{0:03d}.inc".format(j+1), skiprows=2)

# only 44550 of 60048 cells are active
mask = np.loadtxt("null.inc", dtype=bool, skiprows=2)
X = X[mask,:]

# evaluate forward operator on prior ensemble and save results
D = np.array(pool.map(G, [m for m in X.T])).T
if pool.is_master():
    np.savetxt("Dprior.dat", D)

# tell slaves to proceed
pool.close()

# ensemble in feature space (ncomps << nfeatures)
kpca = KernelPCA()
kpca.train(X, ncomps=ncomps)
CSI = kpca.featurize(X)

mprior = Nonparametric(CSI.T)

# KDE-based proposal
def kde_proposal(CSI):
    return mprior.sample(n_samples=nsamples)

# chosen timesteps for Bayesian inference
timesteps = np.array([37, 59, 71, 83, 101, 107, 111, 113, 119, 122]) - 1 # for 0-indexing
measurements = np.loadtxt("observation.csv", skiprows=2, usecols=xrange(33,53))

# history-based uncertainty mitigation
for i, t in enumerate(timesteps, 1):
    dobs = measurements[t,:]
    dprior = multivariate_normal(mean=dobs, cov=.1)

    # likelihood under perfect forwarding assumption
    def lnlike(csi):
        m = kpca.predict(csi).clip(0, 1)
        return dprior.logpdf(G(m, t))

    # posterior sigma_m(m) ~ rho_d(G(m)) * rho_m(m)
    def lnprob(csi):
        m = kpca.predict(csi).clip(0, 1)
        return mprior.logpdf(csi) + dprior.logpdf(G(m, t))

    if pool.is_master():
        ### There are two possible configurations:

        # a) (symmetric) stretch move
        #sampler = emcee.EnsembleSampler(nsamples, ncomps, lnprob, pool=pool, live_dangerously=True)
        #mcmc = sampler.sample(CSI.T, iterations=100, storechain=False)

        # b) KDE move
        sampler = emcee.EnsembleSampler(nsamples, ncomps, lnlike, pool=pool, live_dangerously=True)
        mcmc = sampler.sample(CSI.T, iterations=100, storechain=False, mh_proposal=kde_proposal)

        for j, (ensemble, logp, state) in enumerate(mcmc, 1):
            np.savetxt("ensemble{0:03d}-{1:03d}.dat".format(i,j), ensemble)
            np.savetxt("lnprob{0:03d}-{1:03d}.dat".format(i,j), logp)
            np.savetxt("acceptance{0:03d}-{1:03d}.dat".format(i,j), sampler.acceptance_fraction)

        # update prior with posterior
        mprior = Nonparametric(ensemble)
        CSI = np.array(ensemble).T

        # we're done with this timestep, tell slaves to proceed
        pool.close()
    else:
        # create dummy variable and wait for instructions
        ensemble = None
        pool.wait()

# broadcast last ensemble
CSI = MPI.COMM_WORLD.bcast(ensemble)

# G* = (G o m)(csi)
def G_star(csi):
    m = kpca.predict(csi)
    return G(m)

# evaluate forward operator on posterior ensemble and save results
D = np.array(pool.map(G_star, [csi for csi in CSI])).T
if pool.is_master():
    np.savetxt("Dpost.dat", D)

pool.close()
