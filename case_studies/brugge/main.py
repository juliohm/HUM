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

import os
import subprocess
from string import Template
import numpy as np
from scipy.stats import multivariate_normal
import emcee
from pyhum.decomposition import KernelPCA
from pyhum.distribution import Nonparametric
from utils import CMGFile

# make sure results are reproducible
np.random.seed(2014)

# initialize the MPI-based pool
pool = emcee.utils.MPIPool()

# forward operator as a function of timestep
def G(m, timestep):
    basename = "rank%i" % pool.rank
    cmgfile = CMGFile(basename)

    # active cells in the grid
    mask = np.loadtxt("null.inc", dtype=bool, skiprows=2)

    # poro-perm regression
    phi = np.zeros_like(mask, dtype=float)
    phi[mask] = m
    Kx = 0.01*np.exp(45.633*phi)
    Kz = 0.4*Kx

    # dump input to file
    np.savetxt(cmgfile.poro, phi, comments="", header="POR ALL")
    np.savetxt(cmgfile.permx, Kx, comments="", header="PERMI ALL")
    np.savetxt(cmgfile.permz, Kz, comments="", header="PERMK ALL")

    # create *.dat from brugge.tmpl
    with open("brugge.tmpl", "r") as tmpl, open(cmgfile.dat, "w") as dat:
        t = Template(tmpl.read())
        content = t.substitute(POROSITY_INC=cmgfile.poro, PERMI_INC=cmgfile.permx, PERMK_INC=cmgfile.permz)
        dat.write(content)

    # create *.rwd from report.tmpl
    with open("report.tmpl", "r") as tmpl, open(cmgfile.rwd, "w") as rwd:
        t = Template(tmpl.read())
        content = t.substitute(IRFFILE=cmgfile.irf)
        rwd.write(content)

    # call IMEX + Results Report
    with open(cmgfile.log, "w") as log:
        subprocess.check_call(["RunSim.sh", "imex", "2012.10", cmgfile.dat, "-log", "-wait"], stdout=log)
        subprocess.check_call(["report.exe", "-f", cmgfile.rwd, "-o", cmgfile.rwo], stdout=log)

    # oil rate SC for all 20 producer wells
    history = np.loadtxt(cmgfile.rwo, skiprows=6)

    # clean up
    for filename in cmgfile:
        os.remove(filename)

    return history[timestep,:]

# tuning parameters
ncomps, nsamples = 50, 100

# initial ensemble from disk (nfeatures x nsamples)
X = np.empty([60048, nsamples])
for j in xrange(nsamples):
    X[:,j] = np.loadtxt("prior/POR_{0:03d}.inc".format(j+1), skiprows=2)

# only 44550 of 60048 cells are active
mask = np.loadtxt("null.inc", dtype=bool, skiprows=2)
X = X[mask,:]

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
            np.savetxt("ensemble{0:03d}-{0:03d}.dat".format(i,j), ensemble)
            np.savetxt("lnprob{0:03d}-{0:03d}.dat".format(i,j), logp)
            np.savetxt("acceptance{0:03d}-{0:03d}.dat".format(i,j), sampler.acceptance_fraction)

        # update prior with posterior
        mprior = Nonparametric(ensemble)
        CSI = np.array(ensemble).T

        # we're done with this timestep, tell slaves to proceed
        pool.close()
    else:
        # wait from instructions from the master process
        pool.wait()
