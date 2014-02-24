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
## Created: 11 Feb 2014
## Author: Júlio Hoffimann Mendes

import numpy as np
from os import remove
from time import time, sleep
from subprocess import Popen, check_call
from collections import namedtuple
from string import Template
from mpi4py import MPI

def CMGFile(basename):
    """
    A simple wrapper for retrieving CMG file extensions given the basename.
    """
    Extension = namedtuple("Extension", "dat out irf mrf rwd rwo log poro permx permz")
    return Extension(basename+".dat",
                     basename+".out",
                     basename+".irf",
                     basename+".mrf",
                     basename+".rwd",
                     basename+".rwo",
                     basename+".log",
                     basename+"-phi.inc",
                     basename+"-kx.inc",
                     basename+"-kz.inc")


def IMEX(m, timestep):
    """
    IMEX reservoir simulator

    Returns the history for a given timestep.
    """
    basename = "rank%i" % MPI.COMM_WORLD.Get_rank()
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

    # hardcode number of wells
    nwells = 20

    # call IMEX + Results Report
    with open(cmgfile.log, "w") as log:
        proc = Popen(["RunSim.sh", "imex", "2012.10", cmgfile.dat, "-log", "-wait"], stdout=log)

        start = time()
        while proc.poll() is None: # IMEX still running?
            if time() - start > 300: # 5min timeout
                proc.kill()
                break
            sleep(10)

        if proc.returncode == 0:
            check_call(["report.exe", "-f", cmgfile.rwd, "-o", cmgfile.rwo], stdout=log)
        else:
            # create dummy *.rwo file
            np.savetxt(cmgfile.rwo, np.zeros(nwells), header="\n"*5)

    # oil rate SC for all 20 producer wells
    history = np.loadtxt(cmgfile.rwo, skiprows=6)

    # clean up
    for filename in cmgfile:
        remove(filename)

    # return zero rate in case of premature termination
    return history[timestep,:] if timestep < history.shape[0] else np.zeros(nwells)
