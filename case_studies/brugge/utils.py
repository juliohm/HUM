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

import re
import numpy as np
from socket import gethostname
from os import remove, getcwd, getenv, path
from subprocess import Popen, check_call
from collections import namedtuple
from string import Template
from mpi4py import MPI

# IMEX work directory
workdir = getcwd()

def CMGFile(basename):
    """
    A simple wrapper for retrieving CMG file extensions given the basename.
    """
    # If on CENEPAD-PE cluster, avoid expensive I/O file transfers by
    # writing into the appropriate directory for this computational node.
    if re.match("super\d+", gethostname()):
        prefix = "/var/tmp/" + getenv("SLURM_JOB_ID")
        basename = path.join(prefix, basename)
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

# all timesteps for which there are measurements
alltimes = np.loadtxt("observation.csv", skiprows=2, usecols=[0])

def IMEX(m, timesteps=alltimes):
    """
    IMEX reservoir simulator

    Returns the history at given timestep(s).
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
        content = t.substitute(PORO_INC  = cmgfile.poro,
                               PERMI_INC = cmgfile.permx,
                               PERMK_INC = cmgfile.permz,
                               NET_INC   = path.join(workdir,"prior/NETGROSS_mean.inc"),
                               SW_INC    = path.join(workdir,"prior/SWCON_mean.inc"),
                               GRID_INC  = path.join(workdir,"brugge_grid.inc"),
                               NULL_INC  = path.join(workdir,"null.inc"),
                               WELL_INC  = path.join(workdir,"well_operation.inc"))
        dat.write(content)

    # create *.rwd from report.tmpl
    with open("report.tmpl", "r") as tmpl, open(cmgfile.rwd, "w") as rwd:
        t = Template(tmpl.read())
        content = t.substitute(IRFFILE=cmgfile.irf, TIMESTEPS=" ".join(str(step) for step in timesteps))
        rwd.write(content)

    # call IMEX + Results Report
    with open(cmgfile.log, "w") as log:
        proc = Popen(["mx201210.exe", "-f", cmgfile.dat, "-log", "-wait", "-dd"], stdout=log)
        proc.wait() # wait for IMEX exit code

        # columns in output spreadsheet (lexicographic order)
        wells = [0,11] + range(13,20) + range(1,11) + [12]

        if proc.returncode == 0:
            # get oil rate SC for all 20 producer wells
            check_call(["report.exe", "-f", cmgfile.rwd, "-o", cmgfile.rwo], stdout=log)
            history = np.loadtxt(cmgfile.rwo, skiprows=6, usecols=wells)
        else:
            # IMEX has failed, nullify history
            history = np.zeros([len(timesteps), len(wells)])

    # clean up
    for filename in cmgfile:
        try: remove(filename)
        except OSError: continue

    return history.flatten()
