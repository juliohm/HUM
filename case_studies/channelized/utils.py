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
## Created: 06 Feb 2014
## Author: Júlio Hoffimann Mendes

from os import devnull, remove
from subprocess import check_call
from tempfile import NamedTemporaryFile
import numpy as np

# make sure results are reproducible
ar2gems_seed = 1

# This function relies on a recent version of SGeMS with support to
# the command `LoadCartesianGrid` written by myself. The command is
# for sure available at https://github.com/juliohm/ar2tech-SGeMS-public
def filtersim(N):
    """
    Generate N realizations from 250x250 training image

    Every call produces a different ensemble with seed = 1,2,...
    """
    global ar2gems_seed

    ar2gems_input = """
LoadCartesianGrid  mtrue.dat::mguess

NewCartesianGrid  filtersimGrid::250::250::1::1.0::1.0::1.0::0::0::0::0.00

RunGeostatAlgorithm  filtersim_cont::/GeostatParamUtils/XML::<parameters>  <algorithm name="filtersim_cont" />     <GridSelector_Sim value="filtersimGrid" region=""  />     <Property_Name_Sim  value="perm" />     <Nb_Realizations  value="{nb_realizations}" />     <Seed  value="{seed}" />     <PropertySelector_Training  grid="mguess"   property="permeability"   region=""  />     <Patch_Template_ADVANCED  value="7 7 1" />     <Scan_Template  value="11 11 1" />     <Trans_Result  value="0"  />     <Hard_Data  grid=""   property=""   region=""  />     <Use_SoftField  value="0"  />     <Region_Indicator_Prop  value=""  />     <Active_Region_Code  value="" />     <Use_Previous_Simulation  value="0"  />     <Previous_Simulation_Prop  value=""  />     <Use_Region  value="0"  />     <Nb_Multigrids_ADVANCED  value="3" />     <Debug_Level  value="0" />     <Cmin_Replicates  value="10 10 10" />     <Data_Weights  value="0.5 0.3 0.2" />     <CrossPartition  value="1"  />     <KMeanPartition  value="0"  />     <Nb_Bins_ADVANCED  value="5" />     <Nb_Bins_ADVANCED2  value="2" />     <Use_Normal_Dist  value="0"  />     <Use_Score_Dist  value="1"  />     <Filter_Default  value="1"  />     <Filter_User_Define  value="0"  />   </parameters>

SaveGeostatGrid  filtersimGrid::proposed_ensemble.csv::csv
""".format(seed=ar2gems_seed, nb_realizations=N)

    with NamedTemporaryFile() as cmdfile, open(devnull, "wb") as trash:
        cmdfile.write(ar2gems_input)
        cmdfile.flush()
        check_call(["ar2gems", cmdfile.name], stdout=trash, stderr=trash)

    X = np.loadtxt("proposed_ensemble.csv", delimiter=",", skiprows=1)

    remove("proposed_ensemble.csv")
    ar2gems_seed += 1

    return X


def OPMSimulator(m, pool):
    """
    OPM-based blackoil simulator

    Parameters
    ----------
    m: ndarray
        flattened permeability field

    pool: emcee.utils.MPIPool
        MPI pool used for parallelism

    Returns
    -------
    d: ndarray
        flattened production history
    """
    # dump input to file
    basename = "rank%i" % pool.rank
    infile = basename+".dat"
    np.savetxt(infile, m, header="250x250 permeability field")

    # call external simulator
    logfile = basename+".log"
    with open(logfile, "w") as log:
        check_call(["./simulator", "-f", infile], stdout=log, stderr=log)

    # load output back
    outfile = basename+".out"
    d = np.loadtxt(outfile, skiprows=1, usecols=xrange(8)) # 8 producer wells

    # clean up
    remove(infile); remove(outfile); remove(logfile)

    return d.flatten()
