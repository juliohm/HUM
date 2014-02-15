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

from collections import namedtuple

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
