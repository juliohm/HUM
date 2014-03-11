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

import numpy as np
import pylab as pl
import numpy.ma as ma

lnprob_start = np.loadtxt("lnprob0001.dat")
lnprob_end   = np.loadtxt("lnprob1000.dat")

# purge outliers
lnprob_start = ma.masked_less(lnprob_start, -2000).compressed()
lnprob_end   = ma.masked_less(lnprob_end, -2000).compressed()

try:
    from statsmodels.graphics.boxplots import beanplot
    fig = pl.figure()
    ax = fig.add_subplot(111)
    beanplot((lnprob_start, lnprob_end),
              ax=ax, jitter=False, labels=["prior","posterior"],
              plot_opts={"bean_mean_color":"c",
                         "bean_median_color":"m",
                         "violin_fc":"w",
                         "violin_alpha":1})
    ax.set_xlabel("prior vs. posterior")
    ax.set_ylabel("log-probability")
    pl.show()
except ImportError:
    print "Consider installing StatsModels package for bean plots."
finally:
    fig.savefig("lnprob.pdf", bbox_inches="tight")
