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
## Created: 12 Mar 2014
## Author: Júlio Hoffimann Mendes

import pylab as pl

def plot_lnprob(tup):
    """
    Bean plot of prior vs. posterior log-probabilities

    Parameters
    ----------
    tup: tuple of arrays
        Tuple (lnprob_prior, lnprob_post) of log-probabilities

    Returns
    -------
    fig: matplotlib figure
        The figure with the plot
    """
    try:
        from statsmodels.graphics.boxplots import beanplot
        fig = pl.figure()
        ax = fig.gca()
        beanplot(tup, ax=ax, jitter=False, labels=("prior","posterior"),
                 plot_opts={"bean_mean_color":"c",
                            "bean_median_color":"m",
                            "violin_fc":"w",
                            "violin_alpha":1})
        ax.set_xlabel("prior vs. posterior")
        ax.set_ylabel("log-probability")
        return fig
    except ImportError:
        print "Consider installing StatsModels package for bean plots."


def plot_acceptance(acc):
    """
    Plot acceptance fraction for each walker

    Parameters
    ----------
    acc: array
        Acceptance fraction for each walker

    Returns
    -------
    fig: matplotlib figure
        The figure with the plot
    """
    nwalkers = acc.size
    mean = acc.mean()
    fig = pl.figure()
    ax = fig.gca()
    pl.bar(xrange(1, nwalkers+1), acc, align="center", color="k", edgecolor="c")
    pl.axhline(mean, color="r", linestyle="--", dash_capstyle="round")
    ax.annotate("mean acceptance = %.1f %%" % (100*mean),
                color="r", xy=(10, mean), xytext=(10, mean+0.02))
    ax.set_xlim(0, nwalkers+1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("walker index")
    ax.set_ylabel("acceptance fraction")
    return fig
