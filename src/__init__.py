"""
Uncertainty Quantifican Toolbox for Floating Offshore Wind Turbine
===============================

This module contains tools for performing uncertainty
quantification of FOWT based on chaospy
"""
import envi
import doe
import solver
import utilities
import MUSEPlot
# from utilities import dataIO, gen_gauss_time_series, get_stats #file_rename, 

pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# pltlinestyles = ['solid','dashed','dashdotted','dotted','loosely dashed','loosely dashdotted']
"""
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
"""
pltlinestyles = [ (0, (1, 5)),(0, (3, 5, 1, 5)),(0, (5, 5)),(0, ()),(0, (3, 5, 1, 5, 1, 5)), 
        (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10)),(0, (1, 10)), 
        (0, (5, 1)),  (0, (3, 1, 1, 1)),   (0, (3, 1, 1, 1, 1, 1)),(0, (1, 1))  ]

pltmarkers = ['o','v','s','d','+','*']*10

