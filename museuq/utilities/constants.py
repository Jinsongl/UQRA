# settings.py 
# Global settings for variables

import matplotlib.pyplot as plt
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


DOE_METHOD_FULL_NAMES = {
    "gq"    : "QUADRATURE"  , "quad"  : "QUADRATURE",
    "mc"    : "MONTE CARLO" , "fix"   : "FIXED POINT"
    } 

DOE_RULE_FULL_NAMES = {
    "cc"    : "clenshaw_curtis"  , "leg"  : "gauss_legendre"  , "pat" : "gauss_patterson",
    "gk"    : "genz_keister"     , "gwel" : "golub_welsch"    , "leja": "leja",
    "hem"   : "gauss_hermite"    , "lag"  : "gauss_laguerre"  , "cheb": "gauss_chebyshev",
    "hermite"   :"gauss_hermite" ,
    "legendre"  :"gauss_legendre",
    "jacobi"    :"gauss_jacobi"  ,
    "R" : "Pseudo-Random"   , "r" : "Pseudo-Random"     ,
    "RG": "Regular Grid"    , "rg": "Regular Grid"      ,
    "NG": "Nested Grid"     , "ng": "Nested Grid"       ,
    "L" : "Latin Hypercube" , "l" : "Latin Hypercube"   ,
    "S" : "Sobol"           , "s" : "Sobol"             ,
    "H" : "Halton"          , "h" : "Halton"            , 
    "M" : "Hammersley"      , "m" : "Hammersley"        ,
    "FIX": "Fixed point"    ,"fix": "Fixed point"       ,
    }
