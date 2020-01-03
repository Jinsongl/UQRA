"""
Top-level package for Uncertainty Quantifican Toolbox 
===============================
This module contains tools for performing uncertainty quantification 

"""

# The values are copied by reference, 
# So if you were to mutate MY_CONSTANT in any of the modules, it would mutate everywhere. 
# If you were to re-assign MY_CONSTANT in any of the modules, it would only affect that module. In this case, you must reference by attribute, i.e. mypackage.constants.MY_CONSTANT

import importlib
import numpy as np
import pandas as pd

# from museuq.doe.ExperimentDesign import ExperimentDesign as DoE
from museuq.doe.quadrature import QuadratureDesign
from museuq.doe.random_design import RandomDesign
from museuq.doe.lhs import LatinHyperCube as LHS
from museuq.doe.optimal_design import OptimalDesign 

# from museuq.solver.Solver import Solver
from museuq.solver.dynamic import linear_oscillator
from museuq.solver.static import * 


# from museuq.surrogate_model.SurrogateModel import SurrogateModel
from museuq.simParameters import simParameters
from museuq.surrogate_model.polynomial_chaos_expansion import PolynomialChaosExpansion as PCE
from museuq.utilities.ErrorClass import NullError, IidError, CovError

