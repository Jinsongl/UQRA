"""
Uncertainty Quantifican Toolbox 
===============================

This module contains tools for performing uncertainty
quantification 
"""
# The values are copied by reference, 
# So if you were to mutate MY_CONSTANT in any of the modules, it would mutate everywhere. 
# If you were to re-assign MY_CONSTANT in any of the modules, it would only affect that module. In this case, you must reference by attribute, i.e. mypackage.constants.MY_CONSTANT
# from museuq.constants import *
#from museuq.setup import setup 
from museuq.doe.ExperimentDesign import ExperimentDesign as DoE
from museuq.solver.Solver import Solver
from museuq.surrogate_model.SurrogateModel import SurrogateModel
from museuq.simParameters import simParameters


