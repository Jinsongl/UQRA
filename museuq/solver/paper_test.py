#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import matplotlib.pyplot as plt
import utilities as utlt
from dynamic_models import duffing_oscillator



def psd_input(f, c=2):
    """
    Test FFT and iFFT for spectrum and acf 
    F(w) = Fourier(f(t))
    where
    F(w) = 2c / (c**2 + w**2)
    f(t) = e^(-c|t|)

    Arguments:
        f: frequencies to be evaluated at (Hz)
        c: arbitrary real constant larger than 0
    Returns:
        sf: psd value at specified f
        sa: approximated area under psd curve with specified f
        
    """
    f = 2 * np.pi * f
    sf = 2*c/(c**2 + f**2)
    df = f[1] - f[0]
    sa = np.sum(sf*df) 
    return sf, sa

zeta = 0.02
omega0 = 1
tmax = 100
dt = 0.1

Hw = lambda w: np.sqrt(1/((w**2 - omega0**2 )**2 + (2*zeta*omega0)**2 )) 


