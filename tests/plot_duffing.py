#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import context
import numpy as np
import envi, doe, solver, utilities

from envi import environment
from metaModel import metaModel
from simParams import simParameter
from run_sim import run_sim
from solver.dynamic_models import lin_oscillator
from solver.dynamic_models import duffing_equation
from solver.dynamic_models import duffing_oscillator
from solver.static_models import ishigami
from solver.static_models import poly5


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs

# The potential and its first derivative, as callables.
# V = lambda x: 0.5 * x**2 * (0.5 * x**2 - 1)
# dVdx = lambda x: x**3 - x

# # The potential energy function on a grid of x-points.
# xgrid = np.linspace(-1.5, 1.5, 100)
# Vgrid = V(xgrid)

# plt.plot(xgrid, Vgrid)
# plt.xlabel('$x$')
# plt.ylabel('$V(x)$')


# Set up the motion for a oscillator with initial position
# x0 and initially at rest.
x0, v0 = 0, 0
tmax, t_trans = 180, 30
omega = 1.4
gamma, delta = 0.39, 0.1
dt_per_period = 100
alpha, beta = 1, 1
# Solve the equation of motion.
source_func =lambda t, kwargs=None: gamma*np.cos(omega*t)

t, X, dt, pstep = duffing_equation(tmax, dt_per_period, x0, v0, gamma, delta, omega, t_trans=t_trans, alpha=alpha, beta = beta)
x, xdot = X.T
dt = 2*np.pi/omega/dt_per_period
omega0 = np.sqrt(alpha)
mu = beta/alpha
zeta = delta/(2*omega0)
t1, X1, dt1, pstep1 = duffing_oscillator(tmax, dt, x0, v0, zeta,omega0,mu, t_trans=t_trans,source_func=source_func)
x1, xdot1 = X1.T
# # The animation
fig, ax = plt.subplots(nrows=2,ncols=2)
# ax1 = ax[0,0]
# ax1.plot(xgrid, Vgrid)
# ax1.set_ylim(-0.3, 0.15)
# ln1, = ax1.plot([], [], 'mo')
# ax1.set_xlabel(r'$x / \mathrm{m}$')
# ax1.set_ylabel(r'$V(x) / \mathrm{J}$')

# Position as a function of time
ax2 = ax[1,0]
ax2.set_xlabel(r'$t / \mathrm{s}$')
ax2.set_ylabel(r'$x / \mathrm{m}$')
# ln2, = ax2.plot(t[:100], x[:100])
ln2, = ax2.plot(t, x)
ln2, = ax2.plot(t1, x1)
ax2.set_ylim(np.min(x), np.max(x))

# Phase space plot
ax3 = ax[1,1]
ax3.set_xlabel(r'$x / \mathrm{m}$')
ax3.set_ylabel(r'$\dot{x} / \mathrm{m\,s^{-1}}$')
ln3, = ax3.plot([], [])
ax3.set_xlim(np.min(x), np.max(x))
ax3.set_ylim(np.min(xdot), np.max(xdot))

# Poincaré section plot
ax4 = ax[0,1]
ax4.set_xlabel(r'$x / \mathrm{m}$')
ax4.set_ylabel(r'$\dot{x} / \mathrm{m\,s^{-1}}$')
ax4.scatter(x[::pstep], xdot[::pstep], s=2, lw=0, c=sbs.color_palette()[0])
scat1 = ax4.scatter([x0], [v0], lw=0, c='m')
plt.tight_layout()


plt.show()
# def animate(i):
    # """Update the image for iteration i of the Matplotlib animation."""

    # ln1.set_data(x[i], V(x[i]))
    # ln2.set_data(t[:i+1], x[:i+1])
    # ax2.set_xlim(t_trans, t[i])
    # ln3.set_data(x[:i+1], xdot[:i+1])
    # if not i % pstep:
        # scat1.set_offsets(X[i])
    # return

# anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=1)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# anim.save('duffing.mp4', writer=writer)

