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
import matplotlib.pyplot as plt
import seaborn as sbs
from matplotlib import animation


def duffing_equation(x0,v0,delta,alpha,beta,response,source_vals=None):

    def animate(i):
        """
        Update the image for iteration i of the Matplotlib animation.
        """
        ln1.set_data(x[i], V(x[i]))
        ln2.set_data(t[:i+1], x[:i+1])
        ax2.set_xlim(0, t[i])
        ln3.set_data(x[:i+1], xdot[:i+1])
        # if not i % pstep:
            # scat1.set_offsets(X[i])
        return

    V = lambda x: beta/4 * x**4 - alpha/2 * x**2 
    dVdx = lambda x: beta*x**3 - alpha*x

# The potential energy function on a grid of x-points.
    xgrid = np.linspace(-5, 5, 100)
    Vgrid = V(xgrid)
    t, x, xdot = response[:,0], response[:,1], response[:,2]
    dt = t[2]-t[1]
    
# # The animation
    fig, ax = plt.subplots(nrows=2,ncols=2)
    ax1 = ax[0,0]
    ax1.plot(xgrid, Vgrid)
    # ax1.set_ylim(-0.3, 0.15)
    # ax1.set_xlim(-1.2*np.sqrt(2*alpha/beta), 1.2*np.sqrt(2*alpha/beta))
    ln1, = ax1.plot([], [], 'mo')
    ax1.set_xlabel(r'$x / \mathrm{m}$')
    ax1.set_ylabel(r'$V(x) / \mathrm{J}$')

# Position as a function of time
    ax2 = ax[1,0]
    ax2.set_xlabel(r'$t / \mathrm{s}$')
    ax2.set_ylabel(r'$x / \mathrm{m}$')
    ln2, = ax2.plot(t[:100], x[:100])
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
    ax4.scatter(x, xdot, s=2, lw=0, c=sbs.color_palette()[0])
    scat1 = ax4.scatter([x0], [v0], lw=0, c='m')
    plt.tight_layout()



    anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=2)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    anim.save('duffing.mp4', writer=writer)
    # plt.show()

