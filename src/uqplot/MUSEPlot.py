#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""


import sys
sys.path.append('/Users/jinsongliu/Box Sync/Dissertation_UT/OMAE2018/UQ_FOWT')
import os
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.cm as cm
from math import atan2
from utility.dataIO import *



MUSEmarker  = ['o','s','p','*','x','d','h']

MUSEcolors  = ['b','g','r','c','m','y','k']
# labels      = {
            # 'QUAD'  : 'QuadPts: ',
            # 'QUANT' : 'LHS Quantile: ',
            # 'MC'    : 'MCS: ',
            # 'ED'    : 'DOE: '
            # }


def setMSUEPlot():

    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 18}
    figure = {'figsize': (10,10)}

    mpl.rc('font',  **font)
    mpl.rc('figure', **figure)
    plt.clf()



def ECPlot(prefix, space,figHandle=None, figname='Norway5EC2DHT'):
    setMSUEPlot()
    if figHandle:
        fig, ax = figHandle
    else:
        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

    if space.upper()=='ZETA':
        space2plot = np.arange(0,2)
        pos = np.array([[1,1,1,1,1,1],[3,5.5,8,10,12,13]]).T
    elif space.upper()=='PHY':
        space2plot = np.arange(2,4)
        pos = np.array([[2,2,1.8,1.8,1.8,1.8],[15,20,23,27,30,32]]).T
    else:
        ValueError('Space is Zeta or Phy')

    filelist = [f for f in os.listdir(os.getcwd()) if f.startswith(prefix)]
    print prefix, " Number of files:", len(filelist)
    for i, filename in enumerate(filelist):
        print "Processing file:", i
        p = float(filename[21:-4])
        # data = np.genfromtxt(filename,delimiter=',')
        data = iter_loadtxt(filename)
        data = data[:,space2plot]
        mu = np.mean(data, axis=0)
        data = data - mu
        data = data.tolist()
        data.sort(key=lambda c:atan2(c[0], c[1]))
        data = np.array(data)
        data = data + mu
        # plt.plot(data[:,0],data[:,1],'-', label='50-year EC' + filename[21:-4])
        ax.plot(data[:,0],data[:,1],'-',color='Gray')
        ax.text(pos[i,0],pos[i,1], 'p3='+'{:.0e}'.format(p),fontsize=10,color='Gray')

        if space.upper() == 'ZETA':
            ax.set_title("Environmental Contour in $\zeta$ Space")
            ax.set_xlabel("$\zeta_1$")
            ax.set_ylabel("$\zeta_2$")
            ax.set_xlim(0,20)
            ax.set_ylim(0,20)
            # plt.axis('equal')
        else:
            ax.set_title("Environmental Contour in Physical Space")
            ax.set_ylabel("Peak period, $T_{p} (s)$")
            ax.set_xlabel("Significant wave height, $H_s (m)$")
        ax.grid('on')
        ax.legend(loc=0)
        fig.savefig(figname+'_'+space+'.eps')
    return (fig,ax)

def sampPlot(space, prefix,labels=[], figHandle=None):
    if figHandle:
        fig, ax = figHandle
    else:
        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

    if space.upper()=='ZETA':
        space2plot = np.arange(0,2)
        pos = np.array([[1,1,1,1,1,1],[3,5.5,8,10,12,13]]).T
    elif space.upper()=='PHY':
        space2plot = np.arange(-2,0)
        pos = np.array([[2,2,1.8,1.8,1.8,1.8],[15,20,23,27,30,32]]).T
    else:
        ValueError('Space is Zeta or Phy')

    filelist = [f for f in os.listdir(os.getcwd()) if f.startswith(prefix)]
    print prefix, "Number of files:", len(filelist)
    for i, filename in enumerate(filelist):
        print "Processing file:",filename 
        # data = np.genfromtxt(filename,delimiter=',')
        data = iter_loadtxt(filename)
        data = data[:,space2plot]
        if labels:
            ax.scatter(data[:,0],data[:,1],s=50,marker = MUSEmarker[i],color=MUSEcolors[i],label=labels[i])
        else:
            ax.scatter(data[:,0],data[:,1],s=50,marker = MUSEmarker[i],color=MUSEcolors[i])

        if space.upper() == 'ZETA':
            ax.set_title("Traning Samples in $\zeta$ Space")
            ax.set_xlabel("$\zeta_1$")
            ax.set_ylabel("$\zeta_2$")
            ax.set_xlim(0,20)
            ax.set_ylim(0,20)
            # plt.axis('equal')
        else:
            ax.set_title("Training Samples in Physical Space")
            ax.set_ylabel("Peak period, $T_{p} (s)$")
            ax.set_xlabel("Significant wave height, $H_s (m)$")
            ax.set_xlim(0,22)
            ax.set_ylim(0,50)
        ax.grid()
        ax.legend(loc=0)
        fig.savefig('TrainingSamples_'+space+'.eps')
    
    return (fig,ax)


def ExPlot(data,q=1e-4, R=1,labels=[],color='k',figHandle=None,figname='ExceedencePlot'):
    if figHandle:
        fig, ax = figHandle
    else:
        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    # data = iter_loadtxt(filename)
    M = int(len(data)/R)
    if M < 1.0/q:
        raise ValueError('Not enough samples to get specified quantile ', str(q))

    data = data.reshape((M,R))
    conf=[]

    for i in xrange(R):
        ecdf = ECDF(data[:,i])
        conf.append(next(ecdf.x[i] for i, xx in enumerate(1-ecdf.y) if xx<q))
        x,y = ecdf.x, ecdf.y
        M1  = int(M * 0.9)
        M2  = M - M1
        ind1 = np.linspace(0,M1,num=int(M1/10),dtype=int)
        ind2 = np.linspace(M1+1, M, num=M2, dtype=int)
        ind = np.append(ind1, ind2)
        if labels:
            ax.plot(x, 1-y, color=color, label=labels[i])
        else:
            ax.plot(x, 1-y, color=color)

        
    # for i, filename in enumerate(filelist):
        # print "Processing file:", i
        # data = np.genfromtxt(filename,delimiter=',')
        # # print data
        # ecdf = ECDF(data[:,4])
        # conf.append(next(ecdf.x[i] for i, xx in enumerate(1-ecdf.y) if xx<1e-4))
        # x,y = ecdf.x,ecdf.y
        # if len(x) > 1e6:
            # ind = np.linspace(0,1e5,1e4,dtype=np.int32)
            # ind = np.append(ind, np.arange(1e5,1e6,dtype=np.int32))
            # x=x[ind]
            # y=y[ind]
        # plt.plot(x, 1-y)
        

    conf.sort()
    q0 = 1.0/M
    print "Exceedence interval: [", conf[0], conf[-1], " ]"
    ax.plot([conf[0],conf[0]],[q0,q], '--', color='Gray')
    ax.plot([conf[-1],conf[-1]],[q0,q], '--', color='Gray')
    ax.plot([0, conf[-1]],[q,q], '--', color='Gray')
    ax.text(2,1e-5,'$10^{-'+ '{:.1E}'.format(q)[-1] +'}$' +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    # ax.text(2,1e-5, '{:.1E}'.format(q) +' Exceedence Interval:\n ['+ '{:.2f}'.format(conf[0]) +' , '+ '{:.2f}'.format(conf[-1])+']')
    ax.set_yscale('log')
    ax.set_xlabel('QoI: $f^{T}_{max}$')
    ax.set_xlim(0,22)
    ax.set_ylabel('Exceedence')
    ax.set_title('Exceedence plot of SDOF system with fixed phases')
    plt.savefig(figname + '.eps')
    return (fig,ax)


# def CVPlot(x,y,*arg)
    # plt.plot(x,y, arg)

   # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(data[:,2], data[:,3],valiData.Y, 'o',label='Validate Data')
    # ax.plot(data[:,2], data[:,3],f_cv[:,0], 'o',label='5')
    # # ax.plot(valiData.X[:,0], valiData.X[:,1],f_cv[:,1], 'o', label='6')
    # ax.legend()
    # plt.show()




