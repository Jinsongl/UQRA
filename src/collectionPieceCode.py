#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
samp_zeta = spl.genVar(QuantParams.schemePts[0],QuantParams, QuantModel)
ind = quantInd(EDParams.schemePts[0])
samp = []
for i,_ in enumerate(ind):
    samp.append(samp_zeta[:,i].T)
    samp.append(samp_zeta[:,-i].T)
samp = np.array(samp).T
samp1 = siteEnvi.zeta2phy(dist_zeta, samp_zeta)
res = np.vstack((samp_zeta,samp1)).T

dataIO.saveData(res, 'test.csv', QuantParams.outdirName,isMove=False)


EC50 = siteEnvi.getEC2D(numPts=360,dist_zeta=dist_zeta, u3=0.00001)
res = []
for arg in EC50[:,2:]:
    f_obs = slv.SDOF(QuadParams,*arg)
    f_stats = getStats.getStats(f_obs, stats=QuadParams.stats)
    res.append(f_stats[2,4])
res = np.array(res)
val = np.column_stack((EC50,res))
dataIO.saveData(val, 'Norway5EC2D_HT_50_u3_0.00001.csv', QuadParams.outdirName,isMove=False)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(valiData.X[:,0], valiData.X[:,1],valiData.Y, 'ro')
    QuadModel.crossValidate(valiData)
    y1=[]
    for f in QuadModel.f_hats:
        y1.append([float(f(*val)) for val in valiData.X])
    y1 = np.array(y1)
    for i, y_0 in enumerate(y1):
        ax.plot(valiData.X[:,0], valiData.X[:,1], y_0.T, label = 'QUAD ' )

    QuantModel.crossValidate(valiData)
    y2=[]
    for i, f in enumerate(QuantModel.f_hats):
        y2.append([float(f(*val)) for val in valiData.X])
    y2 = np.array(y2)
    for i, y_0 in enumerate(y2):
        ax.plot(valiData.X[:,0], valiData.X[:,1], y_0.T, label = 'QUANT' )

    MCModel.crossValidate(valiData)
    y3=[]
    for i, f in enumerate(MCModel.f_hats):
        y3.append([float(f(*val)) for val in valiData.X])
    y3 = np.array(y3)
    for i, y_0 in enumerate(y3):
        ax.plot(valiData.X[:,0], valiData.X[:,1], y_0.T, label = 'MC' )

    EDModel.crossValidate(valiData)
    y4=[]
    for i, f in enumerate(EDModel.f_hats):
        y4.append([float(f(*val)) for val in valiData.X])
    y4 = np.array(y4)
    for i, y_0 in enumerate(y4):
        ax.plot(valiData.X[:,0], valiData.X[:,1], y_0.T, label = 'ED' )


    ax.legend()
    plt.show()
    print QuadModel.CVError
    dataIO.saveData(np.array(QuadModel.CVError),'CVError.csv', QuadParams.outdirName, mode='a', isMove=False)

