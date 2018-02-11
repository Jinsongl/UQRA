#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import Norway5_2 as norway2
import Norway5_3 as norway3
import numpy as np
import chaospy as cp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utility'))
from utility import *
# from UQ_FOWT.utility.dataIO import *

class environment(object):

    def __init__(self, site, isTrunc=False,bnd=[3,25]):
        self.site = site
        self.isTurnc = isTrunc
        self.bnd = bnd
        
    def getdistFun(self):
        """
        Return a list of functions, each function take given arguments to return 
        a distribution
        """
        if self.site == 'Norway5_2':
            return [norway2.dist_Hs, norway2.distC_Tp]
        if self.site == 'Norway5_3':
            return [norway3.dist_Uw, norway3.distC_Hs, norway3.distC_Tp]
    def getdistC(self,**kwargs):
        """
        Return a list of conditional distributions in predefined priority order.
        based on given vals (either percentile q or physical values phy) 
        kwargs: string-ndarray pair
                ppf - vals[(ndim,) array]
                phy - vals[(ndim,) array]
        """
        if kwargs is not None:
            if len(kwargs) != 1:
                raise ValueError("Only one key-value pair is expected")
            key, vals = kwargs.popitem()
        else:
            raise ValueError("key-value pair is NONE") 

        distFun = self.getdistFun()
        if len(distFun) != len(vals):
            raise ValueError(" Number of conditional distributions and q are not equal")
        
        distC = []
        phy = [] # All conditional distributions except the first one are defined based on physical variables
        if self.site == 'Norway5_2': 
            # Hs distribution is pievewise 
            distcur = distFun[0](key,vals[0])
        elif self.site == 'Norway5_3':
            distcur = distFun[0](self.isTurnc, self.bnd)
        else:
            raise ValueError("Site not defined")

        distC.append(distcur)
        if key == "ppf":
            phy.append(float(distcur.inv(vals[0])))
        elif key =="phy":
            phy = vals
        else:
            raise ValueError("key-value pair is not defined") 

        for i in range(1,len(vals)):
            distcur = distFun[i](*phy[:i])
            distC.append(distcur)
            if key == "ppf":
                phy.append(float(distcur.inv(vals[i])))
        phy = np.array(phy)
        return distC,phy


    def getsamp_ppf(self,q):
        """
        Return samples based on given percentiles (inverse of cdf)
        q: (ndim*nsamples) array 
        """
        vals = []
        if q.ndim == 1:
            _,vals = self.getdistC(ppf=q)
        else:
            for val in q.T:
                _, val = self.getdistC(ppf=val)
                vals.append(val)
            vals = np.array(vals).T
        return vals

    def phy2norm(self, vals):
        """
        Transform variables from physical domain to standard normal domain
        phy must be in the same order as defined in the conditional distributions
        vals: ndim * nsamples
        """
        dist_norm = cp.Normal()
        norm = []
        for val in vals.T:
            v = []
            distC,_ = self.getdistC(phy=val)
            for i, dist in enumerate(distC):
                q = float(dist.cdf(val[i]))
                v.append(float(dist_norm.inv(q)))
            norm.append(v)
        norm = np.array(norm).T
    def zeta2phy(self, dist_zeta, zeta):
        """
        Transform variables from zeta domain to physical domain
        Inputs:
            dist_zeta: list of zeta distributions
            # zeta: zeta to be 
        """
        vals = []

        if zeta.ndim == 1:
            ppf = np.array([dist.cdf(zeta[i]) for i, dist in enumerate(dist_zeta)])
            vals = self.getsamp_ppf(ppf)
        else:
            for val in zeta.T:
                ppf = np.array([dist.cdf(val[i]) for i, dist in enumerate(dist_zeta)])
                vals.append(self.getsamp_ppf(ppf))
            vals = np.array(vals).T
        return vals


    def getEC2D(self,T=50,simTime=1,numPts=3600, u3=0.5, dist_zeta=[]):
        """
        2D Environmental Contour return variables in physical domain, only works for 2D
        """
        dist_norm = cp.Normal()
        zeta    = []
        val     = []
        beta    = np.abs(dist_norm.inv(simTime/(T*365.25*24)))
        theta   = np.linspace(0,360,numPts) 
        beta    = np.sqrt(beta**2 - dist_norm.inv(1-u3)**2)
        u       = beta *np.column_stack((np.cos(theta),np.sin(theta)))
        ppf     = dist_norm.cdf(u)
        print "Generating 2D Environmental Contour..."
        for p in ppf:
            phy = self.getsamp_ppf(p).tolist()
            if dist_zeta:
                zeta = [float(dist.inv(p[i])) for i, dist in enumerate(dist_zeta)]
            val.append(zeta+phy)
        val = np.array(val)
        return val


def main():
    dist_norm = cp.Normal()
    q1 = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 0.1,0.3,0.5,0.7,0.9,1-1e-2,1-1e-3,1-1e-4,1-1e-5, 1-1e-6])
    q2 = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 0.1,0.3,0.5,0.7,0.9,1-1e-2,1-1e-3,1-1e-4,1-1e-5, 1-1e-6])
    q = np.array([q1,q2])
    test = environment('Norway5_2')
    v = test.getEC2D(numPts=3600,isSave=True,filename='test.csv')
    v = test.getsamp_ppf(q)
    print "2D: ", v.shape
    v = test.phy2norm(v,isSave=False)
    print "2D: ", dist_norm.cdf(v)
if __name__=='__main__':
    main()
