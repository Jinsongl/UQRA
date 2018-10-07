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
import itertools
import numpy.linalg as nla
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utility'))
# from utility import *
# from UQ_FOWT.utility.dataIO import *

def sphere_points(ndim, r=1, npts=360):
    theta   = np.linspace(0,2*np.pi,npts) 
    cord = None
    if ndim == 2:
        x = np.cos(theta) 
        y = np.sin(theta)
        cord = r * np.array([x,y])
    elif ndim == 3:
        for u in np.linspace(-1,1,npts):
            x = np.sqrt(1-u**2) * np.cos(theta) 
            y = np.sqrt(1-u**2) * np.sin(theta)
            z = np.ones(size(x)) * u
            cord_ = r * np.array([x,y,z])
        cord = cord_ if not cord else np.vstack((cord, cord_))
    else: 
        raise ValueError('ndim not defined')

    return cord

class environment(object):

    def __init__(self, site, isTrunc=False,bnd=[3,25]):
        self.site = site
        self.isTurnc = isTrunc
        self.bnd = bnd
        
    def get_distfuncs(self):
        """
        Return a list of functions, each function take given arguments to return 
        a distribution
        """
        if self.site == 'Norway5_2':
            return [norway2.dist_Hs, norway2.distC_Tp]
        if self.site == 'Norway5_3':
            return [norway3.dist_Uw, norway3.distC_Hs, norway3.distC_Tp]
    def get_cdists(self, vals, key='phy'):
        """
        Return a list of conditional distributions in predefined priority order.
        based on given vals (either percentile q or physical values phy) 
        Only for one set!
        cdists[0] <-- phy[0] <=> vals[0]  # the first one defined based on either physical or percentile value
        cdists[1] <-- phy[0] # From the second, conditional distributions are defined based solely on physical values 
        cdists[2] <-- phy[0,1]
        cdists[3] <-- phy[0,1,2]
        ...

        Arguments:
        vals: ndarray, (ndim,) or (ndim-1, ) 
        key: string,  'ppf' or 'phy'

        Returns:
        cdists: list of conditional distribution 
        phy: if key == 'phy', return vals. if key == 'ppf', return physical values 
                corresponding to those ppf.
        """
        dist_funcs = self.get_distfuncs()

        assert len(dist_funcs) == len(vals) or len(dist_funcs) == len(vals)+1,\
                'Dimension of distributions(ndim:{0:3d}) and random variables(ndim:{1:3d}) not correct'\
                .format(len(dist_funcs), len(vals))
        
        cdists   = []
        phy     = [] # All conditional distributions except the first one are defined based on physical variables
        if self.site == 'Norway5_2': 
            # Hs distribution is pievewise 
            distcur = dist_funcs[0](vals[0], key=key)
        elif self.site == 'Norway5_3':
            distcur = dist_funcs[0](self.isTurnc, self.bnd)
        else:
            raise ValueError("Site not defined")

        cdists.append(distcur)
        if key == "ppf":
            phy.append(float(distcur.inv(vals[0])))
        elif key =="phy":
            phy = vals
        else:
            raise ValueError("key is not defined") 

        for idist in range(1,len(dist_funcs)):
            distcur = dist_funcs[idist](*phy[:idist])
            cdists.append(distcur)
            if key == "ppf":
    # if vals is (ndim-1,) then the last physical variable value is appended as nan 
                phy.append(float(distcur.inv(vals[idist])) if idist < len(vals) else np.nan)
        phy = np.array(phy)
        assert len(cdists) == len(dist_funcs), \
                'Expecting {:3d} conditional distributions, {:3d} returned'.format(len(dist_funcs), len(cdists))

        return cdists, phy


    def get_invcdf(self,qs, key='phy'):
        """
        Return inverse cdf (samples based on given percentiles )

        Arguments:
        qs: array-like of shape (ndim, nsamples) 

        Return:
        vals: ndarray  of shape (ndim, nsamples)
        """
        qs = np.asfarray(qs)
        vals = np.empty(qs.shape)
        if qs.ndim == 1:
            _,vals = self.get_cdists(qs, key=key) 
        else:
            for i, q in enumerate (qs.T):
                _,val = self.get_cdists(q, key=key) 
                vals[:,i] = val 
        return vals

    def phy2norm(self, vals):
        """
        Transform variables from physical domain to standard normal domain
        phy must be in the same order as defined in the conditional distributions

        Arguments:
        vals: physical variable (x1,x2,...,xn) of shape (ndim, nsamples)

        Returns:
        u: variables in standard normal domain, same shape as vals
        """
        dist_norm = cp.Normal()
        u = []
        invalid = []
        for val in vals.T:
            v = []
            cdists,_ = self.get_cdists(val, key='phy')
            for i, dist in enumerate(cdists):
                q = float(dist.cdf(val[i]))
                # print(q)
                try:
                    v.append(float(dist_norm.inv(q)))
                except AssertionError:
                    print('Invalid sea states: Uw = {:.2f}, Hs = {:.2f}'.format(val[0], val[1]))
                    invalid.append(i)
            u.append(v)
        ## remove invalid sea state sets 
        ind = list(set(np.arange(vals.shape[1])).difference(set(invalid)))
        u = np.array(u).T
        u = u[:, np.array(ind, dtype=np.int32)]
        return u, invalid


    def zeta2phy(self, dist_zeta, zeta):
        """
        Transform variables from zeta domain (selected underlying random 
        variable for polynomial chaos) to physical domain
        dist_zeta are assumed to be same for all random variables for NOW. It 
        should accept different distributions from Wiener-Askey scheme

        Arguments:
            dist_zeta: list of marginal zeta distributions
            zeta: variables in zeta domain to be transformed

        Returns:

        """
        zeta = np.asfarray(zeta)
        vals = np.empty(zeta.shape)
        assert len(dist_zeta) == zeta.shape[0]
        for i, val in enumerate (zeta.T):
            q = list(map(lambda dist, x: float(dist.cdf(x)), dist_zeta, val))
            vals[:,i] = self.get_invcdf(q, key='ppf')
        assert vals.shape == zeta.shape
        return vals


    def getEC2D(self,T=50,simTime=1,numPts=3600, u3=0.5, dist_zeta=[]):
        """
        2D Environmental Contour return variables in physical domain, only works for 2D

        Arguments:
        T: return period
        simTime: simulaiton duration in hour
        numPts: number of points on the unit circle
        u3: exceedence probability of third variable 0.5: median
        dist_zeta: zeta distributions selected for polynomial chaos

        Returns:
        phy, ndarray of shape(numPts, ndim)
        zeta, ndarray of shape(numPts, ndim) if any
        """
        dist_norm = cp.Normal()
        beta    = np.abs(dist_norm.inv(simTime/(T*365.25*24)))

        u = sphere_points(2,npts=numPts)
        ## third variable taken median, u3 = 0
        u = np.vstack((u, 0*u[0,:]))
        q = dist_norm.cdf(u)
        print("Generating 2D Environmental Contour...")
        # print(p.shape)
        phy = self.get_invcdf(q)
        if dist_zeta:
            zeta = np.array([float(dist.inv(q[i,:])) for i, dist in enumerate(dist_zeta)])
        return (phy, zeta) if dist_zeta else phy

    def get_tp(self, uw, hs):
        """
        Given a grid selection of (uw,hs), calculate the corresponding median Tp 
        """
        p3 = []
        tp = []
        uw_hs_grid = np.array(list(itertools.product(uw,hs)))
        invalid = []
        for i, val in enumerate(uw_hs_grid):
            cdists,_ = self.get_cdists(val, key='phy')
            tp_ = float(cdists[-1].inv(0.5))
            if np.isnan(tp_):
                print('Invalid sea states: Uw = {:.2f}, Hs = {:.2f}'.format(val[0], val[1]))
                invalid.append(i)
            tp.append(tp_)

        tp = np.array(tp).reshape(-1,1)
        val = np.column_stack((uw_hs_grid,tp)).T
        ## remove invalid sea state sets 
        ind = list(set(np.arange(uw_hs_grid.shape[0])).difference(set(invalid)))
        val = val[:, np.array(ind, dtype=np.int32)]
        return val, invalid
    def cal_p3(self, var, beta):
        """
        var in normal space
    
        """
        if not np.all(var[-1,:]==0):
            print('Warning: The last variable is not taken the distribution median ')
        
        p3_norm = np.sqrt(beta**2 - nla.norm(var, axis=0)**2)
        dist_norm = cp.Normal()
        p3 = 1 - dist_norm.cdf(p3_norm)
        return p3


            
def main():
    dist_norm = cp.Normal()
    # q1 = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 0.1,0.3,0.5,0.7,0.9,1-1e-2,1-1e-3,1-1e-4,1-1e-5, 1-1e-6])
    # q2 = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 0.1,0.3,0.5,0.7,0.9,1-1e-2,1-1e-3,1-1e-4,1-1e-5, 1-1e-6])
    # q = np.array([q1,q2])
    test = environment('Norway5_3')
    # v = test.getEC2D(numPts=3600,isSave=True,filename='test.csv')
    # v = test.getEC2D(numPts=3600)
    # v = test.get_invcdf(q)
    # print("2D: ", v.shape)
    # v = test.phy2norm(v,isSave=False)
    # v = test.phy2norm(v)
    # print("2D: ", dist_norm.cdf(v))
    uw = np.arange(3,26) ## Hub-height wind speed
    # uw = np.array([3]) ## Hub-height wind speed
    u10 = uw/(133.5/10)**0.1
    hs = np.arange(0.5, 14, 0.5)
    # hs = np.array([5])
    # cdists,_ = test.get_cdists(np.array([2.31512,5]), key='phy')
    # print(dist_norm.inv(cdists[0].cdf(2.31512)))
    # print(cdists[1].cdf(5))
    # print(dist_norm.inv(cdists[1].cdf(5)))
    # print(cdists[2])
    var_phy,invalid = test.get_tp(u10,hs)
    # uw_hs = np.column_stack((uw_hs_grid, tp)).T
    var_norm,_ = test.phy2norm(var_phy)
    var_phy[0,:] = var_phy[0,:]*(133.5/10)**0.1
    np.savetxt('Grid_2D.txt', var_phy.T, fmt='%.4e')
    np.savetxt('Grid_2D_norm.txt', var_norm.T, fmt='%.4e')
    # print(var_norm.shape)
    # print(np.all(var_norm[2,:] == 0))
    # p3 = test.cal_p3(var_norm, 4.58)
if __name__=='__main__':
    main()
