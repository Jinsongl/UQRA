import chaospy as cp
import numpy as np
import numpy.random as rn
import numpy.linalg as la
import csv
import os
from genVar import *
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

from chaospy.dist.baseclass import Dist
import chaospy.dist

""" Define Environment random variables.
Reference: 
        Norway 5:
        Li L, Gao Z, Moan T. Joint environmental data at five european offshore sites for design of combined wind and wave
        energy concepts. 32nd International Conference on Ocean, Offshore, and Arctic Engineering, Nantes, France, Paper
        No. OMAE2013-10156, 2013.
"""

########################################################
# Joint Distribution for Wind & Wave Dist(Uw, Hs, Tp)
########################################################

def dist_Uw(isTrunc=False,bnd=[3,25]):
# Marginal distribution of 10-meter wind speed
    a_u, b_u = 2.029, 9.409
    if isTrunc:
        dist = cp.Truncweibull(bnd[0],bnd[1],a_u, b_u)
    else:
        dist = cp.Weibull(a_u, b_u)
    return dist

# class dist_Uw(Dist):
    # def __init__(self,isTrunc,bnd):
        # Dist.__init__(self)
        # self.isTrunc = isTrunc
        # self.bnd = bnd
        # self.a_u = 2.029
        # self.b_u = 9.409
        # self.truncdist = cp.Truncweibull(self.bnd[0], self.bnd[1], self.a_u, self.b_u)
        # self.dist = cp.Weibull(self.a_u, self.b_u)
    # def _pdf(self, x):
        # return self.truncdist.pdf(x) if self.isTrunc else self.dist.pdf(x)
    # def _ppf(self, q):
        # return self.truncdist.inv(q) if self.isTrunc else self.dist.inv(q)
    # def _cdf(self, x):
        # return self.truncdist.cdf(x) if self.isTrunc else self.dist.cdf(x)
    # def _bnd(self):
        # return self.range()

def distC_Hs(*var):
# Hs distribution conditional on Uw
    a1, a2, a3 = 2.136, 0.013, 1.709
    b1, b2, b3 = 1.816, 0.024, 1.787
    Uw = var[0]
    a_h = a1 + a2 * Uw ** a3
    b_h = b1 + b2 * Uw ** b3
    dist = cp.Weibull(a_h, b_h)
    
    return dist

# class distC_Hs(Dist):
    # def __init__(self,u):
        # Dist.__init__(self)
        # self.Uw = u
        # self.a1 = 2.136
        # self.a2 = 0.013
        # self.a3 = 1.709
        # self.b1 = 1.816
        # self.b2 = 0.024
        # self.b3 = 1.787
        # a_h = self.a1 + self.a2 * self.Uw ** self.a3
        # b_h = self.b1 + self.b2 * self.Uw ** self.b3
        # self.dist = cp.Weibull(a_h, b_h)

    # def _pdf(self,x):
        # return self.dist.pdf(x) 
    # def _ppf(self, q):
        # return self.dist.ppf(q)
    # def _cdf(self, x):
        # return self.dist.cdf(x)
    # def _bnd(self):
        # return self.dist.range()

def distC_Tp(*var):
    """
    Conditional distribution of Tp given var
    var: list of one or two values
    len(var) == 1, Tp|Hs
    len(var) == 2, Tp|(Uw,Hs)

    """
    if len(var) == 1: 
        c1, c2, c3 = 1.886, 0.365, 0.312
        d1, d2, d3 = 0.001, 0.105, -0.264
        h = var[0][0]
        mu_LTC = c1 + c2 * h ** c3

        sigma_LTC = (d1 + d2 * np.exp(d3 * h))** 0.5

        dist = cp.Lognormal(mu_LTC, sigma_LTC)
        return dist
    elif len(var) == 2:
        theta, gamma = -0.255, 1.0
        e1, e2, e3 = 8.0, 1.938, 0.486
        f1, f2, f3 = 2.5, 3.001, 0.745
        k1, k2, k3 = -0.001, 0.316, -0.145 

        Uw,h = var 
        
        Tp_bar = e1 + e2 * h**e3 
        u_bar = f1 + f2 * h**f3
        niu_Tp = k1 + k2 * np.exp(h*k3)

        mu_Tp = Tp_bar * (1 + theta * ((Uw - u_bar)/u_bar)**gamma)

        mu_lnTp = np.log(mu_Tp / (np.sqrt(1 + niu_Tp**2)))
        sigma_lnTp = np.sqrt(np.log(niu_Tp**2 + 1))

        dist = cp.Lognormal(mu_lnTp, sigma_lnTp)
        
        return dist
    else:
        raise NotImplementedError("Conditional Tp distribution defined only for one or two conditional variables")

# def distC2_Tp(Uw,h):
# # Tp distribution conditional on Uw, Hs

    # theta, gamma = -0.255, 1.0
    # e1, e2, e3 = 8.0, 1.938, 0.486
    # f1, f2, f3 = 2.5, 3.001, 0.745
    # k1, k2, k3 = -0.001, 0.316, -0.145 
    
    # Tp_bar = e1 + e2 * h**e3 
    # u_bar = f1 + f2 * h**f3
    # niu_Tp = k1 + k2 * np.exp(h*k3)

    # mu_Tp = Tp_bar * (1 + theta * ((Uw - u_bar)/u_bar)**gamma)

    # mu_lnTp = np.log(mu_Tp / (np.sqrt(1 + niu_Tp**2)))
    # sigma_lnTp = np.sqrt(np.log(niu_Tp**2 + 1))

    # dist = cp.Lognormal(mu_lnTp, sigma_lnTp)
    
    # return dist
# class distC2_Tp(Dist):
    # def __init__(self, u, h):
        # Dist.__init__(self)
        # self.Uw = u
        # self.h = h
        # self.theta = -0.255
        # self.gamma = 1.0
        # self.e1 = 8.0
        # self.e2 = 1.938
        # self.e3 = 0.486
        # self.f1 = 2.5
        # self.f2 = 3.001
        # self.f3 = 0.745
        # self.k1 = -0.001
        # self.k2 = 0.316
        # self.k3 = -0.145
        # Tp_bar = self.e1 + self.e2 * self.h**self.e3 
        # u_bar = self.f1 + self.f2 * self.h**self.f3
        # niu_Tp = self.k1 + self.k2 * np.exp(self.h*self.k3)
        # mu_Tp = Tp_bar * (1 + self.theta * ((self.Uw - u_bar)/u_bar)**self.gamma)
        # mu_lnTp = np.log(mu_Tp / (np.sqrt(1 + niu_Tp**2)))
        # sigma_lnTp = np.sqrt(np.log(niu_Tp**2 + 1))

        # self.dist = cp.Lognormal(mu_lnTp, sigma_lnTp)
    # def _pdf(self,x):
        # return self.dist.pdf(x)
    # def _cdf(self,x):
        # return self.dist.cdf(x)
    # def _ppf(self, q):
        # return self.dist.ppf(q)
    # def _bnd(self):
        # return self.dist.range()
        
########################################################
# Joint Distribution ONLY for wave Dist(Hs, Tp)
########################################################


# def dist_Hs(var):
    # h0 = 5.0
    # mu_LHM = 0.871
    # sigma_LHM = 0.506
    # alpha_HM = 1.433
    # beta_HM = 2.547
    # h = var[0]

    # dist1 = cp.Lognormal(mu_LHM, sigma_LHM)
    # dist2 = cp.Weibull(alpha_HM, beta_HM) 

    # return dist1 if h <= h0 else dist2

def dist_Hs(**kwargs):
    h0          = 5.0
    mu_LHM      = 0.871
    sigma_LHM   = 0.506
    alpha_HM    = 1.433
    beta_HM     = 2.547
    
    dist1 = cp.Lognormal(mu_LHM, sigma_LHM)
    dist2 = cp.Weibull(alpha_HM, beta_HM) 
    cdf0 = dist1.cdf(h0)
    # h       = var[0]
    if kwargs is not None:
        for key, value in kwargs.items():
            if key == "h":
                return dist1 if value <= h0 else dist2
            elif key == "cdf":
                return dist1 if value <= cdf0 else dist2 
    else:
        raise NotImplementedError("kwargs is not defined")
    # return dist1
# class dist_Hs(Dist):
    # def __init__(self):
        # Dist.__init__(self)
        # self.h0 = 5.0
        # self.mu_LHM = 0.871
        # self.sigma_LHM = 0.506
        # self.alpha_HM = 1.433
        # self.beta_HM = 2.547
        # self.dist1 = cp.Lognormal(self.mu_LHM, self.sigma_LHM)
        # self.dist2 = cp.Weibull(self.alpha_HM, self.beta_HM)
    # def _pdf(self, h):
        # return self.dist1.pdf(h) if h <= self.h0 else self.dist2.pdf(h)
    # def _cdf(self, h):
        # if h <= self.h0:
            # return self.dist1.cdf(h)
        # else:
            # return (self.dist2.cdf(h)
    # def _ppf(self, q):
        # if q < self.dist1.cdf(self.h0):
            # return self.dist1.inv(q)
        # else:
            # return self.dist2.inv(q)

        # # return self.dist1.inv(q) if q < self.dist1.cdf(self.h0) else self.dist2.inv(q)

    # def _bnd(self):
        # return 0, 1e10

# def distC1_Tp(h):
    # c1, c2, c3 = 1.886, 0.365, 0.312
    # d1, d2, d3 = 0.001, 0.105, -0.264

    # mu_LTC = c1 + c2 * h ** c3

    # sigma_LTC = (d1 + d2 * np.exp(d3 * h))** 0.5

    # dist = cp.Lognormal(mu_LTC, sigma_LTC)
    # return dist

# class distC1_Tp(Dist):
    # def __init__(self, h):
        # Dist.__init__(self)
        # self.h = h
        # self.c1 = 1.886
        # self.c2 = 0.365
        # self.c3 = 0.312
        # self.d1 = 0.001
        # self.d2 = 0.105
        # self.d3 = -0.264

        # mu_LTC = c1 + c2 * h ** c3
        # sigma_LTC = (d1 + d2 * np.exp(d3 * h))** 0.5
        # self.dist = cp.Lognormal(mu_LTC, sigma_LTC)
    # def _pdf(self,x):
        # return self.dist.pdf(x)

    # def _cdf(self,x):
        # return self.dist.cdf(x)
    # def _ppf(self,q):
        # return self.dist.ppf(q)
    # def _bnd(self):
        # return self.dist.range()




def EC2D_UH(T=50,p3=0.5,simTime=1,numPts=3600, isTrunc=False,bnd=[-1,1]):
    """    
    2D Environmental Contour, Tp taken with given fractile
    T: return period, in years
    p3: fractile for Tp, p3=0.5 for median
    simTime: simulation duration performed for each run
    numPts: number of points on contour
    """
    dist_normal = cp.Normal()
    phyrv   = []
    beta    = np.abs(dist_normal.inv(simTime/(T*365.25*24)))
    theta   = np.linspace(0,360,numPts) 
    beta    = np.sqrt(beta**2 - dist_normal.inv(1-p3)**2)
    p       = beta *np.column_stack((np.cos(theta),np.sin(theta)))
    dist1   = dist_Uw(isTrunc,bnd)
    for i in xrange(p.shape[0]):
        u1 = dist_normal.cdf(p[i][0])
        zeta1 = dist_zeta1.inv(u1)
        rv1 = dist1.inv(u1) 
        dist2 = distC_Hs(rv1)
        u2 = dist_normal.cdf(p[i][1])
        zeta2 = dist_zeta2.inv(u2)
        rv2 = dist2.inv(u2)
        dist3 = distC_Tp(rv1, rv2)
        u3 = dist_normal.cdf(p3)
        zeta3 = dist_zeta3.inv(u3)
        rv3 = dist3.inv(1-p3)
        # phyrv.append([u1, u2, u3, zeta1, zeta2, zeta3, rv1, rv2, rv3])
        phyrv.append([p[i][0],p[i][1],p3, zeta1, zeta2, zeta3, rv1, rv2, rv3])
    phyrv = np.asarray(phyrv).T
    return phyrv

def EC2D_HT(T=50,simTime=1,numPts=3600):
    """    
    2D Environmental Contour, only for Hs and Tp 
    T: return period, in years
    simTime: simulation duration performed for each run
    numPts: number of points on contour
    """
    dist_normal = cp.Normal()
    phyrv = []
    beta = np.abs(dist_normal.inv(simTime/(T*365.25*24)))
    theta = np.linspace(0,360,numPts) 
    p = beta *np.column_stack((np.cos(theta),np.sin(theta)))
    for i in xrange(p.shape[0]):
        u1 = dist_normal.cdf(p[i][0])
        dist1 = dist_Hs(cdf = u1)
        rv1 = dist1.inv(u1) 
        dist2 = distC_Tp([rv1,])
        u2 = dist_normal.cdf(p[i][1])
        rv2 = dist2.inv(u2)
        zeta1 = dist_zeta1.inv(u1)
        zeta2 = dist_zeta2.inv(u2)
        phyrv.append([p[i][0], p[i][1], zeta1, zeta2,rv1, rv2])
    phyrv = np.asarray(phyrv).T
    return phyrv
def norm2phy(zeta, *dist):
    """ 
    Mapping standard normal random variables to physical space.
    Output order depends on Rosenblatt conditional sequence.
    """
    phyrv = []
    dist_normal = cp.Normal()
    dim, nsamp = zeta.shape
    assert dim == len(dist), "Number of distribution provided is not equal to the dimension of standard normal"
    for izeta in zeta.T:
        rv = [dist[0]().inv(dist_normal.cdf(izeta[0]))]
        for i in range(1, dim):
            rv.append(dist[i](rv[:i]).inv(dist_normal.cdf(izeta[i])))
        # rv.append(dist[i](rv[:i]).inv(dist_normal.cdf(izeta[i])) for i in range(1,dim))
        phyrv.append(rv)
    phyrv = np.asarray(phyrv)
    return phyrv





def getEC2D_UH():
    print " SNL 13.2 MW Semisubmersible Offshore Wind Turbine Model" 
    print " Uhub cut in: 3 m/s, Uhub cut out: 25 m/s"
    print " Hub height = 133.5 m"
    print " Evaluating Environment Contour..."
    Uhub_in, Uhub_out = 3, 25
    Hhub = 133.5
    alpha= 0.1
    Uw_in = (Hhub/ 10.0) ** (-alpha) * Uhub_in
    Uw_out = (Hhub/ 10.0) ** (-alpha) * Uhub_out

    filename = 'Norway5EC2D_UH_'
    NEWDIR = './Data/'+filename[:-1]
    if not os.path.exists(NEWDIR):
        os.mkdir(NEWDIR)
    # plt.clf()
    # plt.figure(1)
    for _, p in enumerate(np.hstack((np.logspace(-5,-1,5), np.array([0.5])))):
        print "EC: p3 = ", "{:.1e}".format(p)
        rnv = EC2D_UH(p3=p,isTrunc=False, bnd=[Uw_in, Uw_out])
        rnv[0,:] = rnv[0,:] * (Hhub/10.0)**alpha
        filenamet = filename + '{:.0E}'.format(p) + '.csv'
        with open(filenamet, 'wb') as fileid:
            writer = csv.writer(fileid, delimiter=',')
            for v in rnv.T: 
                writer.writerow(['{:8.4e}'.format(float(x)) for x in v])
            # writer.writerows(rnv)
        os.rename(filenamet,NEWDIR+'/'+filenamet)
            
    # dist_normal = cp.Normal()
    # EC2D_UH_sel = []
    # p3=0.5
    # beta = np.abs(dist_normal.inv(1.0/(50*365.25*24)))
    # beta = np.sqrt(beta**2 - dist_normal.inv(1-p3)**2)
    # dist1 = dist_Uw(isTrunc=True,bnd=[Uw_in, Uw_out])
    # dist_normal = cp.Normal()
    # print beta
    # for rv1 in np.arange(19,25):
        # p1 = dist_normal.inv(dist1.cdf(rv1))
        # dist2 = distC_Hs(rv1)
        # print p1
        # p2 = np.sqrt(beta**2 - p1**2)
        # print p2, rv1
        # rv2 = dist2.inv(dist_normal.cdf(p2))
        # EC2D_UH_sel.append([rv1, rv2])
    # EC2D_UH_sel = np.array(EC2D_UH_sel).T
    # EC2D_UH_sel[0,:] = EC2D_UH_sel[0,:] * (Uhub/10.0) **alpha
    # filename = 'EC2D_UH_sel.csv'
    # with open(filename, 'wb') as fileid:
        # writer = csv.writer(fileid, delimiter=',')
        # for v in EC2D_UH_sel.T: 
            # writer.writerow(['{:8.4e}'.format(float(x)) for x in v])
        # # writer.writerows(rnv)
    # os.rename(filename,NEWDIR+'/'+filename)
    
        # plt.plot(rnv[:,0],rnv[:,1],label= 'p3='+str("{:.0e}".format(p)))
    # plt.title("Environment Contour")
    # plt.xlabel("Hub height wind speed, $U_{hub} (m/s)$")
    # plt.ylabel("Significant wave height, $H_s (m)$")
    # plt.xlim(2,26)
    # plt.legend(loc=0)
    # plt.savefig('Figures/EC2D_UHNorway5.eps')
    ## plt.clf()
# fig = plt.figure(1)
# plt.contourf(x,y,likelihood,20)
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x, y, likelihood, cmap=cm.jet)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.grid()
# plt.show()
    # plt.show()
def getEC2D_HT(T=50):
    print " Site: Norway 5  "
    print " Evaluating " + str(T)+ " year Environment Contour..."
    filename = 'Norway5EC2D_HT_'
    NEWDIR = './Data/'+filename[:-1]
    if not os.path.exists(NEWDIR):
        os.mkdir(NEWDIR)
    rnv = EC2D_HT(T=T)
    filenamet = filename + '{:}'.format(T) + '.csv'
    with open(filenamet, 'wb') as fileid:
        writer = csv.writer(fileid, delimiter=',')
        for v in rnv.T: 
            writer.writerow(['{:8.4e}'.format(float(x)) for x in v])
        # writer.writerows(rnv)
    os.rename(filenamet,NEWDIR+'/'+filenamet)
            
def main():
    getEC2D_HT()
    # getEC2D_UH()
if __name__=="__main__":
    main()
