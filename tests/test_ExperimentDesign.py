# -*- coding: utf-8 -*-

import uqra, unittest,warnings,os, sys, math
from tqdm import tqdm
import time, random
import numpy as np, scipy as sp 
import scipy.stats as stats
import scipy
import pickle
import copy
sys.stdout  = uqra.utilities.classes.Logger()
from statsmodels.distributions.empirical_distribution import ECDF
class Data(): pass
def cdf_chebyshev(x):
    """
    x in [-1,1]
    """
    return np.arcsin(x)/np.pi  + 0.5

def rejection_sampling(f, g, M, n):
    """
    rejection sampling from proposal distribution g to get samples from f
    Arguments:
        f: target distribution
        g: proposal distribution
        M: f <= M*g for all x. i.e. M = sup(f/g)
        n: int, number of samples
    """

    pass



class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    np.set_printoptions(2)
    def test_experimentBase(self):
        doe = uqra.experiment._experimentbase.ExperimentBase()
        print(doe.ndim)
        print(doe.samplingfrom)

    def test_MCS(self):
        print('Testing: Random Monte Carlo...')
        print('testing: d=1, theta: default')
        doe = uqra.MCS(sp.stats.norm)
        doe_x = doe.samples(size=1e5)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

        print('testing: d=2, theta: default[0,1], same distribution')
        doe = uqra.MCS([sp.stats.norm,]*2)
        doe_x = doe.samples(size=1e5)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))


        print('testing: d=1, theta: default[[0,1]]')
        doe = uqra.MCS(sp.stats.norm)
        doe_x = doe.samples(size=1e5, loc=[0], scale=[1,])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))


        print('testing: d=2, theta: default[[0,1]]')
        doe = uqra.MCS([sp.stats.norm,]*2)
        doe_x = doe.samples(size=1e5, loc=[0,], scale=[1,])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

        print('testing: d=2, theta: default[[0,1], [2,3]]')
        doe = uqra.MCS([sp.stats.norm,]*2)
        doe_x = doe.samples(size=1e5, loc=[0,1],scale=[2,3])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

    def test_LHS(self):
        print('Testing: Latin Hypercube...')
        data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/LHS'
        ndim = 2
        # u_dist = [stats.uniform(0,17.5), stats.uniform(0, 34.5)]
        # doe = uqra.LHS([sp.stats.norm(0,1),]*2)
        doe = uqra.LHS([sp.stats.uniform(-1,2),]*ndim)
        # doe = uqra.LHS(u_dist, criterion='maxmin')
        for n in range(10,1500):
            print(' number of sampels {:d}'.format(n))
            doe_x = np.array([doe.samples(size=n) for _ in range(50)])
            filename = 'DoE_Lhs{:d}_2uniform.npy'.format(n)
            np.save(os.path.join(data_dir, filename), doe_x)

    def test_OptimalDesign(self):
        A = np.random.rand(100,10)
        B = np.random.rand(100,10)
        DoE = uqra.OptimalDesign(A)
        s0 = DoE._cal_svalue_over(B, A)
        s1 = DoE._update_S_Optimality(A,B)
        s2 = DoE._update_D_Optimality(A,B)
        print(s1,'\n', s2)
        print(np.array_equal(s0,s1))

        A = np.random.rand(2,10)
        B = np.random.rand(100,10)
        DoE = uqra.OptimalDesign(A)
        s1 = DoE._update_S_Optimality_TSM( A, B)
        s0 = DoE._cal_svalue_under(B, A)
        # s2 = DoE._update_D_Optimality(A,B)
        print(np.array_equal(s0,s1))
        # print(s1, s2)


    def test_OptD(self):
        """
        Test D-Optimality 
        """
        print('Testing D Optimality with greedy method')
        ndim, p = 2, 4
        poly_name = 'heme'
        nsamples= 100
        np.random.seed(100)
        x = stats.norm(0,1).rvs(size=(ndim, nsamples))
        orth_poly = uqra.poly.orthogonal(ndim, p, poly_name)
        X = orth_poly.vandermonde(x)
        q,r,p = scipy.linalg.qr(X.T, pivoting=True)

        print(p)
        optimal_samples = list(p[:X.shape[1]])
        candidate_samples = self._list_diff(list(np.arange(0,nsamples)), optimal_samples) 
        self._check_complement(optimal_samples, candidate_samples,list(np.arange(0, nsamples)))
        while len(candidate_samples) != 0:
            X0 = X[optimal_samples]
            X1 = X[candidate_samples]
            Ds = []
            for x1 in X1:
                X2 = np.concatenate((X0, x1.reshape(1,-1)), axis=0)
                Ds.append(np.linalg.det(X2.T.dot(X2)))
            idx = candidate_samples[np.argmax(Ds)]
            # print(idx)
            assert np.ndim(idx) == 0
            optimal_samples.append(idx)
            candidate_samples = self._list_diff(list(np.arange(0,nsamples)), optimal_samples) 
        print(optimal_samples)

        doe = uqra.OptimalDesign(X)
        idx = doe.samples('D', 99, initialization='RRQR')
        print(idx)

    def test_OptS(self):
        pass
        
    def test_CLS(self):
        np.set_printoptions(precision=4)
        np.set_printoptions(threshold=8)
        np.set_printoptions(suppress=True)
        np.random.seed(None)
        print(' Testing CLS1, Chebyshev ECDF plot')
        ### multi dimensional CLS1 just tensor product of 1d, so just need to check against 1d chebyshev
        ### generate Chebyshev samples, if x ~ U[0, pi], then cos(X) ~ chebyshev
        ndim= 1
        n   = int(1e6)
        print(' Checking CLS1: ndim={}, n={}'.format(ndim, n))
        x0 = np.cos(stats.uniform(0, np.pi).rvs(size=n))
        x1 = uqra.CLS('CLS1',ndim).samples(size=n)
        print(' - Domain checking: ')
        print('     - {:<10s}: [{}, {}]'.format('Expected', np.amin(x0), np.amax(x0)))
        print('     - {:<10s}: [{}, {}]'.format('UQRA', np.amin(x1), np.amax(x1)))
        print(' - Statistics checking: ')
        print('     - {:<10s}: [{}, {}]'.format('Expected', np.mean(x0), np.std(x0)))
        print('     - {:<10s}: [{}, {}]'.format('UQRA', np.mean(x1), np.std(x1)))
        print(' - Run UQRA_TEst.ipynb to check cdf/pdf plots ')
        filename = 'DoE_CLS1E6D1.npy'
        np.save(os.path.join('Data', filename), np.array([x0.reshape(ndim, -1),x1]))



        print(' Testing CLS4 ')
        ndim, n = 1, int(1E6)
        print(' Checking CLS4: ndim={}, n={}'.format(ndim, n))
        ### d=1, v(x) = pi * sqrt(2-x**2)**1/2
        ## perform rejection sampling to get samples from v(x)
        ## proposal distribution N(0,1)
        ## M = sup(v/g)
        M  = np.sqrt(2/np.pi * 2.72)
        n0 = math.ceil(M*2) * n
        u  = stats.uniform(0,1).rvs(size=n0) 
        y  = stats.norm(0,1).rvs(size=n0)
        y  = y[np.where(abs(y) <= np.sqrt(2))]
        u  = u[np.where(abs(y) <= np.sqrt(2))]
        vy = 1.0/np.pi*np.sqrt(2-y**2)
        gy = stats.norm(0,1).pdf(y)
        x0 = y[np.where(u < vy/(M*gy))][:n]
        x1 = uqra.CLS('CLS4',ndim).samples(size=n)
        print(x0.shape)
        print(x1.shape)
        print(' - Domain checking: ')
        print('     - {:<10s}: [{}, {}]'.format('Expected', np.amin(x0), np.amax(x0)))
        print('     - {:<10s}: [{}, {}]'.format('UQRA', np.amin(x1), np.amax(x1)))
        print(' - Statistics checking [mu, std]: ')
        print('     - {:<10s}: [{}, {}]'.format('Expected ', np.mean(x0), np.std(x0)))
        print('     - {:<10s}: [{}, {}]'.format('UQRA ', np.mean(x1), np.std(x1)))
        filename = 'DoE_CLS4E6D1.npy'
        np.save(os.path.join('Data', filename), np.array([x0.reshape(ndim, -1),x1]))

        ndim, n = 2, int(1E6)
        print(' Checking CLS4: ndim={}, n={}'.format(ndim, n))
        ### d=1, v(x) = 2*pi * (2-x**2)
        ## perform rejection sampling to get samples from v(x)
        ## proposal distribution N(0,1)
        ## M = sup(v/g)
        # M  = np.sqrt(2/np.pi)
        # n0 = math.ceil(M*2) * n
        # u  = stats.uniform(0,1).rvs(size=n0) 
        # y  = stats.norm(0,1).rvs(size=n0)
        # y  = y[np.where(abs(y) <= np.sqrt(2))]
        # u  = u[np.where(abs(y) <= np.sqrt(2))]
        # vy = 1.0/np.pi*np.sqrt(2-y**2)
        # gy = stats.norm(0,1).pdf(y)
        # x0 = y[np.where(u < vy/(M*gy))][:n]
        x1 = uqra.CLS('CLS4',ndim).samples(size=n)
        x0 = x1 
        print(x0.shape)
        print(x1.shape)
        print(' - Domain checking: ')
        print('     - {:<10s}: [{}, {}]'.format('Expected', np.amin(x0), np.amax(x0)))
        print('     - {:<10s}: [{}, {}]'.format('UQRA', np.amin(x1), np.amax(x1)))
        print(' - Statistics checking [mu, std]: ')
        print('     - {:<10s}: [{}, {}]'.format('Expected ', np.mean(x0), np.std(x0)))
        print('     - {:<10s}: [{}, {}]'.format('UQRA ', np.mean(x1), np.std(x1)))
        filename = 'DoE_CLS4E6D{:d}.npy'.format(ndim)
        np.save(os.path.join('Data', filename), np.array([x0.reshape(ndim, -1),x1]))
    def test_gauss_quadrature(self):
        pass

    def _list_union(self, ls1, ls2):
        """
        append ls2 to ls1 and check if there exist duplicates
        return the union of two lists and remove duplicates
        """
        ls = list(copy.deepcopy(ls1)) + list(copy.deepcopy(ls2))
        if len(ls) != len(set(ls1).union(set(ls2))):
            raise ValueError('Duplicate elements found in list when append to each other')
        return ls

    def _list_diff(self, ls1, ls2):
        """
        returns a list that is the difference between two list, elements present in ls1 but not in ls2
        """
        ls1 = list(copy.deepcopy(ls1))
        ls2 = list(copy.deepcopy(ls2))
        for element in ls2:
            try:
                ls1.remove(element)
            except ValueError:
                pass
        return ls1

    def _list_inter(self, ls1, ls2):
        """
        return common elements between ls1 and ls2 
        """
        ls = list(set(ls1).intersection(set(ls2)))
        return ls

    def _check_complement(self, A, B, U=None):
        """
        check if A.union(B) = U and A.intersection(B) = 0
        """
        A = set(A)
        B = set(B)
        U = set(np.arange(self.X.shape[0])) if U is None else set(U)
        if A.union(B) != U:
            raise ValueError(' Union of sets A and B are not the universe U')
        if len(A.intersection(B)) != 0:
            raise ValueError(' Sets A and B have common elements: {}'.format(A.intersection(B)))
        return True

if __name__ == '__main__':
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    unittest.main()
