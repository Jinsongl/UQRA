# -*- coding: utf-8 -*-

import museuq, unittest,warnings,os, sys, math
from tqdm import tqdm
import time
import numpy as np, chaospy as cp, scipy as sp 

sys.stdout  = museuq.utilities.classes.Logger()

def cdf_chebyshev(x):
    """
    x in [-1,1]
    """
    return np.arcsin(x)/np.pi  + 0.5

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_experimentBase(self):
        doe = museuq.experiment._experimentbase.ExperimentBase()
        print(doe.ndim)
        print(doe.samplingfrom)

    def test_RandomDesign(self):
        print('Testing: Random Monte Carlo...')
        print('testing: d=1, theta: default')
        doe = museuq.RandomDesign(sp.stats.norm, 'MCS')
        doe_x = doe.samples(n_samples=1e5)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

        print('testing: d=2, theta: default[0,1], same distribution')
        doe = museuq.RandomDesign([sp.stats.norm,]*2, 'MCS')
        doe_x = doe.samples(n_samples=1e5)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))


        print('testing: d=1, theta: default[[0,1]]')
        doe = museuq.RandomDesign(sp.stats.norm, 'MCS')
        doe_x = doe.samples(n_samples=1e5, theta=[[0,1]])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))


        print('testing: d=2, theta: default[[0,1]]')
        doe = museuq.RandomDesign([sp.stats.norm,]*2, 'MCS')
        doe_x = doe.samples(n_samples=1e5, theta=[[0,1]])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

        print('testing: d=2, theta: default[[0,1], [2,3]]')
        doe = museuq.RandomDesign([sp.stats.norm,]*2, 'MCS')
        doe_x = doe.samples(n_samples=1e5, theta=[[0,1], [2,3]])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

    def test_LatinHyperCube(self):
        print('Testing: Latin Hypercube...')
        # doe = museuq.LHS([sp.stats.norm(0,1),]*2)
        doe = museuq.LHS([sp.stats.uniform(-1,2),]*2)
        doe_u, doe_x = doe.samples(256)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))
        print(np.min(doe_x, axis=1))
        print(np.max(doe_x, axis=1))

        np.save('DoE_Lhs_d2_Uniform256.npy', doe_x)

    def test_Doptimality(self):
        """
        Optimal Design
        """
        # data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Uniform'
        data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Normal'
        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(data_dir, filename))

        np.random.seed(100)
        ndim= 2
        p   = np.array([2])

        curr_set = []
        for p in np.array([10]):
            orth_poly = museuq.Hermite(d=ndim,deg=p)
            n_cand    = int(1e5)
            u_samples = data_set[0:ndim, :n_cand]
            design_matrix = orth_poly.vandermonde(u_samples)
            # n_budget  = 10 * design_matrix.shape[1]
            n_budget  = 2048 
            # n_budget  =  int(np.exp2(math.ceil(np.log2(design_matrix.shape[1]))))
            
            start    = time.time()
            doe      = museuq.OptimalDesign('D', curr_set=curr_set)
            doe_index= doe.samples(design_matrix, n_samples=n_budget, orth_basis=True)
            print(doe_index)
            done     = time.time()
            print('   >> OED-{:s} (n={:d}) time elapsed: {}'.format('S', n_cand, done - start))
            np.save('DoE_McsE6R0_d2_p{:d}_D.npy'.format(p), doe_index)

    def test_CLS(self):
        print('Testing: Random Sampling from Pluripotential Equilibrium ...')
        print('testing: d=1, theta: default')
        doe = museuq.RandomDesign(sp.stats.uniform, 'CLS')
        doe_x = doe.samples(n_samples=1e5)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))
        print(np.min(doe_x, axis=1))
        print(np.max(doe_x, axis=1))


        print('Testing: Random Sampling from Pluripotential Equilibrium ...')
        print('testing: d=2, theta: default')
        doe = museuq.RandomDesign( [sp.stats.uniform,] * 2, 'CLS')
        doe_x = doe.samples(n_samples=1e5)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))
        print(np.min(doe_x, axis=1))
        print(np.max(doe_x, axis=1))

        print('Testing: Random Sampling from Pluripotential Equilibrium ...')
        print('testing: d=2, theta: default')
        doe = museuq.RandomDesign( [sp.stats.norm,], 'CLS')
        doe_x = doe.samples(n_samples=1e6)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))
        print(np.min(doe_x, axis=1))
        print(np.max(doe_x, axis=1))


        print('Testing: Random Sampling from Pluripotential Equilibrium ...')
        print('testing: d=2, theta: default')
        for i in range(10):
            doe = museuq.RandomDesign( [sp.stats.norm,] * 10, 'CLS')
            doe_x = doe.samples(n_samples=1e6)
            np.save('DoE_McsE6R{:d}.npy'.format(i), doe_x)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))
        print(np.min(doe_x, axis=1))
        print(np.max(doe_x, axis=1))


    def test_Soptimality(slef):
        np.random.seed(100)
        x = np.linspace(-1,1,1000)
        y = cdf_chebyshev(x)
        ndim    = 2
        n_cand  = int(1e5)

        data_dir= r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Uniform'
        filename= r'DoE_McsE6R0.npy'
        mcs_data_set  = np.load(os.path.join(data_dir, filename))
        x_cand  = mcs_data_set[:ndim,:n_cand].reshape(ndim, -1)

        for i, p in enumerate([5, ]):
            mean_kappa = []
            for _ in range(1):
                np.random.seed()
                orth_poly = museuq.Legendre(d=ndim,deg=p)
                doe     = museuq.OptimalDesign('S', curr_set=[3284,])
                X       = orth_poly.vandermonde(x_cand)
                idx     = doe.samples(X, n_samples=math.ceil(1.2 * orth_poly.num_basis), orth_basis=True)
                print('adding:')
                print(idx)
                print('current:')
                print(doe.curr_set)
                x_samples = x_cand[:,idx]
                X_train = orth_poly.vandermonde(x_samples)
                _, s, _ = np.linalg.svd(X_train)
                ## condition number, kappa = max(svd)/min(svd)
                kappa = max(abs(s)) / min(abs(s)) 
                mean_kappa.append(kappa)
                print('mean condition number: {}'.format(np.mean(mean_kappa)))


    # def test_gauss_quadrature(self):
        # """
        # https://keisan.casio.com/exec/system/1329114617
        # """

        # print('========================TESTING: 1D GAUSS QUADRATURE=======================')
        # dists2test = [cp.Uniform(-1,1), cp.Normal(), cp.Gamma(1,1), cp.Beta(1,1)]
        # rules2test = ['leg', 'hem', 'lag', 'jacobi']
        # order2test = [2,3,4,5,6,7,8]
        # for idist2test, irule2test in zip(dists2test, rules2test):
            # print('-'*50)
            # print('>>> Gauss Quadrature with polynominal: {}'.format(const.DOE_RULE_FULL_NAMES[irule2test.lower()]))
            # museuq.blockPrint()
            # quad_doe = museuq.DoE('QUAD', irule2test, order2test, idist2test)
            # museuq_samples = quad_doe.get_samples()
            # # quad_doe.disp()
            # museuq.enablePrint()
            # if irule2test == 'hem':
                # for i, iorder in enumerate(order2test):
                    # print('>>> order : {}'.format(iorder))
                    # coord1d_e, weight1d_e = np.polynomial.hermite_e.hermegauss(iorder)
                    # print('{:<15s}: {}'.format('probabilist', np.around(coord1d_e,2)))
                    # coord1d, weight1d = np.polynomial.hermite.hermgauss(iorder)
                    # print('{:<15s}: {}'.format('physicist', np.around(coord1d,2)))
                    # print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))

            # elif irule2test == 'leg':
                # for i, iorder in enumerate(order2test):
                    # print('>>> order : {}'.format(iorder))
                    # coord1d, weight1d = np.polynomial.legendre.leggauss(iorder)
                    # print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                    # print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))
            # elif irule2test == 'lag':
                # for i, iorder in enumerate(order2test):
                    # print('>>> order : {}'.format(iorder))
                    # coord1d, weight1d = np.polynomial.laguerre.laggauss(iorder)
                    # print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                    # print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))
            # elif irule2test == 'jacobi':
                # print('NOT TESTED YET')


        # print('Compared results here: https://keisan.casio.com/exec/system/1329114617')


if __name__ == '__main__':
    unittest.main()
