# -*- coding: utf-8 -*-

import museuq, unittest,warnings,os, sys
from tqdm import tqdm
import time
import numpy as np, chaospy as cp, scipy as sp 

sys.stdout  = museuq.utilities.classes.Logger()

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
        doe = museuq.LHS([sp.stats.uniform,]*2)
        doe_u, doe_x = doe.samples(2000, theta=[[-1,2],[-1,2]])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))
        print(np.min(doe_x, axis=1))
        print(np.max(doe_x, axis=1))

    def test_OptimalDesign(self):
        """
        Optimal Design
        """
        data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Uniform'
        filename = 'DoE_McsE6R0.npy'
        data_set = np.load(os.path.join(data_dir, filename))

        np.random.seed(100)
        ndim= 2
        p   = np.array([20])
        orth_poly = museuq.Legendre(d=ndim,deg=p)


        n_cand    = int(1e5)
        u_samples = data_set[0:ndim, :n_cand]
        design_matrix = orth_poly.vandermonde(u_samples)
        n_budget = 10 * design_matrix.shape[1]

        start    = time.time()
        doe      = museuq.OptimalDesign('S')
        doe_index= doe.samples(design_matrix, n_samples=n_budget, is_orth=True)
        done     = time.time()
        print('   >> OED-{:s} (n={:d}) time elapsed: {}'.format('S', n_cand, done - start))


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
