# -*- coding: utf-8 -*-

import uqra, unittest,warnings,os, sys, math
from tqdm import tqdm
import time
import numpy as np, scipy as sp 

sys.stdout  = uqra.utilities.classes.Logger()

def cdf_chebyshev(x):
    """
    x in [-1,1]
    """
    return np.arcsin(x)/np.pi  + 0.5

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_experimentBase(self):
        doe = uqra.experiment._experimentbase.ExperimentBase()
        print(doe.ndim)
        print(doe.samplingfrom)

    def test_RandomDesign(self):
        print('Testing: Random Monte Carlo...')
        print('testing: d=1, theta: default')
        doe = uqra.RandomDesign(sp.stats.norm, 'MCS')
        doe_x = doe.samples(n_samples=1e5)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

        print('testing: d=2, theta: default[0,1], same distribution')
        doe = uqra.RandomDesign([sp.stats.norm,]*2, 'MCS')
        doe_x = doe.samples(n_samples=1e5)
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))


        print('testing: d=1, theta: default[[0,1]]')
        doe = uqra.RandomDesign(sp.stats.norm, 'MCS')
        doe_x = doe.samples(n_samples=1e5, theta=[[0,1]])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))


        print('testing: d=2, theta: default[[0,1]]')
        doe = uqra.RandomDesign([sp.stats.norm,]*2, 'MCS')
        doe_x = doe.samples(n_samples=1e5, theta=[[0,1]])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

        print('testing: d=2, theta: default[[0,1], [2,3]]')
        doe = uqra.RandomDesign([sp.stats.norm,]*2, 'MCS')
        doe_x = doe.samples(n_samples=1e5, theta=[[0,1], [2,3]])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))

    def test_LatinHyperCube(self):
        print('Testing: Latin Hypercube...')
        # doe = uqra.LHS([sp.stats.norm(0,1),]*2)
        doe = uqra.LHS([sp.stats.uniform(-1,2),]*2)
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
            orth_poly = uqra.Hermite(d=ndim,deg=p)
            n_cand    = int(1e5)
            u_samples = data_set[0:ndim, :n_cand]
            design_matrix = orth_poly.vandermonde(u_samples)
            # n_budget  = 10 * design_matrix.shape[1]
            n_budget  = 2048 
            # n_budget  =  int(np.exp2(math.ceil(np.log2(design_matrix.shape[1]))))
            
            start    = time.time()
            doe      = uqra.OptimalDesign('D', curr_set=curr_set)
            doe_index= doe.samples(design_matrix, n_samples=n_budget, orth_basis=True)
            print(doe_index)
            done     = time.time()
            print('   >> OED-{:s} (n={:d}) time elapsed: {}'.format('S', n_cand, done - start))
            np.save('DoE_McsE6R0_d2_p{:d}_D.npy'.format(p), doe_index)

    def test_CLS(self):

        # d = 2
        # print('Testing: Random Sampling from Pluripotential Equilibrium ...')
        # print('testing: d={:d}, theta: default'.format(d))
        # for i in range(1,6):
            # try:
                # doe = uqra.RandomDesign([sp.stats.uniform,]*d, 'CLS{:d}'.format(i))
                # doe_x = doe.get_samples(n_samples=1e5)
                # print('CLS{:d}'.format(i))
                # print('x shape: {}'.format(doe_x.shape))
                # print('x mean: {}'.format(np.mean(doe_x, axis=1)))
                # print('x std : {}'.format(np.std(doe_x, axis=1)))
                # print('x min : {}'.format(np.min(doe_x, axis=1)))
                # print('x max : {}'.format(np.max(doe_x, axis=1)))
            # except NotImplementedError:
                # pass

        ndim = 2
        doe_method = 'CLS4'
        print('{:s}, d={:d}'.format(doe_method, ndim))
        doe = uqra.RandomDesign([sp.stats.uniform,]*ndim, doe_method)
        for r in range(10):
            doe_x = doe.get_samples(n_samples=1e7)
            np.save('DoE_{:s}E7d{:d}R{:d}.npy'.format(doe_method.capitalize(), ndim, r), doe_x)


        
        # print('Testing: Random Sampling from Pluripotential Equilibrium ...')
        # print('testing: d=2, theta: default')
        # doe = uqra.RandomDesign( [sp.stats.uniform,] * 2, 'CLS')
        # doe_x = doe.samples(n_samples=1e5)
        # print(doe_x.shape)
        # print(np.mean(doe_x, axis=1))
        # print(np.std(doe_x, axis=1))
        # print(np.min(doe_x, axis=1))
        # print(np.max(doe_x, axis=1))

        # print('Testing: Random Sampling from Pluripotential Equilibrium ...')
        # ndim = 1
        # print('testing: d={:d}, theta: default'.format(ndim))
        # doe = uqra.RandomDesign( [sp.stats.norm,] *ndim, 'CLS')
        # doe_x = doe.get_samples(n_samples=1e5)
        # print(doe_x.shape)
        # print(np.mean(doe_x, axis=1))
        # print(np.std(doe_x, axis=1))
        # print(np.min(doe_x, axis=1))
        # print(np.max(doe_x, axis=1))
        # # np.save('cls_norm_d2', doe_x)


        # print('\nTesting: Random Sampling from Pluripotential Equilibrium ...')
        # ndim = 3
        # print('testing: d={:d}, theta: default'.format(ndim))
        # doe = uqra.RandomDesign( [sp.stats.norm,] *ndim, 'CLS')
        # doe_x = doe.get_samples(n_samples=1e7)
        # print(doe_x.shape)
        # print(np.mean(doe_x, axis=1))
        # print(np.std(doe_x, axis=1))
        # print(np.min(doe_x, axis=1))
        # print(np.max(doe_x, axis=1))
        # # np.save('cls_norm_d2', doe_x)

        # data = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/CLS/Norm/DoE_ClsE6d3R0.npy')
        # print(data.shape)
        # print(np.mean(data, axis=1))
        # print(np.std(data, axis=1))
        # print(np.min(data, axis=1))
        # print(np.max(data, axis=1))


        # print('\nTesting: Random Sampling from Pluripotential Equilibrium ...')
        # ndim = 4
        # print('testing: d={:d}, theta: default'.format(ndim))
        # for i in range(10):
            # doe = uqra.RandomDesign( [sp.stats.norm,] * ndim, 'CLS')
            # doe_x = doe.get_samples(n_samples=1e7)
            # np.save('DoE_ClsE6d{:d}R{:d}.npy'.format(ndim,i), doe_x)
        # print(doe_x.shape)
        # print(np.mean(doe_x, axis=1))
        # print(np.std(doe_x, axis=1))
        # print(np.min(doe_x, axis=1))
        # print(np.max(doe_x, axis=1))



    def test_Soptimality(slef):
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
                np.random.seed(100)
                orth_poly = uqra.Legendre(d=ndim,deg=p)
                # orth_poly = uqra.Hermite(d=ndim,deg=p, hem_type='physicists')
                doe     = uqra.OptimalDesign('S', selected_index=[3284,])
                X       = orth_poly.vandermonde(x_cand)
                idx     = doe.get_samples(X, n=math.ceil(1.2 * orth_poly.num_basis), orth_basis=True)
                print('adding:')
                print(idx)
                print('current:')
                print(doe.selected_index)
                x_samples = x_cand[:,idx]
                X_train = orth_poly.vandermonde(x_samples)
                _, s, _ = np.linalg.svd(X_train)
                ## condition number, kappa = max(svd)/min(svd)
                kappa = max(abs(s)) / min(abs(s)) 
                mean_kappa.append(kappa)
                print('mean condition number: {}'.format(np.mean(mean_kappa)))


    def test_gauss_quadrature(self):
        """
        https://keisan.casio.com/exec/system/1329114617
        """

        print('========================TESTING: 1D GAUSS QUADRATURE=======================')
        dists2test = [cp.Uniform(-1,1), cp.Normal(), cp.Gamma(1,1), cp.Beta(1,1)]
        rules2test = ['leg', 'hem', 'lag', 'jacobi']
        order2test = [2,3,4,5,6,7,8]
        for idist2test, irule2test in zip(dists2test, rules2test):
            print('-'*50)
            print('>>> Gauss Quadrature with polynominal: {}'.format(const.DOE_RULE_FULL_NAMES[irule2test.lower()]))
            uqra.blockPrint()
            quad_doe = uqra.DoE('QUAD', irule2test, order2test, idist2test)
            uqra_samples = quad_doe.get_samples()
            # quad_doe.disp()
            uqra.enablePrint()
            if irule2test == 'hem':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d_e, weight1d_e = np.polynomial.hermite_e.hermegauss(iorder)
                    print('{:<15s}: {}'.format('probabilist', np.around(coord1d_e,2)))
                    coord1d, weight1d = np.polynomial.hermite.hermgauss(iorder)
                    print('{:<15s}: {}'.format('physicist', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('uqra', np.around(np.squeeze(uqra_samples[i][:-1,:]),2)))

            elif irule2test == 'leg':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d, weight1d = np.polynomial.legendre.leggauss(iorder)
                    print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('uqra', np.around(np.squeeze(uqra_samples[i][:-1,:]),2)))
            elif irule2test == 'lag':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d, weight1d = np.polynomial.laguerre.laggauss(iorder)
                    print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('uqra', np.around(np.squeeze(uqra_samples[i][:-1,:]),2)))
            elif irule2test == 'jacobi':
                print('NOT TESTED YET')


if __name__ == '__main__':
    unittest.main()
