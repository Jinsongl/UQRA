# -*- coding: utf-8 -*-

import museuq, unittest,warnings,os, sys
from tqdm import tqdm
import numpy as np, chaospy as cp, scipy as sp 

sys.stdout  = museuq.utilities.classes.Logger()

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_experimentBase(self):
        doe = museuq.experiment._experimentbase.ExperimentBase()
        print(doe.ndim)
        print(doe.distributions)


    def test_RandomDesign(self):
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
        doe = museuq.LHS(distributions=[sp.stats.norm,]*2)
        doe_u, doe_x = doe.samples(2000, theta=[[0,1],[2,3]])
        print(doe_x.shape)
        print(np.mean(doe_x, axis=1))
        print(np.std(doe_x, axis=1))


        # doe = museuq.LHS(n_samples=1e3,dist_names=['uniform', 'norm'],ndim=2,dist_theta=[(-1, 2*2), (2,1)])
        # doe.samples()
        # print(np.mean(doe.x, axis=1))
        # print(np.std(doe.x, axis=1))

    def test_OptimalDesign(self):
        """
        Optimal Design
        """
        ### Ishigami function
        # data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/Ishigami/Data'
        ### SDOF system
        data_dir = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # data_dir = 'E:\Run_MUSEUQ'
        np.random.seed(100)
        # dist_x = cp.Normal()
        dist_u= cp.Iid(cp.Normal(),2)
        u_samples = dist_u.sample(100)
        basis = cp.orth_ttr(10,dist_u)
        X = basis(*u_samples).T
        doe = museuq.OptimalDesign('D', n_samples=10)
        doe_index = doe.samples(X, is_orth=True)
        doe_index = doe.adaptive(X, n_samples=10)
        print(doe_index)
        doe_index = doe.adaptive(X, n_samples=10)
        print(doe_index)


        ### 2D
        # quad_orders = range(4,11)
        # alpha = [1.0, 1.1, 1.3, 1.5, 2.0,2.5, 3.0,3.5, 5]
        # dist_u= cp.Iid(cp.Normal(),2)
        # for iquad_orders in quad_orders:
            # basis = cp.orth_ttr(iquad_orders-1,dist_u)
            # for r in range(10):
                # filename  = 'DoE_McsE6R{:d}_stats.npy'.format(r)
                # data_set  = np.load(os.path.join(data_dir, filename))
                # samples_y = np.squeeze(data_set[:,4,:]).T
                
                # filename  = 'DoE_McsE6R{:d}.npy'.format(r)
                # data_set  = np.load(os.path.join(data_dir, filename))
                # samples_u = data_set[0:2, :]
                # samples_x = data_set[2:4, :]
                # # samples_y = data_set[6  , :].reshape(1,-1)
                # print('Quadrature Order: {:d}'.format(iquad_orders))
                # print('Candidate samples filename: {:s}'.format(filename))
                # print('   >> Candidate sample set shape: {}'.format(samples_u.shape))
                # design_matrix = basis(*samples_u).T
                # print('   >> Candidate Design matrix shape: {}'.format(design_matrix.shape))
                # for ia in alpha:
                    # print('   >> Oversampling rate : {:.2f}'.format(ia))
                    # doe_size = min(int(len(basis)*ia), 10000)
                    # doe = museuq.OptimalDesign('S', n_samples = doe_size )
                    # doe.samples(design_matrix, u=samples_u, is_orth=True)
                    # data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
                    # filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_p{:d}_OptS{:d}'.format(r,iquad_orders,doe_size))
                    # np.save(filename, data)

                # for ia in alpha:
                    # print('   >> Oversampling rate : {:.2f}'.format(ia))
                    # doe_size = min(int(len(basis)*ia), 10000)
                    # doe = museuq.OptimalDesign('D', n_samples = doe_size )
                    # doe.samples(design_matrix, u=samples_u, is_orth=True)
                    # data = np.concatenate((doe.I.reshape(1,-1),doe.u,samples_x[:,doe.I], samples_y[:,doe.I]), axis=0)
                    # filename = os.path.join(data_dir, 'DoE_McsE6R{:d}_p{:d}_OptD{:d}'.format(r,iquad_orders,doe_size))
                    # np.save(filename, data)
                    
                    

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
            museuq.blockPrint()
            quad_doe = museuq.DoE('QUAD', irule2test, order2test, idist2test)
            museuq_samples = quad_doe.get_samples()
            # quad_doe.disp()
            museuq.enablePrint()
            if irule2test == 'hem':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d_e, weight1d_e = np.polynomial.hermite_e.hermegauss(iorder)
                    print('{:<15s}: {}'.format('probabilist', np.around(coord1d_e,2)))
                    coord1d, weight1d = np.polynomial.hermite.hermgauss(iorder)
                    print('{:<15s}: {}'.format('physicist', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))

            elif irule2test == 'leg':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d, weight1d = np.polynomial.legendre.leggauss(iorder)
                    print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))
            elif irule2test == 'lag':
                for i, iorder in enumerate(order2test):
                    print('>>> order : {}'.format(iorder))
                    coord1d, weight1d = np.polynomial.laguerre.laggauss(iorder)
                    print('{:<15s}: {}'.format('numpy ', np.around(coord1d,2)))
                    print('{:<15s}: {}'.format('museuq', np.around(np.squeeze(museuq_samples[i][:-1,:]),2)))
            elif irule2test == 'jacobi':
                print('NOT TESTED YET')


        print('Compared results here: https://keisan.casio.com/exec/system/1329114617')


if __name__ == '__main__':
    unittest.main()
