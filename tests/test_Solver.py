# -*- coding: utf-8 -*-

import museuq, unittest,warnings,os, sys 
from tqdm import tqdm
import numpy as np, scipy as sp 
from museuq.solver.PowerSpectrum import PowerSpectrum
from museuq.environment.Kvitebjorn import Kvitebjorn as Kvitebjorn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pickle
import scipy.stats as stats

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/museuq/examples/JupyterNotebook'
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_sparse_poly(self):
        print('========================TESTING: sparse poly =======================')
        ndim = 1
        deg  = 4
        poly = museuq.Legendre(d=ndim, deg=deg)
        coef = [0,0,0,1,0]

        solver = museuq.sparse_poly(poly, sparsity=4, coef=coef)
        # x = np.random.normal(size=(ndim, 1000))
        x = np.arange(10)
        y = solver.run(x)
        Leg2 = lambda x: 0.5*(3*x**2 - 1)/(poly.basis_norms[2])**0.5
        Leg3 = lambda x: 0.5*(5*x**3 - 3*x)/(poly.basis_norms[3])**0.5


        assert solver.ndim  == ndim
        assert solver.deg   == deg
        assert solver.coef  == coef
        assert np.array_equal(y,Leg3(x))
        u = np.random.uniform(0,1,size=(2,100))
        x = solver.map_domain(u, [stats.uniform(0,1),]*solver.ndim)
        print(np.max(x))
        print(np.min(x))

    def test_bench4(self):
        print('========================TESTING: BENCH 4 =======================')
        data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/BENCH4/Data'
        model_name  = 'BENCH4'

        # ### grid points
        # x           = np.linspace(-10,20,600).reshape((1,-1))
        # solver      = museuq.Solver(model_name, x)
        # y           = solver.run()
        # res         = np.concatenate((x,y), axis=0)
        # np.save(os.path.join(data_dir,model_name.lower()), res)

        ### data from files
        for r in range(10):
            filename = 'DoE_McRE6R{:d}.npy'.format(r)
            data_set = np.load(os.path.join(data_dir, filename))
            zeta     = data_set[0,:].reshape(1,-1)
            x        = data_set[1,:].reshape(1,-1)
            solver   = museuq.Solver(model_name, x)
            y        = solver.run()
            np.save(os.path.join(data_dir,'DoE_McRE6R{:d}_y_None.npy'.format(r)), y)

    def test_linear_oscillator(self):
        random_seed = 100
        out_responses = [1,2]
        nsim = 3
        out_stats = ['absmax']
        m=1
        c=0.1/np.pi
        k=1.0/np.pi/np.pi
        m,c,k  = [stats.norm(m, 0.05*m), stats.norm(c, 0.2*c), stats.norm(k, 0.1*k)]
        # env    = museuq.Environment([stats.uniform, stats.norm])
        env    = museuq.Environment([2,])
        # env    = Kvitebjorn()
        solver = museuq.linear_oscillator(m=m,c=c,k=k,excitation='spec_test1', environment=env, t=1000, t_transit=10,
                out_responses=out_responses, out_stats=out_stats)
        samples= solver.generate_samples(3)
        print(samples)
        y = solver.run(samples, return_all=True, random_seed=random_seed)
        print(y)
        np.save('y', np.array(y))


        # for r in range(2):
            # # filename = r'DoE_McsE6R{:d}.npy'.format(r)
            # # data_dir = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Uniform/'
            # # u        = np.load(os.path.join(data_dir, filename))[:solver.ndim,:]
            # # x        = solver.map_domain(u, [stats.uniform(-1,2),] * solver.ndim) 
            # # print(np.mean(u, axis=1))
            # # print(np.std(u, axis=1))
            # # print(np.mean(x, axis=1))
            # # print(np.std(x, axis=1))
            # y_QoI = solver.run(samples, random_seed=random_seed) 
            # print(np.array(y_QoI).shape)

    def test_four_branch(self):
        np.random.seed(100)
        solver  = museuq.four_branch_system()
        data_dir_src    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Normal/'
        data_dir_destn  = r'/Volumes/External/MUSE_UQ_DATA/Four_branch_system/Data'

        for r in tqdm(range(10), ascii=True, desc='     -'):
            filename = 'DoE_McsE6R{:d}.npy'.format(r)
            u = np.load(os.path.join(data_dir_src, filename))[:solver.ndim, :]
            x = solver.map_domain(u, [stats.norm(0,1),] * solver.ndim) 
            if not np.array_equal(u, x):
                print(np.max(abs(u-x), axis=1))
            y = solver.run(x).reshape(1,-1)
            data = np.vstack((u,x,y))
            np.save(os.path.join(data_dir_destn, filename), data)
            
    def test_franke(self):
        np.random.seed(100)
        solver  = museuq.franke()
        data_dir_src    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Uniform/'
        data_dir_destn  = r'/Volumes/External/MUSE_UQ_DATA/Franke/Data'

        for r in tqdm(range(10), ascii=True, desc='     -'):
            filename = 'DoE_McsE6R{:d}.npy'.format(r)
            u = np.load(os.path.join(data_dir_src, filename))[:solver.ndim, :]
            x = solver.map_domain(u, [stats.uniform(-1,2),] * solver.ndim) 
            if not np.array_equal(u, x):
                print(np.max(abs(u-x), axis=1))
            y = solver.run(x).reshape(1,-1)
            data = np.vstack((u,x,y))
            np.save(os.path.join(data_dir_destn, filename), data)

    def test_duffing(self):

        # f = lambda t: 8 * np.cos(0.5 * t)
        np.random.seed(100)
        dt = 0.01 
        out_responses = [1,2]
        nsim = 1
        out_stats = ['mean', 'std', 'skewness', 'kurtosis', 'absmax', 'absmin']
        # solver = museuq.duffing_oscillator(m=1,c=0.2*np.pi,k=4*np.pi**2,s=np.pi**2, out_responses=out_responses, out_stats=out_stats, tmax=18000, dt=dt,y0=[1,0])
        f = lambda t: 0.39 * np.cos(1.4 * t)
        solver = museuq.duffing_oscillator(m=1,c=0.1,k=-1,s=1,excitation=f, out_responses=out_responses, out_stats=out_stats, tmax=18000, dt=dt,y0=[0,0])
        x = solver.generate_samples(1)
        print(solver)
        print(x)
        y = solver.run(x,return_raw=True)

        # data_dir_src    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Kvitebjorn/Normal/'
        # data_dir_destn  = r'/Volumes/External/MUSE_UQ_DATA/Duffing/Data/' 
        # for r in range(1):
            # data = []
            # filename = 'DoE_McsE6R{:d}.npy'.format(r)
            # x = np.load(os.path.join(data_dir_src, filename))[:solver.ndim, :]
            # # x = solver.map_domain(u, [stats.norm(0,1),] * solver.ndim) 
            # # y_raw, y_QoI = zip(*[solver.run(x.T) for _ in range(nsim)]) 
            # y_raw, y_QoI = solver.run(x.T)
            # # np.save('duffing_time_series_{:d}'.format(r), y_raw)
            # filename = 'DoE_McsE6R{:d}_stats'.format(r)
            # np.save(os.path.join(data_dir_destn, filename), y_QoI)

    def test_samples_same(self):
        for r in range(10):
            filename = r'DoE_McsE6R{:d}.npy'.format(r)
            print(filename)
            data_dir = r'/Volumes/External/MUSE_UQ_DATA/Four_branch_system/Data/'
            data1    = np.load(os.path.join(data_dir, filename))[:2,:]
            data_dir = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Normal/'
            data2    = np.load(os.path.join(data_dir, filename))[:2,:]
            print(np.array_equal(data1, data2))
    
        # x = np.arange(30).reshape(3,10)
        # solver = museuq.Ishigami()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30)
        # solver = museuq.xsinx()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30)
        # solver = museuq.poly4th()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30).reshape(2,15)
        # solver = museuq.polynomial_square_root_function()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30).reshape(2,15)
        # solver = museuq.four_branch_system()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30).reshape(2,15)
        # solver = museuq.polynomial_product_function()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        ### General Solver run testing 
        # print('========================TESTING: Solver =======================')

        # model_name  = 'linear_oscillator'
        # kwargs  = {
            # 'time_max'  : 100,
            # 'dt'        : 0.2,
                # }
        # tmax,dt = 1000, 0.1
        # t       = np.arange(0,tmax, dt)

        # zeta    = 0.01
        # omega_n = 2 # rad/s
        # m       = 1 
        # k       = (omega_n/2/np.pi) **2 * m 
        # c       = zeta * 2 * np.sqrt(m * k)
        # mck     = (m,c,k)
        # solver  = museuq.Solver(model_name, x)
        # y       = solver.run(**kwargs) 

        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # np.save(os.path.join(data_dir,'Kvitebjørn_EC_P{:d}_{:d}'.format(P, nsim)), EC_y)

        # ## run solver for EC cases
        # P, nsim     = 10, 25
        # data_dir    = '/Users/jinsongliu/BoxSync/MUSELab/museuq/museuq/environment'
        # data_set    = np.load(os.path.join(data_dir, 'Kvitebjørn_EC_P{:d}.npy'.format(P)))
        # EC_x        = data_set[2:,:]
        # model_name  = 'linear_oscillator'
        # solver      = museuq.Solver(model_name, EC_x)
        # EC_y        = np.array([solver.run(doe_method = 'EC') for _ in range(nsim)])

        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # np.save(os.path.join(data_dir,'Kvitebjørn_EC_P{:d}_{:d}'.format(P, nsim)), EC_y)

        # ## run solver for Hs Tp grid points
        # nsim        = 30
        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # filename    = 'HsTp_grid118.npy'
        # data_set    = np.load(os.path.join(data_dir, filename))
        # model_name  = 'linear_oscillator'
        # solver      = museuq.Solver(model_name, data_set)
        # grid_out    = np.array([solver.run(doe_method = 'GRID') for _ in range(nsim)])
        # np.save(os.path.join(data_dir,'HsTp_grid118_out'), grid_out)

        # data_set  = np.load('DoE_McRE3R0.npy')
        # x_samples = data_set[2:,:]

        # model_name  = 'linear_oscillator'
        # solver      = museuq.Solver(model_name, x_samples)
        # kwargs      = {'doe_method': 'MCS'}
        # samples_y   = solver.run(**kwargs )
        # np.save('test_linear_oscillator_y', samples_y)
        # # filename_tags = ['R0']
        # # filename_tags = [itag+'_y' for itag in filename_tags]
        # # museuq_dataio.save_data(samples_y, 'test_linear_oscillator', os.getcwd(), filename_tags)
        # samples_y_stats = solver.get_stats()
        # np.save('test_linear_oscillator_y_stats', samples_y_stats)
        # # filename_tags = [itag+'_y_stats' for itag in filename_tags]
        # # museuq_dataio.save_data(samples_y_stats, 'test_linear_oscillator', os.getcwd(), filename_tags)


if __name__ == '__main__':
    unittest.main()
