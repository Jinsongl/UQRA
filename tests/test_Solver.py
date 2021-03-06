# -*- coding: utf-8 -*-

import uqra, unittest,warnings,os, sys 
from tqdm import tqdm
import numpy as np, scipy as sp 
from uqra.solver.PowerSpectrum import PowerSpectrum
from uqra.environment.Kvitebjorn import Kvitebjorn as Kvitebjorn
import uqra.utilities.helpers as uqhelper
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pickle
import scipy.stats as stats
import scipy.io
import multiprocessing as mp

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/uqra/examples/JupyterNotebook'
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_sparse_poly(self):
        print('========================TESTING: sparse poly =======================')
        ndim = 1
        deg  = 4
        poly = uqra.Legendre(d=ndim, deg=deg)
        coef = [0,0,0,1,0]

        solver = uqra.sparse_poly(poly, sparsity=4, coef=coef)
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
        # solver      = uqra.Solver(model_name, x)
        # y           = solver.run()
        # res         = np.concatenate((x,y), axis=0)
        # np.save(os.path.join(data_dir,model_name.lower()), res)

        ### data from files
        for r in range(10):
            filename = 'DoE_McRE6R{:d}.npy'.format(r)
            data_set = np.load(os.path.join(data_dir, filename))
            zeta     = data_set[0,:].reshape(1,-1)
            x        = data_set[1,:].reshape(1,-1)
            solver   = uqra.Solver(model_name, x)
            y        = solver.run()
            np.save(os.path.join(data_dir,'DoE_McRE6R{:d}_y_None.npy'.format(r)), y)

    def test_linear_oscillator(self):
        random_seed = 100
        np.random.seed(random_seed)
        seeds_st = np.random.randint(0, int(2**31-1), size=20)
        out_responses = [2]
        out_stats = ['absmax']
        m=1
        c=0.1/np.pi
        k=1.0/np.pi/np.pi
        m,c,k  = [stats.norm(m, 0.05*m), stats.norm(c, 0.2*c), stats.norm(k, 0.1*k)]
        # env    = uqra.Environment([stats.uniform, stats.norm])
        env    = uqra.Environment([2,])
        # env    = Kvitebjorn()
        solver = uqra.linear_oscillator(m=m,c=c,k=k,excitation='spec_test1', environment=env, t=1000, t_transit=10,
                out_responses=out_responses, out_stats=out_stats)
        samples= solver.generate_samples(100)
        y = solver.run(samples, seeds_st=seeds_st[:5] )

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
        print(y.shape)

    def test_surge(self):
        random_seed = 100
        out_responses = [2,3]
        out_stats = ['absmax', 'mean']
        m=1e8
        k=280000
        c=0.05*2*np.sqrt(k*m)
        ltf = np.load('/Users/jinsongliu/BoxSync/MUSELab/uqra/uqra/solver/FPSO_ltf.npy')
        qtf = np.load('/Users/jinsongliu/BoxSync/MUSELab/uqra/uqra/solver/FPSO_qtf.npy')
        rao_= np.load('/Users/jinsongliu/BoxSync/MUSELab/uqra/uqra/solver/FPSO_RAO.npy')
        print(ltf.shape)
        print(qtf.shape)
        print(rao_.shape)
        # m,c,k  = [stats.norm(m, 0.05*m), stats.norm(c, 0.2*c), stats.norm(k, 0.1*k)]
        # env    = uqra.Environment([stats.uniform, stats.norm])
        # env    = uqra.Environment([2,])
        env    = Kvitebjorn()
        solver = uqra.surge_model(m=m,c=c,k=k, environment=env, t=4000, t_transit=100, dt=0.1, ltf=ltf[:2],
                out_responses=out_responses, out_stats=out_stats)
        samples= solver.generate_samples(10)
        y = solver.run(samples)

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
        print(y.shape)

    def test_four_branch(self):
        np.random.seed(100)
        solver  = uqra.four_branch_system()
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
        solver  = uqra.franke()
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
        # solver = uqra.duffing_oscillator(m=1,c=0.2*np.pi,k=4*np.pi**2,s=np.pi**2, out_responses=out_responses, out_stats=out_stats, tmax=18000, dt=dt,y0=[1,0])
        f = lambda t: 0.39 * np.cos(1.4 * t)
        solver = uqra.duffing_oscillator(m=1,c=0.1,k=-1,s=1,excitation=f, out_responses=out_responses, out_stats=out_stats, tmax=18000, dt=dt,y0=[0,0])
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

    def test_FPSO(self):
        Kvitebjorn      = uqra.environment.Kvitebjorn()
        data_dir_samples= r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples'
        data_dir_result = os.path.join(r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/', solver.nickname)
        # data_dir_samples= r'/home/jinsong/Documents/MUSE_UQ_DATA/Samples'
        # data_dir_result = r'/home/jinsong/Documents/MUSE_UQ_DATA/FPSO_SDOF'

        # ------------------------ Basic Check ----------------- ###
        # solver = uqra.FPSO()
        # x = np.array([2,4]).reshape(2,-1)
        # y = solver.run(x) 
        # print('Hs = {}, Tp={}'.format(np.around(x[0]), np.around(x[1])))

        ## ------------------------ LHS ----------------- ###
        # n_initial = 20
        # solver    = uqra.FPSO(phase=np.arange(20))
        # Kvitebjorn= uqra.environment.Kvitebjorn()
        # doe   = uqra.LHS([stats.norm(),] * solver.ndim)
        # u_lhs = doe.samples(size=n_initial, loc=0, scale=1, random_state=100)
        # x_lhs = Kvitebjorn.ppf(stats.norm.cdf(u_lhs)) 
        # y_lhs = solver.run(x_lhs)
        # print(y_lhs.shape)
        # data_lhs = np.concatenate((u_lhs, x_lhs, y_lhs), axis=0)
        # np.save(os.path.join(data_dir_result, '{:s}_DoE_Lhs.npy'), data_lhs)
        ## ------------------------ MCS  ----------------- ###
        # MCS for DoE_McsE7R0

        
        n = int(1e7)
        for s in range(10):
            solver = uqra.FPSO(random_state = s)
            data_mcs_u = np.load(os.path.join(data_dir_samples, 'MCS', 'Norm', 'DoE_McsE7R{:d}.npy'.format(s)))
            data_mcs_u = data_mcs_u[:solver.ndim, :n]
            data_mcs_x = Kvitebjorn.ppf(stats.norm.cdf(data_mcs_u))
            y = solver.run(data_mcs_x, verbose=True) 
            data = np.concatenate((data_mcs_u, data_mcs_x, y.reshape(1,-1)))
            np.save(os.path.join(data_dir_result, '{:s}_McsE7R{:d}.npy'.format(solver.nickname,s)), data)

        # ------------------------ Environmental Contour ----------------- ###
        # solver = uqra.FPSO(random_state = np.arange(20))
        # data_ec = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Kvitebjorn/Kvitebjorn_EC_50yr.npy')
        # EC_u, EC_x = data_ec[:2], data_ec[2:]
        # EC_y = solver.run(EC_x, verbose=True) 
        # EC2D_median = np.median(EC_y, axis=0)
        # EC2D_data = np.concatenate((EC_u,EC_x,EC2D_median.reshape(1,-1)), axis=0)
        # y50_EC_idx = np.argmax(EC2D_median)
        # y50_EC     = EC2D_data[:,y50_EC_idx]
        # print('Extreme reponse from EC:')
        # print('   {}'.format(y50_EC))
        # np.save(os.path.join(data_dir_result, '{:s}_Kvitebjorn_EC2D_50yr.npy'.format(solver.nickname)  ), EC2D_data)
        # np.save(os.path.join(data_dir_result, '{:s}_Kvitebjorn_EC2D_50yr_y.npy'.format(solver.nickname)), EC_y)


        ## ------------------------ Environmental Contour ----------------- ###
        # solver  = uqra.FPSO(phase=np.arange(21))
        # dataset = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/FPSO_SDOF/Data/FPSO_Test_McsE7R0.npy')
        # u, x    = dataset[:2], dataset[2:4]
        # y       = solver.run(x, verbose=True) 
        # try:
            # data    = np.concatenate((u,x,y), axis=0)
        # except ValueError:
            # data    = np.concatenate((u,x,y.reshape(1,-1)), axis=0)

        # np.save(os.path.join(data_dir_result, 'FPSO_Test_McsE7R0.npy'  ), data)

        ## ------------------------ Environmental Contour Bootstrap ----------------- ###
        # print('------------------------------------------------------------')
        # print('>>> Environmental Contour for Model: FPSO                   ')
        # print('------------------------------------------------------------')
        # filename    = 'FPSO_DoE_EC2D_T50_y.npy' 
        # EC2D_data_y = np.load(os.path.join(data_dir_result, filename))[short_term_seeds_applied,:] 
        # filename    = 'FPSO_DoE_EC2D_T50.npy' 
        # EC2D_data_ux= np.load(os.path.join(data_dir_result, filename))[:4,:]

        # EC2D_median = np.median(EC2D_data_y, axis=0)
        # EC2D_data   = np.concatenate((EC2D_data_ux,EC2D_median.reshape(1,-1)), axis=0)
        # y50_EC      = EC2D_data[:,np.argmax(EC2D_median)]

        # print(' > Extreme reponse from EC:')
        # print('   - {:<25s} : {}'.format('EC data set', EC2D_data_y.shape))
        # print('   - {:<25s} : {}'.format('y0', np.array(y50_EC[-1])))
        # print('   - {:<25s} : {}'.format('Design state (u,x)', y50_EC[:4]))
        # np.random.seed(100)
        # EC2D_y_boots      = uqra.bootstrapping(EC2D_data_y, 100) 
        # EC2D_boots_median = np.median(EC2D_y_boots, axis=1)
        # y50_EC_boots_idx  = np.argmax(EC2D_boots_median, axis=-1)
        # y50_EC_boots_ux   = np.array([EC2D_data_ux[:,i] for i in y50_EC_boots_idx]).T
        # y50_EC_boots_y    = np.max(EC2D_boots_median,axis=-1) 
        # y50_EC_boots      = np.concatenate((y50_EC_boots_ux, y50_EC_boots_y.reshape(1,-1)), axis=0)
        # y50_EC_boots_mean = np.mean(y50_EC_boots, axis=1)
        # y50_EC_boots_std  = np.std(y50_EC_boots, axis=1)
        # print(' > Extreme reponse from EC (Bootstrap (n={:d})):'.format(EC2D_y_boots.shape[0]))
        # print('   - {:<25s} : {}'.format('Bootstrap data set', EC2D_y_boots.shape))
        # print('   - {:<25s} : [{:.2f}, {:.2f}]'.format('y50[mean, std]',y50_EC_boots_mean[-1], y50_EC_boots_std[-1]))
        # print('   - {:<25s} : {}'.format('Design state (u,x)', y50_EC_boots_mean[:4]))

        # u_center = y50_EC_boots_mean[ :2].reshape(-1, 1)
        # x_center = y50_EC_boots_mean[2:4].reshape(-1, 1)
        # print(' > Important Region based on EC(boots):')
        # print('   - {:<25s} : {}'.format('Radius', radius_surrogate))
        # print('   - {:<25s} : {}'.format('Center U', np.squeeze(u_center)))
        # print('   - {:<25s} : {}'.format('Center X', np.squeeze(x_center)))
        # print('================================================================================')

        ## ------------------------ Validation Dataset with shifted center ----------------- ###

        # random_seed_short_term = np.arange(21)
        # solver = uqra.FPSO(phase=random_seed_short_term)
        # data  = np.load(os.path.join(data_dir_samples, 'MCS', 'Norm', 'DoE_McsE7R0.npy' )) 
        # data  = data[:solver.ndim, np.linalg.norm(data[:2], axis=0)<radius_surrogate]
        # mcs_u = data[:solver.ndim,:int(1e5)]

        # # data  = np.load('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/CLS/DoE_Cls2E7d2R0.npy') 
        # # mcs_u = data[:solver.ndim,:int(1e5)] *  radius_surrogate

        # mcs_u = mcs_u + u_center
        # mcs_x = Kvitebjorn.ppf(stats.norm.cdf(mcs_u))
        # print('--------------------------------------------------')
        # print('>>> Running MCS ')
        # print('--------------------------------------------------')
        # print('  - u samples {}: mean [{}], std [{}] '.format(mcs_u.shape, np.around(np.mean(mcs_u, axis=1), 2), np.around(np.std(mcs_u, axis=1), 2)))
        # print('  - x samples {}: mean [{}], std [{}] '.format(mcs_x.shape, np.around(np.mean(mcs_x, axis=1), 2), np.around(np.std(mcs_x, axis=1), 2)))
        # mcs_y = solver.run(mcs_x, verbose=True) 
        # print(mcs_y.shape)
        # mcs_data = np.concatenate((mcs_u, mcs_x, mcs_y.reshape(len(random_seed_short_term),-1)), axis=0)
        # print(mcs_data.shape)

        # np.save(os.path.join(data_dir_result, 'FPSO_DoE_McsE5R0.npy'), mcs_data)
        # np.save(os.path.join(data_dir_result, 'FPSO_DoE_Cls2E5R0.npy'), mcs_data)


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
        # solver = uqra.Ishigami()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30)
        # solver = uqra.xsinx()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30)
        # solver = uqra.poly4th()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30).reshape(2,15)
        # solver = uqra.polynomial_square_root_function()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30).reshape(2,15)
        # solver = uqra.four_branch_system()
        # solver.run(x)
        # print(solver)
        # print(solver.y.shape)

        # x = np.arange(30).reshape(2,15)
        # solver = uqra.polynomial_product_function()
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
        # solver  = uqra.Solver(model_name, x)
        # y       = solver.run(**kwargs) 

        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # np.save(os.path.join(data_dir,'Kvitebjørn_EC_P{:d}_{:d}'.format(P, nsim)), EC_y)

        # ## run solver for EC cases
        # P, nsim     = 10, 25
        # data_dir    = '/Users/jinsongliu/BoxSync/MUSELab/uqra/uqra/environment'
        # data_set    = np.load(os.path.join(data_dir, 'Kvitebjørn_EC_P{:d}.npy'.format(P)))
        # EC_x        = data_set[2:,:]
        # model_name  = 'linear_oscillator'
        # solver      = uqra.Solver(model_name, EC_x)
        # EC_y        = np.array([solver.run(doe_method = 'EC') for _ in range(nsim)])

        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # np.save(os.path.join(data_dir,'Kvitebjørn_EC_P{:d}_{:d}'.format(P, nsim)), EC_y)

        # ## run solver for Hs Tp grid points
        # nsim        = 30
        # data_dir    = '/Users/jinsongliu/External/MUSE_UQ_DATA/linear_oscillator/Data'
        # filename    = 'HsTp_grid118.npy'
        # data_set    = np.load(os.path.join(data_dir, filename))
        # model_name  = 'linear_oscillator'
        # solver      = uqra.Solver(model_name, data_set)
        # grid_out    = np.array([solver.run(doe_method = 'GRID') for _ in range(nsim)])
        # np.save(os.path.join(data_dir,'HsTp_grid118_out'), grid_out)

        # data_set  = np.load('DoE_McRE3R0.npy')
        # x_samples = data_set[2:,:]

        # model_name  = 'linear_oscillator'
        # solver      = uqra.Solver(model_name, x_samples)
        # kwargs      = {'doe_method': 'MCS'}
        # samples_y   = solver.run(**kwargs )
        # np.save('test_linear_oscillator_y', samples_y)
        # # filename_tags = ['R0']
        # # filename_tags = [itag+'_y' for itag in filename_tags]
        # # uqra_dataio.save_data(samples_y, 'test_linear_oscillator', os.getcwd(), filename_tags)
        # samples_y_stats = solver.get_stats()
        # np.save('test_linear_oscillator_y_stats', samples_y_stats)
        # # filename_tags = [itag+'_y_stats' for itag in filename_tags]
        # # uqra_dataio.save_data(samples_y_stats, 'test_linear_oscillator', os.getcwd(), filename_tags)


if __name__ == '__main__':
    unittest.main()
