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

def run_sparse_poly():
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

def run_bench4():
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

def run_linear_oscillator():
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

def run_surge():
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

def run_four_branch():
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
        
def run_franke():
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

def run_duffing():

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

def run_FPSO():
    # uqra_env = uqra.environment.Kvitebjorn()
    uqra_env = uqra.environment.Norway5(ndim=2)
    current_os  = sys.platform
    if current_os.upper()[:3] == 'WIN':
        data_dir        = os.path.join('G:\\','My Drive','MUSE_UQ_DATA', 'UQRA_Examples')
        data_dir_doe    = os.path.join('G:\\','My Drive','MUSE_UQ_DATA', 'ExperimentalDesign')
        data_dir_random = os.path.join('G:\\','My Drive','MUSE_UQ_DATA', 'ExperimentalDesign', 'Random')
    elif current_os.upper() == 'DARWIN':
        data_dir        = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples'
        data_dir_doe    = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign'
        data_dir_random = r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random'
    elif current_os.upper() == 'LINUX':
        data_dir        = r'/home/jinsong/Documents/MUSE_UQ_DATA/UQRA_Examples'
        data_dir_doe    = r'/home/jinsong/Documents/MUSE_UQ_DATA/ExperimentalDesign'
        data_dir_random = r'/home/jinsong/Documents/MUSE_UQ_DATA/ExperimentalDesign/Random'
    else:
        raise ValueError('Operating system {} not found'.format(current_os))    

    # data_dir_samples= r'/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/FPSO_SRUGE/TestData'
    data_dir_result = os.path.join(data_dir, 'FPSO_SURGE', 'TestData')
    # ------------------------ MCS ----------------- ###
    for r in range(1):
        data = uqra.Data()
        dist_u = stats.uniform(0,1)
        data.y = []
        for itheta in range(1):
            solver = uqra.FPSO(random_state=itheta, distributions=uqra_env)
            data.u = dist_u.rvs(size=(solver.ndim, int(1e7)))
            data.x = solver.map_domain(data.u, dist_u)
            data.y.append(solver.run(data.x))
        if len(data.y) == 1:
            data.y = data.y[0]

        del data.x
        filename = '{:s}_McsE7R{:d}{:d}.npy'.format(solver.nickname, r, itheta)
        print('saving data: {}'.format(os.path.join(data_dir_result, filename)))
        np.save(os.path.join(data_dir_result, filename), data, allow_pickle=True)


        # filename   = 'DoE_McsE6R{:d}_norm.npy'.format(r)
        # data_mcs_u = np.load(os.path.join(data_dir_random, filename))[:uqra_env.ndim,:]
        # data.u     = data_mcs_u 
        # data.xi    = data_mcs_u*np.sqrt(0.5)

        # data_mcs_x = uqra_env.ppf(stats.norm().cdf(data_mcs_u))
        # filename   = '{:s}_McsE6R{:d}_norm.npy'.format(uqra_env.site.capitalize(), r)
        # data.x     = data_mcs_x 

        # data.y = []
        # for itheta in np.arange(5):
            # solver = uqra.FPSO(random_state=itheta)
            # data_mcs_x = uqra_env.ppf(stats.norm.cdf(data_mcs_u))
            # y = solver.run(data_mcs_x, verbose=True) 
            # data.y.append(y)
        # filename = '{:s}_McsE6R{:d}{:d}.npy'.format(solver.nickname,r, itheta)
        # print('saving data: {}'.format(os.path.join(data_dir_result, filename)))
        # np.save(os.path.join(data_dir_result, filename), data, allow_pickle=True)
        # for itheta in np.arange(10,20):
            # solver = uqra.FPSO(random_state=itheta)
            # data_mcs_x = uqra_env.ppf(stats.norm.cdf(data_mcs_u))
            # data.y = solver.run(data_mcs_x, verbose=True) 

            # try:
                # filename = '{:s}_McsE6R{:d}_{:d}.npy'.format(solver.nickname,r, itheta)
                # print('saving data: {}'.format(os.path.join(data_dir_result, filename)))
                # np.save(os.path.join(data_dir_result, filename), data, allow_pickle=True)
            # except:
                # np.save('SDOF_SURGE_McsE6.npy', data, allow_pickle=True)

    # ------------------------ Grid for contour plot----------------- ###
    # solver = uqra.FPSO(random_state=np.arange(100))
    # x0 = np.linspace(0,20,210)
    # x1 = np.linspace(0,35,360)
    # x0, x1  = np.meshgrid(x0, x1)
    # x = np.array([x0.flatten(),x1.flatten()])
    # u = stats.norm(0, 0.5**0.5).ppf(uqra_env.cdf(x))
    # y = solver.run(x)
    # data = uqra.Data()
    # data.x = np.array([x0, x1])
    # data.u = np.array([u[0].reshape(x0.shape), u[1].reshape(x1.shape)])
    # data.y = np.array([iy.reshape(x0.shape) for iy in y])
    # try:
        # np.save(os.path.join(data_dir_result, 'SDOF_SURGE_Grid.npy'), data, allow_pickle=True)
    # except:
        # np.save('SDOF_SURGE_Grid.npy', data, allow_pickle=True)
    

    # ------------------------ Basic Check ----------------- ###
    # solver = uqra.FPSO()
    # x = np.array([2,4]).reshape(2,-1)
    # y = solver.run(x) 
    # print('Hs = {}, Tp={} => y={}'.format(np.around(x[0]), np.around(x[1]), y))

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

    # ------------------------ Environmental Contour ----------------- ###
    # U, X = uqra_env.environment_contour(50, T=1800, n=360, q=0.5)
    # solver = uqra.FPSO(random_state=np.arange(20))
    # data = uqra.Data()
    # data.u = U
    # data.x = X
    # data.y = solver.run(X)
    # filename = '{:s}_Norway5_EC2D_50yr.npy'.format(solver.nickname) 
    # np.save(filename, data, allow_pickle=True)



    ## ------------------------ Environmental Contour ----------------- ###

    # solver  = uqra.FPSO(phase=np.arange(20))
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


def run_samples_same():
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
    # np.save('run_linear_oscillator_y', samples_y)
    # # filename_tags = ['R0']
    # # filename_tags = [itag+'_y' for itag in filename_tags]
    # # uqra_dataio.save_data(samples_y, 'run_linear_oscillator', os.getcwd(), filename_tags)
    # samples_y_stats = solver.get_stats()
    # np.save('run_linear_oscillator_y_stats', samples_y_stats)
    # # filename_tags = [itag+'_y_stats' for itag in filename_tags]
    # # uqra_dataio.save_data(samples_y_stats, 'run_linear_oscillator', os.getcwd(), filename_tags)


def run_Ishigami():
    solver = uqra.Ishigami()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/Ishigami/TestData' 
    data_u = []
    for r in range(10):
        data = uqra.Data()
        filename = 'CDF_McsE7R{:d}.npy'.format(r)
        print(filename)
        data.u = np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:]*2.0-1.0
        data.xi= data.u
        data.x = data.u*np.pi 
        data.y = solver.run(data.x)
        filename = '{:s}_CDF_McsE6R{:d}.npy'.format(solver.nickname, r)
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_FourBranch():
    solver = uqra.FourBranchSystem()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/Branches/TestData' 
    data_u = []
    for r in range(10):
        data = uqra.Data()
        # filename = 'CDF_McsE7R{:d}.npy'.format(r)
        # data.u = stats.norm.ppf(np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:])
        filename = 'DoE_McsE6R{:d}_norm.npy'.format(r)
        data.u = np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:]
        print(filename)
        data.xi= data.u * np.sqrt(0.5)
        data.x = data.u
        data.y = solver.run(data.x)
        print(np.sum(data.y<0)/len(data.y))
        print('u: {} {}'.format(np.mean(data.u, axis=-1), np.std(data.u, axis=-1)))
        print('xi: {} {}'.format(np.mean(data.xi, axis=-1), np.std(data.xi, axis=-1)))
        print('x: {} {}'.format(np.mean(data.x, axis=-1), np.std(data.x, axis=-1)))
        # filename = '{:s}_CDF_McsE7R{:d}.npy'.format(solver.nickname, r)
        filename = '{:s}_McsE6R{:d}.npy'.format(solver.nickname, r)
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_CornerPeak():
    solver = uqra.FourBranchSystem()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/CornerPeak/TestData' 
    data_u = []
    for r in range(10):
        data = uqra.Data()
        # filename = 'CDF_McsE7R{:d}.npy'.format(r)
        # data.u = stats.norm.ppf(np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:])
        filename = 'DoE_McsE6R{:d}_norm.npy'.format(r)
        data.u = np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:]
        print(filename)
        data.xi= data.u * np.sqrt(0.5)
        data.x = data.u
        data.y = solver.run(data.x)
        print('u: {} {}'.format(np.mean(data.u, axis=-1), np.std(data.u, axis=-1)))
        print('xi: {} {}'.format(np.mean(data.xi, axis=-1), np.std(data.xi, axis=-1)))
        print('x: {} {}'.format(np.mean(data.x, axis=-1), np.std(data.x, axis=-1)))
        filename = '{:s}_McsE6R{:d}.npy'.format(solver.nickname, r)
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_LiqudHydrogenTank():
    solver = uqra.LiqudHydrogenTank()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/LiqudHydrogenTank/TestData' 
    data_u = []
    for r in range(10):
        data = uqra.Data()
        # filename = 'CDF_McsE7R{:d}.npy'.format(r)
        # data.u = stats.norm.ppf(np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:])
        filename = 'McsE6R{:d}.npy'.format(r)
        print(filename)
        data.u = np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:]
        print('u: {} {}'.format(np.mean(data.u, axis=-1), np.std(data.u, axis=-1)))
        print('min ,max: {} {}'.format(np.amin(data.u, axis=-1), np.amax(data.u, axis=-1)))

        data.x = solver.map_domain(data.u, stats.uniform(0,1))
        print('x: {} {}'.format(np.mean(data.x, axis=-1), np.std(data.x, axis=-1)))
        data.y = solver.run(data.x)
        print(np.sum(np.isnan(data.y)))
        print(data.u[:,np.isnan(data.y)])
        # print('xi: {} {}'.format(np.mean(data.xi, axis=-1), np.std(data.xi, axis=-1)))
        # print('x: {} {}'.format(np.mean(data.x, axis=-1), np.std(data.x, axis=-1)))
        print('Probability of failure {:.2e}'.format(np.sum(data.y<0)/len(data.y)))
        filename = '{:s}_McsE6R{:d}.npy'.format(solver.nickname, r)
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_InfiniteSlope():
    solver = uqra.InfiniteSlope()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/LiqudHydrogenTank/TestData' 
    data_u = []
    for r in range(10):
        data = uqra.Data()
        filename = 'DoE_McsE6R{:d}_uniform.npy'.format(r)
        data.u = np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:]
        print(data.u.shape)
        # data.u = stats.norm.rvs(size= (5, int(1e7)))
        print(filename)
        data.xi= data.u
        data.x = solver.map_domain(data.u, stats.uniform(-1,2))
        data.y = solver.run(data.x)
        np.set_printoptions(precision=4)
        print('u: {} {}'.format(np.mean(data.u, axis=-1), np.std(data.u, axis=-1)))
        print('xi: {} {}'.format(np.mean(data.xi, axis=-1), np.std(data.xi, axis=-1)))
        print('x: {} {} {}'.format(np.mean(data.x, axis=-1), np.amin(data.x, axis=-1),np.amax(data.x, axis=-1)))
        print('x: {} {}'.format(np.mean(np.log(data.x[2:4]), axis=-1), np.std(np.log(data.x[2:4]), axis=-1)))
        theta_deg, phi_deg = data.x[2:4]
        print(np.mean(theta_deg), np.std(theta_deg), np.std(theta_deg)/np.mean(theta_deg))
        print(np.mean(phi_deg), np.std(phi_deg), np.std(phi_deg)/np.mean(phi_deg))
        print('Probability of failure {:.2e}'.format(np.sum(data.y<0)/len(data.y)))
        filename = '{:s}_McsE6R{:d}.npy'.format(solver.nickname, r)
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_GaytonHat():
    solver = uqra.GaytonHat()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples', solver.nickname, 'TestData')
    data_u = []
    if not os.path.exists(data_dir_test):
        os.makedirs(data_dir_test)
    for r in range(10):
        data = uqra.Data()
        data.u = stats.norm.rvs(size= (solver.ndim, int(5e7)))
        filename = '{:s}_Mcs5E7R{:d}.npy'.format(solver.nickname, r)
        print(filename)
        print(data.u.shape)
        # data.xi= data.u * np.sqrt(0.5)
        # data.x = solver.map_domain(data.u, stats.norm(0,1))
        data.x = data.u
        data.y = solver.run(data.x)
        np.set_printoptions(precision=4)
        print('Probability of failure {:.2e}'.format(np.sum(data.y<0)/len(data.y)))
        data.pf = np.sum(data.y<0)/len(data.y)
        data.ecdf = uqra.ECDF(data.y, data.pf, compress=True)
        del data.x
        del data.y
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_CompositeGaussian():
    solver = uqra.CompositeGaussian()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples', solver.nickname, 'TestData')
    if not os.path.exists(data_dir_test):
        os.makedirs(data_dir_test)
    data_u = []
    for r in range(10):
        data = uqra.Data()
        data.u = stats.norm.rvs(size= (solver.ndim, int(1e7)))
        filename = '{:s}_McsE7R{:d}.npy'.format(solver.nickname, r)
        print(filename)
        data.x = solver.map_domain(data.u, stats.norm(0,1))
        data.y = solver.run(data.x)
        print('Probability of failure {:.2e}'.format(np.sum(data.y<0)/len(data.y)))
        data.pf = np.sum(data.y<0)/len(data.y)
        data.ecdf = uqra.ECDF(data.y, data.pf, compress=True)
        del data.x
        del data.y

        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_Rastrigin():
    solver = uqra.Rastrigin()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples', solver.nickname, 'TestData')
    if not os.path.exists(data_dir_test):
        os.makedirs(data_dir_test)
    data_u = []
    for r in range(10):
        data = uqra.Data()
        filename = 'McsE6R{:d}.npy'.format(r)
        print(filename)
        data.u = np.load(os.path.join(data_dir_testin, filename))[:solver.ndim,:]
        print('u: {} {}'.format(np.mean(data.u, axis=-1), np.std(data.u, axis=-1)))
        print('min ,max: {} {}'.format(np.amin(data.u, axis=-1), np.amax(data.u, axis=-1)))

        data.x = solver.map_domain(data.u, stats.uniform(0,1))
        print('x: {} {}'.format(np.mean(data.x, axis=-1), np.std(data.x, axis=-1)))
        data.y = solver.run(data.x)
        print('Probability of failure {:.2e}'.format(np.sum(data.y<0)/len(data.y)))
        data.pf = np.sum(data.y<0)/len(data.y)
        data.ecdf = uqra.ECDF(data.y, data.pf, compress=True)
        del data.x
        del data.y

        filename = '{:s}_McsE6R{:d}.npy'.format(solver.nickname, r)
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_Borehole():
    solver = uqra.Borehole()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples', solver.nickname, 'TestData')
    if not os.path.exists(data_dir_test):
        os.makedirs(data_dir_test)
    data_u = []
    for r in range(10):
        data = uqra.Data()
        data.u = stats.uniform(-1,2).rvs(size= (solver.ndim, int(1e6)))
        filename = '{:s}_McsE6R{:d}.npy'.format(solver.nickname, r)
        print(filename)
        data.x = solver.map_domain(data.u, stats.uniform(-1,2))
        data.y = solver.run(data.x)
        # print('Probability of failure {:.2e}'.format(np.sum(data.y<0)/len(data.y)))
        # data.pf = np.sum(data.y<0)/len(data.y)
        # data.ecdf = uqra.ECDF(data.y, data.pf, compress=True)
        del data.x
        del data.y
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

def run_ExpTanh():
    solver = uqra.ExpTanh()
    data_dir_testin = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/Random' 
    data_dir_test   = os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples', solver.nickname, 'TestData')
    if not os.path.exists(data_dir_test):
        os.makedirs(data_dir_test)
    data_u = []
    for r in range(10):
        data = uqra.Data()
        data.u = stats.uniform(-1,2).rvs(size= (solver.ndim, int(1e6)))
        filename = '{:s}_McsE6R{:d}.npy'.format(solver.nickname, r)
        print(filename)
        data.x = solver.map_domain(data.u, stats.uniform(-1,2))
        data.y = solver.run(data.x)
        # print('Probability of failure {:.2e}'.format(np.sum(data.y<0)/len(data.y)))
        # data.pf = np.sum(data.y<0)/len(data.y)
        # data.ecdf = uqra.ECDF(data.y, data.pf, compress=True)
        del data.x
        del data.y
        np.save(os.path.join(data_dir_test, filename), data, allow_pickle=True)

if __name__ == '__main__':
    run_FPSO()
    # run_Ishigami()
    # run_FourBranch()
    # run_CornerPeak()
    # run_LiqudHydrogenTank()
    # run_InfiniteSlope()
    # run_GaytonHat()
    # run_CompositeGaussian()
    # run_Rastrigin()
    # run_Borehole()
    # run_ExpTanh()

