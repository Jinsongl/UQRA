#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra
import numpy as np, os, sys
import scipy.stats as stats
from tqdm import tqdm
import itertools, copy, math
import multiprocessing as mp
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()
class Data():
    pass
def observation_error(y, mu=0, cov=0.03, random_state=100):
    e = stats.norm(0, cov * abs(y)).rvs(size=len(y), random_state=random_state)
    return e

def main(s=0):

    ## ------------------------ Displaying set up ------------------- ###
    print('\n#################################################################################')
    print(' >>>  Start UQRA : {:d}'.format(s), __file__)
    print('#################################################################################\n')
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    pf = np.array([1e-6])
    np.random.seed(100)
    ## ------------------------ Define solver ----------------------- ###
    # solver      = uqra.ExpAbsSum(stats.uniform(-1,2),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.uniform(-1,2),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.uniform(-1,2), d=2)
    # solver      = uqra.ProductPeak(stats.uniform(-1,2), d=2,c=[-3,2],w=[0.5,0.5])
    # solver      = uqra.Franke()
    # solver      = uqra.Ishigami()

    # solver      = uqra.ExpAbsSum(stats.norm(0,1),d=2,c=[-2,1],w=[0.25,-0.75])
    # solver      = uqra.ExpSquareSum(stats.norm(0,1),d=2,c=[1,1],w=[1,0.5])
    # solver      = uqra.CornerPeak(stats.norm(0,1), d=3, c=np.array([1,2,3]), w=[0.5,]*3)
    # solver      = uqra.ProductPeak(stats.norm(0,1), d=2, c=[-3,2], w=[0.5,]*2)
    # solver      = uqra.ExpSum(stats.norm(0,1), d=3)
    solver      = uqra.FourBranchSystem()

    uqra_env = solver.distributions[0]

    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling()
    model_params.name    = 'PCE'
    model_params.degs    = np.arange(2,11) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Hem'
    model_params.fitting = 'OLS' 
    model_params.n_splits= 50
    model_params.alpha   = 2
    model_params.num_test= int(1e6)
    model_params.num_pred= int(1e8)
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters()
    doe_params.doe_sampling = 'CLS4' 
    doe_params.optimality   = [None,'S', 'D']
    doe_params.poly_name    = model_params.basis 
    doe_params.num_cand     = int(1e5)
    # data_dir_cand   = '/Users/jinsongliu/BoxSync/Research/Working_Papers/OE2020_LongTermExtreme/Data/FPSO_SURGE/UniformBall'
    if doe_params.doe_sampling.lower() == 'lhs':
        data_dir_optimal = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/ExperimentalDesign/LHS'
        doe_params.update_output_dir(data_dir_optimal=data_dir_optimal)

    u = []
    xi= []
    y = []
    print('----------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------')
    for r in range(10):
        filename   = '{:s}_CDF_McsE6R{:d}.npy'.format(solver.nickname, r)
        print(filename)
        date_pred_ = np.load(os.path.join('/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples', \
            solver.nickname, 'TestData', filename), allow_pickle=True).tolist()
        u.append(date_pred_.u)
        xi.append(date_pred_.xi)
        y.append(date_pred_.y)
    data_pred = uqra.Data()
    data_pred.u = np.concatenate(u, axis=-1)
    data_pred.xi= np.concatenate(xi,axis=-1)
    data_pred.y = np.concatenate(y, axis=-1)
    print(data_pred.y.shape)

    # with mp.Pool(processes=mp.cpu_count()) as p:
        # y0_ecdf= list(tqdm(p.imap(uqra.ECDF, [(uqra.bootstrapping(data_pred.y, 1, bootstrap_size=model_params.num_pred), pf, True) for _ in range(10)]), ncols=80, total=10, desc='  [Boostraping]'))
    # print(y0_ecdf)

    ndim_deg_cases = np.array(list(itertools.product([model_params.ndim,], model_params.degs)))
    output_ndim_deg= []
    for ndim, deg in ndim_deg_cases:
        print('    ----------------------------------------------------------------------------------')
        print('    ----------------------------------------------------------------------------------')
        ## ------------------------ UQRA Surrogate model----------------- ###
        orth_poly = uqra.poly.orthogonal(ndim, deg, model_params.basis)
        pce_model = uqra.PCE(orth_poly)
        pce_model.info()

        ## ------------------------ Updating DoE parameters ----------------- ###
        idoe_params = copy.deepcopy(doe_params)
        idoe_params.ndim = ndim
        idoe_params.deg  = int(deg)
        ## Specify filename template function
        # filename_template= lambda s: r'DoE_Ball5pt6E5R{:d}'.format(s)
        # idoe_params.update_filenames(s, filename_template)
        ## If not specified, default values will be used
        idoe_params.update_filenames(s)
        ### return data dirctories and filenames
        filename_cand   = idoe_params.fname_cand
        data_dir_cand   = idoe_params.data_dir_cand
        data_dir_optimal= idoe_params.data_dir_optimal

        ## ------------------------ UQRA Simulation Parameters ----------------- ###
        sim_params = uqra.Simulation(solver, pce_model, idoe_params)
        sim_params.update_filenames(s)
        filename_testin = sim_params.fname_testin
        filename_test   = sim_params.fname_test
        data_dir_test   = sim_params.data_dir_test
        data_dir_testin = sim_params.data_dir_testin
        data_dir_result = sim_params.data_dir_result
        figure_dir      = sim_params.figure_dir

        print('   > {:<25s}'.format('Input/Output Directories:'))
        print('     - {:<23s} : {:s}'.format(' Candiate samples'  , data_dir_cand))
        print('     - {:<23s} : {:s}'.format(' UQRA DoE data '    , data_dir_optimal))
        print('     - {:<23s} : {:s}'.format(' Test input '       , data_dir_testin))
        print('     - {:<23s} : {:s}'.format(' Test output'       , data_dir_test))
        print('     - {:<23s} : {:s}'.format(' UQRA output data ' , data_dir_result))
        print('     - {:<23s} : {:s}'.format(' UQRA output figure', figure_dir))
        print('   > {:<25s}'.format('Input/Output files'))
        print('     - {:<23s} : {}'.format(' Cadidate samples'  , filename_cand  ))
        print('     - {:<23s} : {}'.format(' Test input data'   , filename_testin))
        print('     - {:<23s} : {}'.format(' Test output data'  , filename_test  ))
        if filename_cand:
            data_cand = np.load(os.path.join(data_dir_cand, filename_cand))[:ndim, :]
            print('     ..{:<30s} shape: {}'.format(' Candidate samples loaded,', data_cand.shape))

        ### 2. Get test data set
        try:
            data_test = np.load(os.path.join(data_dir_test, filename_test), allow_pickle=True).tolist()
            if isinstance(data_test,  uqra.Data):
                pass
            else:
                data_test = data_test[0]
                assert isinstance(data_test, (Data, uqra.Data)), 'Type: {}'.format(type(data_test))
        except FileNotFoundError:
            print('     - Preparing Test data (UQRA.Solver: {:s})... '.format(solver.nickname))
            filename_testin = os.path.join(data_dir_cand, filename_testin)
            print('     .. Input test data:', filename_testin)

            data_test   = uqra.Data()
            data_test.u = np.load(filename_testin)[:ndim, :model_params.num_test]
            if doe_params.doe_sampling.lower() == 'cls4':
                data_test.xi = data_test.u* np.sqrt(0.5)
            else:
                data_test.xi = data_test.u
            data_test.x = uqra_env.ppf(stats.norm.cdf(data_test.u))
            data_test.y = solver.run(data_test.x)
            np.save(os.path.join(data_dir_test, filename_test), data_test, allow_pickle=True)
            print('     .. Saving test data to {:s}, shape: x={}, y={} '.format(filename_test, 
                data_test.x.shape, data_test.y.shape))
        print('     ..{:<30s} shape: {} '.format(' Test data loaded,', data_test.y.shape))

        output_indim_ideg = uqra.Data()
        if idoe_params.doe_sampling.lower() == 'lhs':
            all_doe_cases = [(idoe_params.doe_sampling, None)]
        else:
            all_doe_cases = [(idoe_params.doe_sampling, ioptimality) for ioptimality in idoe_params.optimality] 

        for idoe_sampling, ioptimality in all_doe_cases:
            idoe_sampling = idoe_sampling.lower()
            idoe_nickname = idoe_params.doe_nickname(idoe_sampling, ioptimality)
            n_samples    = model_params.alpha * pce_model.num_basis
            print('     --------------------------------------------------------------------------------')
            print('   >> UQRA Training with Experimental Design {} '.format(idoe_nickname))
            print('   -> Training with (n={:d}, alpha={:.2f}) samples'.format(n_samples, model_params.alpha))

            if idoe_sampling.lower() == 'lhs':
                filename_design = idoe_params.fname_design(n_samples)
            else:
                filename_design = idoe_params.fname_design
            print('     - {:<23s} : {}'.format(' UQRA DoE filename' , filename_design))
            data_design  = np.load(os.path.join(data_dir_optimal, filename_design), allow_pickle=True).tolist()
            print('     ..{:<23s} : {}'.format(' # optimal sample sets,', len(data_design)))
            ### if data_deisgn has more than one set of optimal samples, choose the first one
            if isinstance(data_design, list):
                data_design = data_design[0]
            if idoe_sampling.lower() == 'lhs':
                data_design = np.array(data_design)
                assert data_design.shape == (ndim, n_samples) 
                u_train = data_design 
            else:
                assert isinstance(data_design, (Data, uqra.Data)),'TypeError: expected uqra.Data, but {} given'.format(type(data_design))
                assert data_design.deg == deg and data_design.ndim == model_params.ndim
                optimal_samples_idx = getattr(data_design, idoe_nickname)
                if len(optimal_samples_idx) < n_samples:
                    raise ValueError(' Requesting {:d} samples but only {:d} available...'.format(
                        n_samples, len(optimal_samples_idx)))
                u_train = data_cand[:model_params.ndim, optimal_samples_idx[:n_samples]]
                if idoe_sampling.lower()=='cls4':
                    u_train = u_train * deg **0.5

            x_train = uqra_env.ppf(pce_model.orth_poly.dist_u.cdf(u_train))
            ### 3. train model 
            y_train = solver.run(x_train) 
            y_train = y_train + observation_error(y_train)
            U_train = pce_model.orth_poly.vandermonde(u_train)
            # X_train = orth_poly.vandermonde(x_train)
            if idoe_sampling.lower().startswith('cls'):
                ### reproducing kernel
                WU_train = pce_model.orth_poly.num_basis**0.5*(U_train.T / np.linalg.norm(U_train, axis=1)).T
                w = pce_model.christoffel_weight(u_train, active=None)
            else:
                WU_train = U_train
                w = None
            ## condition number, kappa = max(svd)/min(svd)
            _, sigular_values, _ = np.linalg.svd(WU_train)
            if idoe_sampling.lower().startswith('cls'):
                data_test_u = data_test.xi
                data_pred_u = data_pred.xi
            elif idoe_sampling.lower().startswith('mcs'):
                data_test_u = data_test.u
                data_pred_u = data_pred.u
            elif idoe_sampling.lower() == 'lhs':
                data_test_u = data_test.u
                data_pred_u = data_pred.u

            # pce_model.fit_lassolars(u_train, y_train, w=w)
            data = uqra.Data()
            data.kappa  = max(abs(sigular_values)) / min(abs(sigular_values))
            pce_model.fit(model_params.fitting, u_train, y_train, w=w, n_jobs=4)
            y_test      = pce_model.predict(data_test_u, n_jobs=4)
            y_pred      = pce_model.predict(data_pred_u, n_jobs=4)
            data.rmse_y = uqra.metrics.mean_squared_error(data_test.y, y_test, squared=False)
            data.model  = pce_model
            data.y0_hat = uqra.metrics.mquantiles(y_pred, prob=1-pf)
            data.y0     = uqra.metrics.mquantiles(data_pred.y, prob=1-pf)
            data.ypred_ecdf = uqra.ECDF(y_pred, alpha=pf, compress=True)
            # data.y0_ecdf=y0_ecdf
            data.score  = pce_model.score
            data.cv_error = pce_model.cv_error


            tqdm.write(' > Summary')
            with np.printoptions(precision=4):
                # tqdm.write('     - {:<15s} : {}'.format( 'QoI'       , QoI))
                tqdm.write('     - {:<15s} : {}'.format( 'RMSE y ' , data.rmse_y))
                tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'  , data.cv_error))
                tqdm.write('     - {:<15s} : {}'.format( 'Score '  , data.score))
                tqdm.write('     - {:<15s} : {}'.format( 'kappa '   , data.kappa))
                tqdm.write('     - {:<15s} : {} [{}]'.format( 'y0 ' , data.y0_hat, data.y0))
            setattr(output_indim_ideg, idoe_nickname, data)
        output_ndim_deg.append(output_indim_ideg)

    ## ============ Saving QoIs ============
    filename = '{:s}_{:s}_{:s}E5R{:d}'.format(solver.nickname, pce_model.tag, doe_params.doe_sampling.capitalize(), s)
    try:
        np.save(os.path.join(data_dir_result, filename), output_ndim_deg, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(os.path.join(os.getcwd(), filename), output_ndim_deg, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))

if __name__ == '__main__':
    main(0)
