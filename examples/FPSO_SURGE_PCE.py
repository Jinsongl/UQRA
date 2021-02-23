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
import itertools, copy, math, collections
import multiprocessing as mp
import random
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def observation_error(y, mu=0, cov=0.03, random_state=100):
    e = stats.norm(0, cov * abs(y)).rvs(size=len(y), random_state=random_state)
    return e

def run_UQRA_OptimalDesign(x, poly, doe_sampling, optimality, n_samples, 
        optimal_samples=[], active_index=None, random_state=None):
    optimal_samples = 'RRQR' if len(optimal_samples) == 0 else copy.deepcopy(optimal_samples)
    x = poly.deg**0.5 * x if doe_sampling.lower() in ['cls4', 'cls5'] else x
    if doe_sampling.lower().startswith('cls'):
        X = poly.vandermonde(x)
        X = poly.num_basis**0.5*(X.T / np.linalg.norm(X, axis=1)).T
    else:
        X = poly.vandermonde(x)
    if active_index is None:
        X = X
    else:
        X = X[:, active_index]

    uqra.blockPrint()
    if str(optimality).upper() == 'NONE':
        idx_cand  = np.arange(X.shape[0])
        random.seed(random_state)
        random.shuffle(idx_cand)
        idx_optimal = idx_cand[:n_samples]
    else:
        doe = uqra.OptimalDesign(X)
        idx_optimal = doe.samples(optimality, n_samples, initialization=optimal_samples) ## additional n_samples new samples
    uqra.enablePrint()
    if isinstance(optimal_samples, (list, tuple)):
        idx_optimal = [i for i in idx_optimal if i not in optimal_samples]
    return idx_optimal

def list_union(ls1, ls2):
    """
    append ls2 to ls1 and check if there exist duplicates
    return the union of two lists and remove duplicates
    """
    ls = list(copy.deepcopy(ls1)) + list(copy.deepcopy(ls2))
    if len(ls) != len(set(ls1).union(set(ls2))):
        print('[WARNING]: list_union: duplicate elements found in list when append to each other')
    ls = list(set(ls))
    return ls

def isOverfitting(cv_err):
    if len(cv_err) < 3 :
        return False
    if cv_err[-1] > cv_err[-2] and cv_err[-2] > cv_err[0]:
        print('WARNING: Overfitting')
        return False

def threshold_converge(y, threshold=0.95):
    y = np.array(y)
    status = True if y[-1]> threshold else False
    return status, threshold

def relative_converge(y, err=0.05):
    """ 
    check if y is converge in relative error
    return: (status, error)
        status: Boolean for convergeing or not
        error: absolute error

    """
    y = np.array(y)
    if len(y) < 2:
        res = (False, np.nan)
    else:
        error = abs((y[-2]-y[-1])/ y[-1])
        res = (error < err, error)
    return res 

def absolute_converge(y, err=1e-4):
    """ 
    check if y is converge in absolute error
    return: (status, error)
        status: Boolean for convergeing or not
        error: absolute error

    """
    y = np.array(y)
    if len(y) < 2:
        res = (False, np.nan)
    else:
        error = abs(y[-2]-y[-1])
        res = (error < err, error)
    return res 

def main(model_params, doe_params, solver, r=0, random_state=None):
    random.seed(random_state)
    data = uqra.Data()
    data.ndim       = []
    data.deg        = []
    data.xi_train   = []
    data.x_train    = []
    data.y_train    = [] 
    data.rmse_y     = []
    data.y0_hat     = [] 
    data.cv_err     = [] 
    data.model      = []
    data.score      = []
    data.yhat_ecdf  = [] 
    data.path       = []
    data.DoI_candidate = []
    data.DoI_optimal   = []
    data.global_optimal= []

    optimal_samples = []
    ndim_deg_cases  = np.array(list(itertools.product([model_params.ndim,], model_params.degs)))

    xi_train = np.empty((model_params.ndim, 0))
    x_train  = np.empty((model_params.ndim, 0))
    y_train  = np.empty((0,))


    for i, (ndim, deg) in enumerate(ndim_deg_cases):
        print('\n==================================================================================')
        print('         <<<< Global iteration No. {:d}: ndim={:d}, p={:d} >>>>'.format(i+1, ndim, deg))
        print('==================================================================================\n')
        ## ------------------------ UQRA Surrogate model----------------- ###
        orth_poly = uqra.poly.orthogonal(ndim, deg, model_params.basis)
        pce_model = uqra.PCE(orth_poly)
        dist_u    = model_params.dist_u 
        dist_xi   = orth_poly.weight
        dist_x    = solver.distributions
        pce_model.info()

        ## ------------------------ Updating DoE parameters ----------------- ###
        idoe_params = copy.deepcopy(doe_params)
        idoe_params.ndim = ndim
        idoe_params.deg  = int(deg)
        ### Specify candidate data filename template function
        ### e.g.  filename_template= lambda r: r'DoE_Ball5pt6E5R{:d}.npy'.format(r)
        ### if not specified, default values will be used
        idoe_params.update_filenames(filename_template=None)
        filename_cand = idoe_params.fname_cand(r)
        # filename_design = idoe_params.fname_design(r)
        print('     - {:<23s} : {}'.format(' Candidate filename'  , filename_cand  ))

        if filename_cand:
            data_cand = np.load(os.path.join(data_dir_cand, filename_cand))
            data_cand = data_cand[:ndim,random.sample(range(data_cand.shape[1]), k=idoe_params.num_cand)]
            print('       {:<23s} : {}'.format(' shape', data_cand.shape))
        else:
            data_cand = None
            print('       {:<23s} : {}'.format(' shape', data_cand))

        idoe_sampling = idoe_params.doe_sampling.lower()
        idoe_nickname = idoe_params.doe_nickname()
        ioptimality   = idoe_params.optimality
        print('     - {:<23s} : {}'.format(' UQRA DoE '  , idoe_nickname))
        ### temp data object containing results from intermedia steps
        data_temp = uqra.Data()
        data_temp.y0_hat   = []
        data_temp.cv_err   = []
        data_temp.kappa    = []
        data_temp.rmse_y   = []
        data_temp.model    = []
        data_temp.score    = []
        data_temp.yhat_ecdf= []
        optimal_samples_ideg=[]
        boundary_data = uqra.Data() 
        DoI_candidate = []
        DoI_optimal   = []

        print(' ------------------------------------------------------------')
        print(' > Adding optimal samples in global domain... ')
        print('   1. optimal samples based on FULL basis')
        active_index = pce_model.active_index
        active_basis = pce_model.active_basis
        if deg == model_params.degs[0]:
            n_samples = len(active_index) *2
        else:
            n_samples = len(active_index)
        n_samples = len(active_index) * model_params.alpha

        print('     - Optimal design:{:s}, Adding {:d} optimal samples'.format(idoe_nickname, n_samples))

        # idx_optimal = run_UQRA_OptimalDesign(data_cand, orth_poly, idoe_sampling, ioptimality, n_samples, 
                # optimal_samples=optimal_samples_ideg, active_index=None)
        # optimal_samples      = list_union(optimal_samples     , idx_optimal)
        # optimal_samples_ideg = list_union(optimal_samples_ideg, idx_optimal)

        xi_train_ = idoe_params.get_samples(data_cand, orth_poly, n_samples, xi_train, active_index=None, initialization='RRQR') 
        # print('     - {:<32s} : {:d}'.format('No. optimal samples [p='+str(deg)+']', len(optimal_samples_ideg)))
        # print('     - {:<32s} : {:d}'.format('Total number of optimal samples', len(optimal_samples)))

        # print('   2. Sparsity estimation with {:s}'.format(model_params.fitting.upper()))
        print('   2. Training with {} '.format(model_params.fitting))
        # xi_train       = data_cand[:, optimal_samples] 
        # global_optimal = data_cand[:, idx_optimal    ]
        # if idoe_sampling.lower()=='cls4':
            # xi_train       = xi_train       * deg **0.5
            # global_optimal = global_optimal * deg **0.5


        x_train_ = solver.map_domain(xi_train_, dist_xi)
        y_train_ = solver.run(x_train_)

        xi_train = np.concatenate((xi_train, xi_train_), axis=1)
        x_train  = np.concatenate((x_train , x_train_ ), axis=1)
        y_train  = np.concatenate((y_train , y_train_ ), axis=0)

        weight  = doe_params.sampling_weight()   ## weight function
        pce_model.fit(model_params.fitting, xi_train, y_train, w=weight,
                n_jobs=model_params.n_jobs) #, n_splits=model_params.n_splits
        data.global_optimal.append(xi_train_)
        print('     - {:<32s} : {:d}'.format('Total number of optimal samples', len(optimal_samples)))
        print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', x_train.shape[1], pce_model.num_basis, 
                        x_train.shape[1]/pce_model.num_basis))
        print('     - {:<32s} : {}'.format('Y train'    , y_train.shape))
        print('     - {:<32s} : {}'.format('Sparsity'   , len(pce_model.active_index)))

        print('   3. Prediction with {} samples '.format(xi_test.shape))
        y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
        data_temp.model.append(pce_model)
        data_temp.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
        data_temp.y0_hat.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
        data_temp.score.append(pce_model.score)
        data_temp.cv_err.append(pce_model.cv_error)
        data_temp.yhat_ecdf.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
        # isOverfitting(data_temp.cv_err) ## check Overfitting
        print('     - {:<32s} : {:.4e}'.format('y0 test [ PCE ]', data_temp.y0_hat[-1]))
        print('     - {:<32s} : {:.4e}'.format('y0 test [TRUE ]', y0_test))
        # isConverge, error_converge = relative_converge(data_temp.y0_hat, err=model_params.rel_err)
        # isConverge, error_converge = absolute_converge(data_temp.y0_hat, err=model_params.abs_err)
        # print('   4. Converge check ...')
        # print('      - Value : {} [Ref: {:e}]'.format(np.array(data_temp.y0_hat), y0_test))
        # print('      - Error : {:.2e}'.format(np.array(error_converge)))
        print(' ------------------------------------------------------------')
        print(' > Adding optimal samples in domain of interest... ')
        i_iteration = 1
        # while True:
            # ####-------------------------------------------------------------------------------- ####
            # print('                 ------------------------------')
            # print('                  <  Local iteration No. {:d} >'.format(i_iteration))
            # print('                 ------------------------------')
            # active_index = pce_model.active_index
            # active_basis = pce_model.active_basis 
            # sparsity     = len(pce_model.active_index)
            # n_samples    = sparsity# 5#min(2*sparsity, pce_model.num_basis)

            # print('   1. optimal samples based on SIGNIFICANT basis in domain of interest... ')
            # if idoe_sampling.lower()=='cls4':
                # xi_data_cand = data_cand*deg **0.5
            # else:
                # xi_data_cand = data_cand 
            # ### locate samples close to estimated y0 (domain of interest)
            # idx_DoI_data_test = np.argsort(abs(y_test_hat- data_temp.y0_hat[-1]))[:11] 

            # idx_DoI_data_cand = []
            # for idx_optimal in idx_DoI_data_test:
                # xi = xi_test[:, idx_optimal].reshape(ndim, -1)
                # idx_DoI_data_cand_ = np.argwhere(np.linalg.norm(xi_data_cand -xi, axis=0) < 0.1).flatten().tolist()
                # ### xi is outside data cand 
                # if len(idx_DoI_data_cand_) == 0:
                    # idx_DoI_data_cand_ = np.argsort(np.linalg.norm(xi_data_cand -xi, axis=0))[:100].tolist()
                # idx_DoI_data_cand = list(set(idx_DoI_data_cand + idx_DoI_data_cand_))
            # data_cand_DoI = data_cand[:, idx_DoI_data_cand] 
            # print('     - {:<32s} : {}'.format('DoI candidate samples', data_cand_DoI.shape ))
            # print('     - {:<32s} : {:d}'.format('Adding optimal boundary samples', n_samples))

            # idx_optimal_DoI = run_UQRA_OptimalDesign(data_cand_DoI, orth_poly, idoe_sampling, ioptimality, n_samples, 
                    # optimal_samples=[], active_index=active_index)
            # idx_optimal = [idx_DoI_data_cand[i] for i in idx_optimal_DoI if idx_DoI_data_cand[i] not in optimal_samples]
            # optimal_samples      = list_union(optimal_samples     , idx_optimal)
            # optimal_samples_ideg = list_union(optimal_samples_ideg, idx_optimal)

            # DoI_candidate.append(solver.map_domain(xi_data_cand[:, idx_DoI_data_cand], dist_xi))
            # DoI_optimal.append(solver.map_domain(xi_data_cand[:, idx_optimal], dist_xi))

            # print('     - {:<32s} : {:d}'.format('No. optimal samples [p='+str(deg)+']', len(optimal_samples_ideg)))
            # print('     - {:<32s} : {:d}'.format('Total number of optimal samples', len(optimal_samples)))

            # xi_train = data_cand[:, optimal_samples] 
            # if idoe_sampling.lower()=='cls4':
                # xi_train = xi_train * deg **0.5
            # x_train = solver.map_domain(xi_train, dist_xi)
            # y_train = solver.run(x_train)
            # print('   2. Training with {} '.format(model_params.fitting))
            # pce_model.fit(model_params.fitting, xi_train, y_train, w=weight,
                    # n_jobs=model_params.n_jobs) #, n_splits=model_params.n_splits
            # print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', x_train.shape[1], pce_model.num_basis, 
                            # x_train.shape[1]/pce_model.num_basis))
            # print('     - {:<32s} : {}'.format('Y train'    , y_train.shape))
            # print('     - {:<32s} : {}'.format('Sparsity'   , len(pce_model.active_index)))

            # print('   3. Prediction with {} samples '.format(xi_test.shape))
            # y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
            # data_temp.model.append(pce_model)
            # data_temp.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
            # data_temp.y0_hat.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
            # data_temp.score.append(pce_model.score)
            # data_temp.cv_err.append(pce_model.cv_error)
            # data_temp.yhat_ecdf.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
            # # isOverfitting(data_temp.cv_err) ## check Overfitting
            # print('     - {:<32s} : {:.4e}'.format('y0 test [ PCE ]', data_temp.y0_hat[-1]))
            # print('     - {:<32s} : {:.4e}'.format('y0 test [TRUE ]', y0_test))
            # isConverge, error_converge = relative_converge(data_temp.y0_hat, err=model_params.rel_err)
            # # isConverge, error_converge = absolute_converge(data_temp.y0_hat, err=model_params.abs_err)
            # print('   4. Converge check ...')
            # print('      - Value : {} [Ref: {:e}]'.format(np.array(data_temp.y0_hat), y0_test))
            # print('      - Error : {:.2e}'.format(np.array(error_converge)))
            # print('   ------------------------------------------------------------')
            # i_iteration +=1
            # if isConverge:
                # print('         !< Model converge for order {:d} >!'.format(deg))
                # break
            # if len(optimal_samples_ideg)>=2*orth_poly.num_basis:
                # print('         !< Number of samples exceeding 2P >!')
                # break

        print(' ------------------------------------------------------------')
        tqdm.write(' > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
        tqdm.write('  - {:<15s} : {:.4e}'.format( 'RMSE y ' , data_temp.rmse_y[-1]))
        tqdm.write('  - {:<15s} : {:.4e}'.format( 'CV MSE'  , data_temp.cv_err[-1]))
        tqdm.write('  - {:<15s} : {:.4f}'.format( 'Score '  , data_temp.score[-1] ))
        tqdm.write('  - {:<15s} : {:.4e} [{:.4e}]'.format( 'y0 ' , data_temp.y0_hat[-1], y0_test))
        print(' ------------------------------------------------------------')

        data.ndim.append(ndim)
        data.deg.append(deg)
        data.xi_train.append(xi_train)
        data.x_train.append( x_train)
        data.y_train.append( y_train)
        data.rmse_y.append ( data_temp.rmse_y[-1])
        data.y0_hat.append ( data_temp.y0_hat[-1])
        data.cv_err.append ( data_temp.cv_err[-1])
        data.model.append  ( data_temp.model [-1])
        data.score.append  ( data_temp.score [-1])
        data.yhat_ecdf.append(data_temp.yhat_ecdf[-1])
        data.DoI_candidate.append(DoI_candidate)
        data.DoI_optimal.append(DoI_optimal)
        del data_temp.yhat_ecdf
        data.path.append(data_temp)

        isOverfitting(data.cv_err) ## check Overfitting
        isConverge0, error_converge0 = relative_converge(data.y0_hat, err=model_params.rel_err)
        # isConverge0, error_converge0 = absolute_converge(data.y0_hat, err=model_params.abs_err)
        isConverge1, error_converge1 = threshold_converge(data.score)
        isConverge = [isConverge0, isConverge1]
        error_converge = [error_converge0, error_converge1]
        for i, (ikey, ivalue) in enumerate(zip(isConverge, error_converge)):
            print('  >  Checking #{:d} : {}, {:.2e}'.format(i, ikey, ivalue))
        if np.array(isConverge).all():
            tqdm.write('###############################################################################')
            tqdm.write('         Model Converge ')
            tqdm.write('    > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
            tqdm.write('     - {:<15s} : {}'.format( 'RMSE y ' , np.array(data.rmse_y)))
            tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'  , np.array(data.cv_err)))
            tqdm.write('     - {:<15s} : {}'.format( 'Score '  , np.array(data.score)))
            tqdm.write('     - {:<15s} : {} [{:.2e}]'.format( 'y0 ' , np.array(data.y0_hat), y0_test))
            for i, (ikey, ivalue) in enumerate(zip(isConverge, error_converge)):
                print('     >  Checking #{:d} : {}, {:.2e}'.format(i, ikey, ivalue))
            tqdm.write('###############################################################################')
            # break
    return data

if __name__ == '__main__':
    ## ------------------------ Displaying set up ------------------- ###
    r = 0
    theta = 1
    np.random.seed(100)
    random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    uqra_env = uqra.environment.Norway5(ndim=2)

    ## ------------------------ Define solver ----------------------- ###
    solver = uqra.FPSO(random_state=theta, distributions=uqra_env)
    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling('PCE')
    model_params.degs    = np.arange(2,11) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Heme'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLS' 
    model_params.n_splits= 50
    model_params.alpha   = 4
    model_params.num_test= int(1e7)
    model_params.num_pred= int(1e7)
    # model_params.pf      = np.array([1e-4])
    model_params.pf      = np.array([0.5/(365.25*24*50)])
    model_params.abs_err = 1e-4
    model_params.rel_err = 2.5e-2
    model_params.n_jobs  = mp.cpu_count()
    model_params.update_basis()
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    # doe_params = uqra.ExperimentParameters('CLS4', 'S')
    doe_params = uqra.ExperimentParameters('MCS', 'S')
    doe_params.poly_name = model_params.basis 
    doe_params.num_cand  = int(1e5)

    ## ------------------------ UQRA Simulation Parameters ----------------- ###
    sim_params = uqra.Simulation(solver, model_params, doe_params)
    filename_test   = lambda r: r'McsE7R{:d}'.format(r)
    sim_params.update_filenames(filename_test)

    data_dir_cand   = doe_params.data_dir_cand
    data_dir_optimal= doe_params.data_dir_optimal
    filename_testin = sim_params.fname_testin(r)
    filename_test   = sim_params.fname_test(r)
    data_dir_result = sim_params.data_dir_result
    figure_dir      = sim_params.figure_dir
    data_dir_test   = sim_params.data_dir_test
    data_dir_testin = sim_params.data_dir_testin

    ### 1. Get test data set
    data_test   = np.load(os.path.join(data_dir_test, filename_test), allow_pickle=True).tolist()
    data_test.x = solver.map_domain(data_test.u, model_params.dist_u)
    data_test.xi= model_params.map_domain(data_test.u, model_params.dist_u)
    data_test.y = solver.run(data_test.x) if not hasattr(data_test, 'y') else data_test.y
    try:
        data_test.y = data_test.y[theta]
    except:
        pass
    xi_test = data_test.xi[:, :model_params.num_test] 
    y_test  = data_test.y [   :model_params.num_test] 
    y0_test = uqra.metrics.mquantiles(y_test, 1-model_params.pf)

    res = []
    ith_batch  = 0
    batch_size = 50
    for i, irepeat in enumerate(range(batch_size*ith_batch, batch_size*(ith_batch+1))):
        print('\n#################################################################################')
        print(' >>>  File: ', __file__)
        print(' >>>  Start UQRA : {:d}[{:d}]/{:d} x {:d}'.format(i, irepeat, batch_size, ith_batch))
        print(' >>>  Test data R={:d}'.format(r))
        print('#################################################################################\n')
        print('   > {:<25s}'.format('Input/Output Directories:'))
        print('     - {:<23s} : {}'.format  (' Candiate samples'  , data_dir_cand))
        print('     - {:<23s} : {:s}'.format(' UQRA DoE data '    , data_dir_optimal))
        print('     - {:<23s} : {:s}'.format(' Test input '       , data_dir_testin))
        print('     - {:<23s} : {:s}'.format(' Test output'       , data_dir_test))
        print('     - {:<23s} : {:s}'.format(' UQRA output data ' , data_dir_result))
        print('     - {:<23s} : {:s}'.format(' UQRA output figure', figure_dir))
        print('   > {:<25s}'.format('Input/Output files'))
        print('     - {:<23s} : {}'.format(' Test input data'   , filename_testin))
        print('     - {:<23s} : {}'.format(' Test output data'  , filename_test  ))
        res.append(main(model_params, doe_params, solver, r=r, random_state=irepeat))
    filename = '{:s}_{:d}{:s}_{:s}E5R{:d}_{:d}{:d}_Alpha{:d}'.format(solver.nickname, 
            solver.ndim, model_params.basis,doe_params.doe_sampling.capitalize(), r, 
            batch_size, ith_batch, model_params.alpha)
    # ## ============ Saving QoIs ============
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))
