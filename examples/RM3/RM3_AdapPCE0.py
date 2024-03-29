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
import numpy as np, os, sys, io
import scipy.stats as stats
from tqdm import tqdm
import itertools, copy, math, collections
import multiprocessing as mp
import random
import scipy
import matlab.engine
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def overfitting_check(cv_err):
    """
    if cv error increase twice in a row, then defined as overfit

    return True if overfit, otherwise False

    """
    if len(cv_err) < 3 :
        return False, np.nan
    elif cv_err[-1] > cv_err[-2] and cv_err[-2] > cv_err[0]:
        return True, cv_err[-3:]
    else:
        return False, np.nan

def threshold_converge(y, threshold=0.95):
    y = np.array(y)
    if len(y) == 0:
        return False, np.nan
    else:
        status = True if y[-1]> threshold else False
        return status, y[-1]

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
    ndim_deg_cases = np.array(list(itertools.product([model_params.ndim,], model_params.degs)))

    main_res = []
    ### object contain all training samples
    data_train = uqra.Data()
    data_train.xi = np.empty((model_params.ndim, 0))
    data_train.x  = np.empty((model_params.ndim, 0))
    data_train.y  = np.empty((0,))

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
        data_ideg = uqra.Data()
        data_ideg.ndim      = ndim
        data_ideg.deg       = deg 
        data_ideg.y0_hat_   = []
        data_ideg.cv_err_   = []
        # data_ideg.kappa_    = []
        # data_ideg.rmse_y_  = []
        data_ideg.model_    = []
        data_ideg.score_    = []
        data_ideg.yhat_ecdf_= []
        data_ideg.DoI_candidate_ = []
        data_ideg.xi_train_ = []
        data_ideg.x_train_  = []
        data_ideg.y_train_  = []

        ## ------------------------ #1: Obtain global optimal samples ----------------- ###
        print(' ------------------------------------------------------------')
        print(' > Adding optimal samples in global domain... ')
        print('   1. optimal samples based on FULL basis')
        active_index = pce_model.active_index
        active_basis = pce_model.active_basis
        if deg == model_params.degs[0]:
            n_samples = math.ceil(len(active_index) * model_params.alpha)
        else:
            n_samples = len(active_index)
        print('     - Optimal design:{:s}, Adding {:d} optimal samples'.format(idoe_nickname, n_samples))

        ## obtain global optimal samples
        xi_train_, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=data_train.xi, 
                active_index=None, initialization='RRQR', return_index=True) 
        x_train_ = solver.map_domain(xi_train_, dist_xi)
        ii = np.where(np.array([iglobal_data.deg for iglobal_data in global_data]) == deg)[0][0]
        iglobal_data = global_data[ii]
        assert np.amax(abs(x_train_-iglobal_data.x_train)) < 1e-6  ## make sure train samples are same
        y_train_ = iglobal_data.y_train[:, model_params.channel, theta]/y_scale

        data_ideg.xi_train_.append(xi_train_)
        data_ideg.x_train_.append (x_train_)
        data_ideg.y_train_.append (y_train_)
        n_samples_deg = len(np.concatenate(data_ideg.y_train_, axis=0))
        data_train.xi  = np.concatenate([data_train.xi, xi_train_], axis=1)
        data_train.x   = np.concatenate([data_train.x , x_train_ ], axis=1)
        data_train.y   = np.concatenate([data_train.y , y_train_ ], axis=0)
        print('     - {:<32s} : {:d}'.format('No. optimal samples [p='+str(deg)+']', n_samples_deg))
        print('     - {:<32s} : {:.2f}'.format('Local oversampling [p='+str(deg)+']', n_samples_deg/pce_model.num_basis))
        print('     - {:<32s} : {:d}'.format('Total number of samples', len(data_train.y)))

        print('   2. Training with {} '.format(model_params.fitting))
        assert np.array_equal( data_train.x, solver.map_domain(data_train.xi, dist_xi))

        weight  = doe_params.sampling_weight()   ## weight function
        pce_model.fit(model_params.fitting, data_train.xi, data_train.y, w=weight,
                n_jobs=model_params.n_jobs) 

        print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], pce_model.num_basis, 
                        data_train.x.shape[1]/pce_model.num_basis))
        print('     - {:<32s} : {}'.format('Y train'    , data_train.y.shape))
        print('     - {:<32s} : {}'.format('Sparsity'   , len(pce_model.active_index)))

        print('   3. Prediction with {} samples '.format(xi_test.shape))
        y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
        data_ideg.model_.append(pce_model)
        # data_ideg.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
        data_ideg.y0_hat_.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
        data_ideg.score_.append(pce_model.score)
        data_ideg.cv_err_.append(pce_model.cv_error)
        # data_ideg.yhat_ecdf_.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
        print('     - {:<32s} : {:.4e}'.format('y0 test [ PCE ]', np.array(data_ideg.y0_hat_[-1])))
        # print('     - {:<32s} : {:.4e}'.format('y0 test [TRUE ]', y0_test))
        print(' ------------------------------------------------------------')
        print(' > Adding optimal samples in domain of interest (DoI)... ')
        i_iteration = 1
        while True:
            ####-------------------------------------------------------------------------------- ####
            active_index = pce_model.active_index
            active_basis = pce_model.active_basis 
            sparsity     = len(pce_model.active_index)
            n_samples    = sparsity# 5#min(2*sparsity, pce_model.num_basis)
            print('                 ------------------------------')
            print('                  <  Local iteration No. {:d} >'.format(i_iteration))
            print('                 ------------------------------')

            print('   1. optimal samples based on SIGNIFICANT basis in domain of interest... ')

            ## obtain DoI candidate samples
            data_cand_DoI, idx_data_cand_DoI = idoe_params.samples_nearby(data_ideg.y0_hat_[-1], xi_test, y_test_hat, data_cand
                    , deg, n0=10, epsilon=0.1, return_index=True)
            data_cand_xi_DoI = deg**0.5 * data_cand_DoI if idoe_params.doe_sampling in ['CLS4', 'CLS5'] else data_cand_DoI
            data_ideg.DoI_candidate_.append(solver.map_domain(data_cand_xi_DoI, dist_xi))

            ## obtain DoI optimal samples
            xi_train_, idx_optimal_DoI = idoe_params.get_samples(data_cand_DoI, orth_poly, n_samples, x0=[], 
                    active_index=active_index, initialization='RRQR', return_index=True) 
            assert xi_train_.shape[1] == n_samples ## make sure return number of samples required
            x_train_ = solver.map_domain(xi_train_, dist_xi)
            eng.workspace['deg']       = float(deg)
            eng.workspace['phaseSeed'] = float(theta)
            y_train_ = []
            for iHs, iTp in tqdm(x_train_.T, ncols=80, desc='   [WEC-SIM]' ):
                eng.workspace['Hs'] = float(iHs)
                eng.workspace['Tp'] = float(iTp)
                # eng.wecSim(nargout=0)
                eng.wecSim(nargout=0,stdout=out,stderr=err)
                y_train_.append(np.squeeze(eng.workspace['maxima'])[model_params.channel+2]/y_scale) ## first two are Hs,Tp
            y_train_ = np.array(y_train_)

            # y_train_ = data_train_.y[:, model_params.channel]
            data_ideg.xi_train_.append(xi_train_)
            data_ideg.x_train_.append (x_train_)
            data_ideg.y_train_.append (y_train_)
            n_samples_deg = len(np.concatenate(data_ideg.y_train_, axis=0))

            data_train.xi  = np.concatenate([data_train.xi, xi_train_], axis=1)
            data_train.x   = np.concatenate([data_train.x , x_train_ ], axis=1)
            data_train.y   = np.concatenate([data_train.y , y_train_ ], axis=0)
            assert np.array_equal( data_train.x, solver.map_domain(data_train.xi, dist_xi))




            print('     - {:<32s} : {}  '.format('DoI candidate samples', data_cand_DoI.shape ))
            print('     - {:<32s} : {:d}'.format('Adding DoI optimal samples', n_samples))
            print('     - {:<32s} : {:d}'.format('No. samples [p='+str(deg)+']', n_samples_deg))
            print('     - {:<32s} : {:.2f}'.format('Local oversampling [p='+str(deg)+']', n_samples_deg/pce_model.num_basis))
            print('     - {:<32s} : {:d}'.format('Total number of samples', len(data_train.y)))

            print('   2. Training with {} '.format(model_params.fitting))
            weight  = doe_params.sampling_weight()   ## weight function
            pce_model.fit(model_params.fitting, data_train.xi, data_train.y, w=weight, 
                    n_jobs=model_params.n_jobs) 
            print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], pce_model.num_basis, 
                            data_train.x.shape[1]/pce_model.num_basis))
            print('     - {:<32s} : {}'.format('Y train'    , data_train.y.shape))
            print('     - {:<32s} : {}'.format('Sparsity'   , len(pce_model.active_index)))

            print('   3. Prediction with {} samples '.format(xi_test.shape))
            y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
            data_ideg.model_.append(pce_model)
            # data_ideg.rmse_y.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
            data_ideg.y0_hat_.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
            data_ideg.score_.append(pce_model.score)
            data_ideg.cv_err_.append(pce_model.cv_error)
            # data_ideg.yhat_ecdf_.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
            print('     - {:<32s} : {:.4e}'.format('y0 test [ PCE ]', np.array(data_ideg.y0_hat_[-1])))
            # print('     - {:<32s} : {:.4e}'.format('y0 test [TRUE ]', y0_test))
            isConverge, error_converge = relative_converge(data_ideg.y0_hat_, err=model_params.rel_err)
            # isConverge, error_converge = absolute_converge(data_ideg.y0_hat_, err=model_params.abs_err)
            print('   4. Converge check ...')
            print('      - Value : {}'.format(np.array(data_ideg.y0_hat_)))
            print('      - Error : {} % [{}]'.format(np.around(error_converge, 4)*100,isConverge))
            print('   ------------------------------------------------------------')
            i_iteration +=1
            if np.all(isConverge):
                print('         !< Model converge for order {:d} >!'.format(deg))
                break
            if n_samples_deg > model_params.alpha*orth_poly.num_basis:
                print('         !< Number of samples exceeding {:.2f}P >!'.format(model_params.alpha))
                break

        data_ideg.y0_hat    = data_ideg.y0_hat_[-1]
        data_ideg.cv_err    = data_ideg.cv_err_[-1]
        # data_ideg.kappa   = data_ideg.kappa_[-1]
        data_ideg.model     = data_ideg.model_[-1]
        data_ideg.score     = data_ideg.score_[-1]
        # data_ideg.yhat_ecdf     = data_ideg.yhat_ecdf_[-1]
        if len(data_ideg.DoI_candidate_) == 0:
            data_ideg.DoI_candidate = []
        else:
            data_ideg.DoI_candidate = np.concatenate(data_ideg.DoI_candidate_, axis=1)
        data_ideg.xi_train = np.concatenate(data_ideg.xi_train_, axis=1)
        data_ideg.x_train  = np.concatenate(data_ideg.x_train_ , axis=1)
        data_ideg.y_train  = np.concatenate(data_ideg.y_train_ , axis=0)
        print(' ------------------------------------------------------------')
        tqdm.write(' > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
        # tqdm.write('  - {:<15s} : {:.4e}'.format( 'RMSE y ' , data_ideg.rmse_y[-1]))
        tqdm.write('  - {:<15s} : {:.4e}'.format( 'CV MSE'  , data_ideg.cv_err))
        tqdm.write('  - {:<15s} : {:.4f}'.format( 'Score '  , data_ideg.score ))
        tqdm.write('  - {:<15s} : {:.4e}'.format( 'y0 ' , data_ideg.y0_hat))
        print(' ------------------------------------------------------------')

        main_res.append(data_ideg)

        cv_err_global = np.array([idata.cv_err for idata in main_res]).T
        y0_hat_global = np.array([idata.y0_hat for idata in main_res]).T
        score_global  = np.array([idata.score  for idata in main_res]).T
        isOverfitting(cv_err_global) ## check Overfitting
        isConverge0, error_converge0 = relative_converge(y0_hat_global, err=2*model_params.rel_err)
        # isConverge0, error_converge0 = absolute_converge(data.y0_hat_, err=model_params.abs_err)
        isConverge1, error_converge1 = threshold_converge(score_global)
        error_converge = [error_converge0, error_converge1]
        isConverge = [isConverge0, isConverge1]
        for i, (ikey, ivalue) in enumerate(zip([isConverge0, isConverge1], [error_converge0, error_converge1])):
            print('  >  Checking #{:d} : {}, {:.4f}'.format(i, ikey, ivalue))
        if np.array(isConverge).all():
            tqdm.write('###############################################################################')
            tqdm.write('         Model Converge ')
            tqdm.write('    > Summary PCE: ndim={:d}, p={:d}'.format(ndim, deg))
            # tqdm.write('     - {:<15s} : {}'.format( 'RMSE y ' , np.array(rmse_y)))
            tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'  , np.array(cv_err_global)))
            tqdm.write('     - {:<15s} : {}'.format( 'Score '  , np.array(score_global)))
            tqdm.write('     - {:<15s} : {}'.format( 'y0 ' , np.array(y0_hat_global)))
            for i, (ikey, ivalue) in enumerate(zip(isConverge, error_converge)):
                print('     >  Checking #{:d} : {}, {:.2e}'.format(i, ikey, ivalue))
            tqdm.write('###############################################################################')
            # break
    return main_res

if __name__ == '__main__':
    ## ------------------------ Displaying set up ------------------- ###
    r, theta= 0, 2
    y_scale = 1e6
    ith_batch  = 0
    batch_size = 1
    np.random.seed(100)
    random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=1000)
    np.set_printoptions(suppress=True)
    uqra_env = uqra.environment.NDBC46022()

    eng = matlab.engine.start_matlab()
    out = io.StringIO()
    err = io.StringIO()
    ## ------------------------ Define solver ----------------------- ###
    # solver = uqra.FPSO(random_state=theta, distributions=uqra_env)
    solver = uqra.Solver('RM3', 2, distributions=uqra_env)
    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling('PCE')
    model_params.degs    = np.arange(2,11) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Hem'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLSLAR' 
    model_params.n_splits= 50
    model_params.alpha   = 2
    model_params.num_test= int(1e7)
    model_params.num_pred= int(1e7)
    model_params.pf      = np.array([1.0/(365.25*24*50)])
    model_params.abs_err = 1e-4
    model_params.rel_err = 2.5e-2
    model_params.n_jobs  = mp.cpu_count()
    model_params.channel = 23 # [2, 23, 24, 30]
    model_params.update_basis()
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters('CLS4', 'S')
    # doe_params = uqra.ExperimentParameters('MCS', None)
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
    xi_test = data_test.xi[:, :model_params.num_test] 
    ### 2. Get finished global data
    filename = '{:s}_Adap{:d}{:s}_{:s}E5R{:d}.npy'.format(solver.nickname, 
            solver.ndim, model_params.basis, doe_params.doe_nickname(), r)
    global_data = np.load(os.path.join(data_dir_result, filename), allow_pickle=True).tolist()
    headers  = global_data[0].headers

    res = []
    for i, irepeat in enumerate(range(batch_size*ith_batch, batch_size*(ith_batch+1))):
        print('\n#################################################################################')
        print(' >>>  File: ', __file__)
        print(' >>>  Start UQRA : Theta: {:d}, [{:d}x{:d}]-{:d}'.format(theta, batch_size, ith_batch, i))
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
    if len(res) == 1:
        res = res[0]
    filename = '{:s}_Adap{:d}{:s}_{:s}E5R{:d}S{:d}_y{:d}'.format(solver.nickname, 
            solver.ndim, model_params.basis, doe_params.doe_nickname(), r, theta,
            model_params.channel)
    eng.quit()
    # ## ============ Saving QoIs ============
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))
