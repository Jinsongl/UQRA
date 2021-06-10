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

def threshold_converge(y, threshold=0.9):
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
    ### object contain all training samples
    data_train = uqra.Data()
    data_train.xi_index = []
    data_train.xi = np.empty((model_params.ndim, 0))
    data_train.x  = np.empty((model_params.ndim, 0))
    data_train.y  = np.empty((0,34)) 

    ndim, deg = model_params.ndim, model_params.degs
    max_sparsity  = 30
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
        data_cand = data_cand * deg ** 0.5 if doe_params.doe_sampling.upper() in ['CLS4', 'CLS5'] else data_cand
        print('       {:<23s} : {}'.format(' shape', data_cand.shape))
    else:
        data_cand = None
        print('       {:<23s} : {}'.format(' shape', data_cand))

    idoe_sampling = idoe_params.doe_sampling.lower()
    idoe_nickname = idoe_params.doe_nickname()
    ioptimality   = idoe_params.optimality
    print('     - {:<23s} : {}'.format(' UQRA DoE '  , idoe_nickname))

    print('\n==================================================================================')
    print('         <<<< Initial Exploration: ndim={:d}, p={:d} >>>>'.format(ndim, deg))
    print('==================================================================================\n')
    ## ------------------------ UQRA Surrogate model----------------- ###
    orth_poly = uqra.poly.orthogonal(ndim, deg, model_params.basis)
    pce_model = uqra.PCE(orth_poly)
    dist_u    = model_params.dist_u 
    dist_xi   = orth_poly.weight
    dist_x    = solver.distributions
    pce_model.info()

    ### data object containing results from intermedia steps
    ## attribute ending with '_' is a collection of variables after each iteration
    data_iteration = uqra.Data()
    data_iteration.ndim         = ndim 
    data_iteration.deg          = deg 
    data_iteration.y0_hat_      = []
    data_iteration.cv_err_      = []
    data_iteration.model_       = []
    data_iteration.score_       = []
    data_iteration.DoI_xi_      = []
    data_iteration.DoI_x_       = []
    data_iteration.exploration0 = []  ## initial exploration sample set
    data_iteration.exploration_ = []  ## exploration sample sets added later
    data_iteration.exploitation_= []  ## exploitation sample sets added later
    data_iteration.deg_converge = False
    data_iteration.iteration_converge  = False
    data_QoIs_iteration = [copy.deepcopy(data_iteration) for _ in range(34)] 
    ### ------------------------ #1: Obtain exploration optimal samples ----------------- ###
    print(' ------------------------------------------------------------')
    print('   > Adding exploration samples in global domain... ')
    print('   1. optimal samples based on FULL basis: {:s}'.format(idoe_nickname))
    ## obtain exploration optimal samples
    n_samples = max_sparsity 
    xi_exploration0, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=[], 
            active_index=None, initialization='RRQR', return_index=True) 

    # samples from evaluated global_data 
    x_exploration0 = solver.map_domain(xi_exploration0, dist_xi)
    ii = np.where(np.array([iglobal_data.deg for iglobal_data in global_data]) == deg)[0][0]
    iglobal_data = global_data[ii]
    ## make sure train samples are same
    if np.amax(abs(xi_exploration0-iglobal_data.xi_train[:,:n_samples])) > 1e-6  :
        print( ' Train samples are not same! max error: {:.2e}'.format(
            np.amax(abs(xi_exploration0-iglobal_data.xi_train[:,:n_samples]))))
        print('  xi train from saved data : \n{}'.format(iglobal_data.xi_train[:,:n_samples]))
        print('  xi train from current DoE: \n{}'.format(xi_exploration0))
    if np.amax(abs(x_exploration0-iglobal_data.x_train[:,:n_samples])) > 1e-6  :
        print( ' Train samples are not same! max error: {:.2e}'.format(
            np.amax(abs(x_exploration0-iglobal_data.x_train[:,:n_samples]))))
        print('  x train from saved data : \n{}'.format(iglobal_data.x_train[:,:n_samples]))
        print('  x train from current DoE: \n{}'.format(x_exploration0))

    y_exploration0 = iglobal_data.y_train[:n_samples,:,theta] ## shape (nsample, nQoIs, n_short_term)

    data_exploration0 = uqra.Data()
    data_exploration0.xi= xi_exploration0
    data_exploration0.x = x_exploration0
    data_exploration0.y = y_exploration0
    for iqoi in model_params.channel:
        data_QoIs_iteration[iqoi].exploration0= data_exploration0

    n_samples_deg  = n_samples 
    data_train.xi  = np.concatenate([data_train.xi, xi_exploration0], axis=1)
    data_train.x   = np.concatenate([data_train.x , x_exploration0 ], axis=1)
    data_train.y   = np.concatenate([data_train.y , y_exploration0 ], axis=0)
    data_train.xi_index = uqra.list_union(data_train.xi_index, idx_optimal)
    print('     - {:<32s} : {:d}'.format('Adding exploration optimal samples', n_samples))
    print('     - {:<32s} : {:d}'.format('No. optimal samples [p='+str(deg)+']', n_samples_deg))
    print('     - {:<32s} : {:.2f}'.format('Local oversampling [p='+str(deg)+']', n_samples_deg/pce_model.num_basis))
    print('     - {:<32s} : {:d}'.format('Total number of samples', len(data_train.y)))

    print('   2. Training PCE (p={:d}) model with {} '.format(deg, model_params.fitting))
    assert np.array_equal( data_train.x, solver.map_domain(data_train.xi, dist_xi))
    print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], 
        pce_model.num_basis, data_train.x.shape[1]/pce_model.num_basis))
    print('     - {:<32s} : {}'.format('Y train'    , data_train.y.shape))
    weight  = doe_params.sampling_weight()   ## weight function
    for iqoi in model_params.channel:
        print('     {:<20s}, prediction samples: {}'.format(headers[iqoi], xi_test.shape))
        pce_model = uqra.PCE(orth_poly)
        pce_model.fit(model_params.fitting, data_train.xi, data_train.y[:, iqoi]/model_params.y_scales[iqoi], 
                w=weight,n_jobs=model_params.n_jobs) 
        data_QoIs_iteration[iqoi].sparsity = len(pce_model.active_index)
        max_sparsity = max(max_sparsity, data_QoIs_iteration[iqoi].sparsity)
        y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
        data_QoIs_iteration[iqoi].y_test_hat = y_test_hat
        data_QoIs_iteration[iqoi].model_.append(pce_model)
        data_QoIs_iteration[iqoi].y0_hat_.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
        data_QoIs_iteration[iqoi].score_.append(pce_model.score)
        data_QoIs_iteration[iqoi].cv_err_.append(pce_model.cv_error)
        print('     - Sparsity={:<2d}, y0 test[PCE]: {:.4e}'.format(data_QoIs_iteration[iqoi].sparsity, 
                np.array(data_QoIs_iteration[iqoi].y0_hat_[-1])))
    n_samples_deg = n_samples
    i_iteration = 1
    while i_iteration <= 20:
        print('                 ------------------------------')
        print('                    <  Iteration No. {:d} >'.format(i_iteration))
        print('                 ------------------------------')
        print(' ------------------------------------------------------------')
        print('   > Adding exploration optimal samples in global domain ... ')
        print('   1-1. optimal samples based on SIGNIFICANT basis in global domain ... ')
        ####-------------------------------------------------------------------------------- ####
        n_samples = min(3, max(3,max_sparsity)) 
        # min(max_sparsity, model_params.alpha *pce_model.num_basis - n_samples_deg, 5)
        # n_samples = min(10, max_sparsity) #len(active_index)
        xi_exploration, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=data_train.xi_index, 
                active_index=pce_model.active_index, initialization='RRQR', return_index=True) 
        assert xi_exploration.shape[1] == n_samples ## make sure return number of samples required
        x_exploration = solver.map_domain(xi_exploration, dist_xi)
        eng.workspace['deg']       = float(deg)
        eng.workspace['phaseSeed'] = float(theta)
        y_exploration = []
        for iHs, iTp in tqdm(x_exploration.T, ncols=80, desc='     - [WEC-SIM]' ):
            eng.workspace['Hs'] = float(iHs)
            eng.workspace['Tp'] = float(iTp)
            # eng.wecSim(nargout=0)
            eng.wecSim(nargout=0,stdout=out,stderr=err)
            y_exploration.append(np.squeeze(eng.workspace['maxima'])[2:]) ## first two are Hs,Tp
        y_exploration = np.array(y_exploration)

        ## save exploration data
        data_exploration   = uqra.Data()
        data_exploration.xi= xi_exploration
        data_exploration.x = x_exploration
        data_exploration.y = y_exploration
        for iqoi in model_params.channel:
            data_QoIs_iteration[iqoi].exploration_.append(data_exploration)
        n_samples_deg += n_samples

        data_train.xi  = np.concatenate([data_train.xi, xi_exploration], axis=1)
        data_train.x   = np.concatenate([data_train.x , x_exploration ], axis=1)
        data_train.y   = np.concatenate([data_train.y , y_exploration ], axis=0)
        data_train.xi_index = uqra.list_union(data_train.xi_index, idx_optimal)


        print('   > exploitation step (SIGNIFICANT basis)... ')

        ## obtain DoI candidate samples from each QoI
        for iqoi in model_params.channel:
            print('     - {:<32s} : {:s}'.format('Domain of Interest (DoI)', headers[iqoi] ))
            print('     - {:<32s} : {}'.format('Iteration Converge', data_QoIs_iteration[iqoi].iteration_converge))
            ## obtain candidate samples for each QoI
            # data_cand_DoI_iqoi, idx_data_cand_DoI = idoe_params.samples_nearby(data_QoIs_iteration[iqoi].y0_hat_[-1], 
                    # xi_test, data_QoIs_iteration[iqoi].y_test_hat, data_cand, deg, n0=10, epsilon=0.1, return_index=True)
            if data_QoIs_iteration[iqoi].iteration_converge:
                print('     - Skip ')
                continue
            else:
                pass

            data_cand_DoI_iqoi = idoe_params.domain_of_interest(data_QoIs_iteration[iqoi].y0_hat_[-1], xi_test, 
                    data_QoIs_iteration[iqoi].y_test_hat, n_centroid=20, epsilon=0.1)

            data_QoIs_iteration[iqoi].DoI_xi_.append(data_cand_DoI_iqoi)
            data_QoIs_iteration[iqoi].DoI_x_.append(solver.map_domain(data_cand_DoI_iqoi, dist_xi ))
            print('     - {:<32s} : {}  '.format('DoI candidate samples', data_cand_DoI_iqoi.shape ))
            ## get optimal samples for each QoI
            print('     - {:<32s} : {:d}'.format('Adding DoI optimal samples', n_samples ))
            xi_exploitation, idx_optimal_DoI = idoe_params.get_samples(data_cand_DoI_iqoi, orth_poly, n_samples, x0=[], 
                    active_index= data_QoIs_iteration[iqoi].model_[-1].active_index, initialization='RRQR', return_index=True) 
            assert xi_exploitation.shape[1] == n_samples ## make sure return number of samples required

            x_exploitation = solver.map_domain(xi_exploitation, dist_xi)
            eng.workspace['deg']       = float(deg)
            eng.workspace['phaseSeed'] = float(theta)
            y_exploitation = []
            for iHs, iTp in tqdm(x_exploitation.T, ncols=80, desc='     - [WEC-SIM]' ):
                eng.workspace['Hs'] = float(iHs)
                eng.workspace['Tp'] = float(iTp)
                # eng.wecSim(nargout=0)
                eng.wecSim(nargout=0,stdout=out,stderr=err)
                y_exploitation.append(np.squeeze(eng.workspace['maxima'])[2:]) ## first two are Hs,Tp
            y_exploitation = np.array(y_exploitation)

            ## save exploitation data
            data_exploitation   = uqra.Data()
            data_exploitation.xi= xi_exploitation
            data_exploitation.x = x_exploitation
            data_exploitation.y = y_exploitation
            data_QoIs_iteration[iqoi].exploitation_.append(data_exploitation)

            ## save all training samples together
            data_train.xi  = np.concatenate([data_train.xi, xi_exploitation], axis=1)
            data_train.x   = np.concatenate([data_train.x , x_exploitation ], axis=1)
            data_train.y   = np.concatenate([data_train.y , y_exploitation ], axis=0)
            data_train.xi_index = uqra.list_union(data_train.xi_index, idx_optimal)

            n_samples_deg += n_samples


        print('   3. training PCE (p={:d}) model with {} '.format(deg, model_params.fitting))
        print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], 
            pce_model.num_basis, data_train.x.shape[1]/pce_model.num_basis))
        print('     - {:<32s} : {}'.format('Y train'    , data_train.y.shape))
        for iqoi in model_params.channel:
            print('     > {:<20s}, prediction samples: {}'.format(headers[iqoi], xi_test.shape))
            pce_model = uqra.PCE(orth_poly)
            weight  = doe_params.sampling_weight()   ## weight function
            pce_model.fit(model_params.fitting, data_train.xi, data_train.y[:, iqoi]/model_params.y_scales[iqoi], 
                    w=weight, n_jobs=model_params.n_jobs) 

            data_QoIs_iteration[iqoi].sparsity = len(pce_model.active_index)
            max_sparsity = max(max_sparsity, data_QoIs_iteration[iqoi].sparsity)
            y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)

            data_QoIs_iteration[iqoi].y_test_hat = y_test_hat
            data_QoIs_iteration[iqoi].model_.append(pce_model)
            data_QoIs_iteration[iqoi].y0_hat_.append(uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf))
            data_QoIs_iteration[iqoi].score_.append(pce_model.score)
            data_QoIs_iteration[iqoi].cv_err_.append(pce_model.cv_error)

            data_QoIs_iteration[iqoi].cv_err = pce_model.cv_error
            data_QoIs_iteration[iqoi].score  = pce_model.score
            data_QoIs_iteration[iqoi].model  = pce_model
            data_QoIs_iteration[iqoi].y0_hat = uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf)
            print('     - Sparsity={:<2d}, y0 test[PCE]: {:.4e}'.format(data_QoIs_iteration[iqoi].sparsity, 
                np.array(data_QoIs_iteration[iqoi].y0_hat_[-1])))
        print('   4. converge check ...')
        is_QoIs_converge = [] 
        for iqoi in model_params.channel:
            is_y0_converge   , y0_converge_err = relative_converge(data_QoIs_iteration[iqoi].y0_hat_, err=2*model_params.rel_err)
            is_score_converge, score_converge  = threshold_converge(data_QoIs_iteration[iqoi].score_)
            data_QoIs_iteration[iqoi].iteration_converge = is_y0_converge and is_score_converge
            is_QoIs_converge.append([is_y0_converge, is_score_converge])
            print('  >  QoI: {:<25s}'.format(headers[iqoi]))
            print('     >  Values: {}'.format(np.array(data_QoIs_iteration[iqoi].y0_hat_)))
            print('     >  Rel Error [%]: {:5.2f}, Converge: {}'.format(y0_converge_err*100, is_y0_converge     ))
            print('     >  Fit Score [%]: {:5.2f}, Converge: {}'.format(score_converge *100, is_score_converge  ))
            print('     -------------------------------------------')

        i_iteration +=1
        if np.all(is_QoIs_converge):
            print('         !< Model converge for order {:d} >!'.format(deg))
            break
        if n_samples_deg > model_params.alpha*orth_poly.num_basis:
            print('     PCE(d={:d},p={:d}) !< Number of samples exceeding {:.2f}P >!'.format(
                ndim, deg, model_params.alpha))
            break

        for iqoi in model_params.channel:
            del data_QoIs_iteration[iqoi].y_test_hat
        data_QoIs.append(data_QoIs_iteration)
        print('--------------------------------------------------')
        print('     Model Performance up to order p={:d}'.format(deg))
        is_QoIs_converge = [] 
        for iqoi in model_params.channel:
            iheader   = headers  [iqoi]
            data_iqoi = [data_QoIs_iteration[iqoi] for data_QoIs_iteration in data_QoIs]
            cv_err_iqoi_degs = np.array([idata.cv_err for idata in data_iqoi]).T
            y0_hat_iqoi_degs = np.array([idata.y0_hat for idata in data_iqoi]).T
            score_iqoi_degs  = np.array([idata.score  for idata in data_iqoi]).T
            is_overfit       , overfit_vals    = overfitting_check(cv_err_iqoi_degs) ## check Overfitting
            is_y0_converge   , y0_converge_err = relative_converge(y0_hat_iqoi_degs, err=model_params.rel_err)
            is_score_converge, score_converge  = threshold_converge(score_iqoi_degs)
            is_QoIs_converge.append([is_y0_converge, is_score_converge])

            data_QoIs[-1][iqoi].deg_overfit  = is_overfit 
            data_QoIs[-1][iqoi].deg_converge = is_y0_converge and  is_score_converge
            print('  >  QoI: {:<25s}'.format(iheader))
            print('     >  Values: {}'.format(np.array(y0_hat_iqoi_degs)))
            print('     >  Overfit : {}, Converge: {}'.format(overfit_vals, is_overfit))
            print('     >  Rel Error [%]: {:5.2f}, Converge: {}'.format(y0_converge_err*100, is_y0_converge     ))
            print('     >  Fit Score [%]: {:5.2f}, Converge: {}'.format(score_converge *100, is_score_converge  ))
        print('--------------------------------------------------')

        if np.array(is_QoIs_converge).all():
            tqdm.write('###############################################################################')
            tqdm.write('         Model Converge in Polynomial total orders ')
            tqdm.write('    > Final PCE: ndim={:d}, p={:d}'.format(ndim, deg))
            # tqdm.write('     - {:<15s} : {}'.format( 'RMSE y ' , np.array(rmse_y)))
            # tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'  , np.array(cv_err_global)))
            # tqdm.write('     - {:<15s} : {}'.format( 'Score '  , np.array(score_global)))
            # tqdm.write('     - {:<15s} : {}'.format( 'y0 ' , np.array(y0_hat_global)))
            tqdm.write('###############################################################################')

    return data_QoIs

if __name__ == '__main__':
    ## ------------------------ Displaying set up ------------------- ###
    r, theta   = 0, 5
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
    solver = uqra.Solver('RM3', 2, distributions=uqra_env)
    ## ------------------------ UQRA Modeling Parameters ----------------- ###
    model_params = uqra.Modeling('PCE')
    model_params.degs    = 10 #np.arange(2,10) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Hem'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLSLAR' 
    model_params.n_splits= 10
    model_params.alpha   = 3
    model_params.num_test= int(1e7)
    model_params.pf      = np.array([1.0/(365.25*24*50)])
    model_params.abs_err = 1e-4
    model_params.rel_err = 2.5e-2
    model_params.n_jobs  = mp.cpu_count()
    model_params.channel = [2, 23, 24, 30]
    model_params.y_scales= np.zeros(34)
    model_params.y_scales[model_params.channel]= [1, 1e6, 1e7, 1e6]
    model_params.update_basis()
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters('CLS4', 'S')
    doe_params.update_poly_name(model_params.basis)
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
    ## 2. Get finished global data
    filename = '{:s}_Adap{:d}{:s}_{:s}E5R{:d}_global.npy'.format(solver.nickname, 
            solver.ndim, model_params.basis[:3], doe_params.doe_nickname(), r)
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
    filename = '{:s}_Adap{:d}{:s}_{:s}E5R{:d}S{:d}'.format(solver.nickname, 
            solver.ndim, model_params.basis, doe_params.doe_nickname(), r, theta)
    eng.quit()
    # ## ============ Saving QoIs ============
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))
