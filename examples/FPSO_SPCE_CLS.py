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
import datetime
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
    ### object contain all training samples
    data_train = uqra.Data()
    data_train.xi_index = []
    data_train.xi = np.empty((model_params.ndim, 0))
    data_train.x  = np.empty((model_params.ndim, 0))
    data_train.y  = np.empty((0,))

    ndim, deg = model_params.ndim, model_params.degs
    sparsity  = 30
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
        np.save('FPSO_CLS4_cand_test.npy', data_cand)
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
    data_iteration = uqra.Data()
    data_iteration.ndim          = ndim
    data_iteration.deg           = deg 
    data_iteration.y0_hat_       = []
    data_iteration.cv_err_       = []
    data_iteration.rmse_y_       = []
    data_iteration.model_        = []
    data_iteration.score_        = []
    data_iteration.yhat_ecdf_    = []
    data_iteration.xi_train_     = []
    data_iteration.x_train_      = []
    data_iteration.y_train_      = []
    data_iteration.exploration0  = []  ## initial exploration sample set
    data_iteration.exploration_  = []  ## exploration sample sets added later
    data_iteration.exploitation_ = []  ## exploitation sample sets added later
    data_iteration.DoI_candidate_= []
    data_iteration.is_converge=[]

    ### ------------------------ #1: Obtain exploration optimal samples ----------------- ###
    print(' ------------------------------------------------------------')
    print('   > Adding exploration samples in global domain... ')
    print('   1. optimal samples based on FULL basis: {:s}'.format(idoe_nickname))
    ## obtain exploration optimal samples
    n_samples = sparsity #max(sparsity, math.ceil(0.8*pce_model.num_basis))
    xi_exploration0, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=[], 
            active_index=None, initialization='RRQR', return_index=True) 

    x_exploration0 = solver.map_domain(xi_exploration0, dist_xi)
    y_exploration0 = solver.run(x_exploration0)

    data_exploration0 = uqra.Data()
    data_exploration0.xi= xi_exploration0
    data_exploration0.x = x_exploration0
    data_exploration0.y = y_exploration0
    data_iteration.exploration0 = data_exploration0

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
    print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], 
        pce_model.num_basis, data_train.x.shape[1]/pce_model.num_basis))
    print('     - {:<32s} : {}'.format('Y train'    , data_train.y.shape))
    weight  = doe_params.sampling_weight()   ## weight function
    pce_model.fit(model_params.fitting, data_train.xi, data_train.y, w=weight, n_jobs=model_params.n_jobs) 
    sparsity = len(pce_model.active_index)
    print('     - {:<32s} : {}'.format('Sparsity'   , sparsity))

    print('   3. Prediction with {} samples '.format(xi_test.shape))
    y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
    data_iteration.model_.append(pce_model)
    data_iteration.rmse_y_.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
    data_iteration.y0_hat_.append(uqra.metrics.mquantiles(np.concatenate([y_test_hat, y_outside], axis=0), 1-model_params.pf))
    data_iteration.score_.append(pce_model.score)
    data_iteration.cv_err_.append(pce_model.cv_error)
    data_iteration.yhat_ecdf_.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
    print('     - {:<32s} : {:.4e}'.format('y0 test [ PCE ]', np.array(data_iteration.y0_hat_[-1])))
    print('     - {:<32s} : {:.4e}'.format('y0 test [TRUE ]', y0_test))

    i_iteration = 1
    while i_iteration <= 20:
        print('                 ------------------------------')
        print('                    <  Iteration No. {:d} >'.format(i_iteration))
        print('                 ------------------------------')
        print(' ------------------------------------------------------------')
        print('   > Adding exploration optimal samples in global domain ... ')
        print('   1-1. optimal samples based on SIGNIFICANT basis in global domain ... ')
        ####-------------------------------------------------------------------------------- ####
        n_samples = min(5, max(3,sparsity)) #math.ceil(2*sparsity/3) 
        #min(sparsity, model_params.alpha *pce_model.num_basis - n_samples_deg, 5)
        # n_samples = min(10, sparsity) #len(active_index)
        xi_exploration, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=data_train.xi_index, 
                active_index=pce_model.active_index, initialization='RRQR', return_index=True) 
                # active_index=None, initialization='RRQR', return_index=True) 

        ## obtain exploration optimal samples
        x_exploration = solver.map_domain(xi_exploration, dist_xi)
        y_exploration = solver.run(x_exploration)

        ## save exploration data
        data_exploration   = uqra.Data()
        data_exploration.xi= xi_exploration
        data_exploration.x = x_exploration
        data_exploration.y = y_exploration
        data_iteration.exploration_.append(data_exploration)
        n_samples_deg  += n_samples 

        data_train.xi  = np.concatenate([data_train.xi, xi_exploration], axis=1)
        data_train.x   = np.concatenate([data_train.x , x_exploration ], axis=1)
        data_train.y   = np.concatenate([data_train.y , y_exploration ], axis=0)
        data_train.xi_index = uqra.list_union(data_train.xi_index, idx_optimal)
        print('     - {:<32s} : {:d}  '.format('Adding exploration optimal samples', n_samples))
        print('     - {:<32s} : {:d}  '.format('No. optimal samples [p='+str(deg)+']', n_samples_deg))
        print('     - {:<32s} : {:.2f}'.format('Local oversampling  [p='+str(deg)+']', n_samples_deg/pce_model.num_basis))
        print('     - {:<32s} : {:d}  '.format('Total number of samples', len(data_train.y)))


        print('   1-2. optimal samples based on SIGNIFICANT basis in domain of interest... ')

        ## obtain DoI candidate samples
        # n_samples = min(10, max(3,sparsity)) #math.ceil(2*sparsity/3) 
        # DoI_candidate_xi, idx_data_cand_DoI = idoe_params.samples_nearby(data_iteration.y0_hat_[-1], 
                # xi_test, y_test_hat, data_cand , deg, n0=10, epsilon=0.1, return_index=True)

        n_samples = min(10, max(3,sparsity)) #math.ceil(2*sparsity/3) 
        DoI_candidate_xi = idoe_params.domain_of_interest(data_iteration.y0_hat_[-1], xi_test, y_test_hat, 
                n_centroid=5, epsilon=0.1)
        DoI_candidate_x  = solver.map_domain(DoI_candidate_xi, dist_xi)
        DoI_candidate = uqra.Data()
        DoI_candidate.xi = DoI_candidate_xi
        DoI_candidate.x  = DoI_candidate_x
        data_iteration.DoI_candidate_.append(DoI_candidate)

        ## obtain DoI optimal samples
        xi_exploitation, idx_optimal_DoI = idoe_params.get_samples(DoI_candidate_xi, orth_poly, n_samples, x0=[], 
                active_index=pce_model.active_index, initialization='RRQR', return_index=True) 
        assert xi_exploitation.shape[1] == n_samples ## make sure return number of samples required
        x_exploitation = solver.map_domain(xi_exploitation, dist_xi)
        y_exploitation = np.array(solver.run(x_exploitation), ndmin=1)

        ## save exploitation data
        data_exploitation   = uqra.Data()
        data_exploitation.xi= xi_exploitation
        data_exploitation.x = x_exploitation
        data_exploitation.y = y_exploitation

        data_iteration.exploitation_.append(data_exploitation)
        n_samples_deg += n_samples

        ## put all training samples together, up to current step
        data_train.xi  = np.concatenate([data_train.xi, xi_exploitation], axis=1)
        data_train.x   = np.concatenate([data_train.x , x_exploitation ], axis=1)
        data_train.y   = np.concatenate([data_train.y , y_exploitation ], axis=0)
        data_train.xi_index = uqra.list_union(data_train.xi_index, idx_optimal_DoI)

        print('     - {:<32s} : {}  '.format('DoI candidate samples', DoI_candidate_xi.shape ))
        print('     - {:<32s} : {:d}'.format('Adding DoI optimal samples', n_samples))
        print('     - {:<32s} : {:d}'.format('No. samples [p='+str(deg)+']', n_samples_deg))
        print('     - {:<32s} : {:.2f}'.format('Local oversampling [p='+str(deg)+']', n_samples_deg/pce_model.num_basis))
        print('     - {:<32s} : {:d}'.format('Total number of samples', len(data_train.y)))

        print('   2. Training PCE (p={:d}) model with {} '.format(deg, model_params.fitting))
        print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], 
            pce_model.num_basis, data_train.x.shape[1]/pce_model.num_basis))
        print('     - {:<32s} : {}'.format('Y train'    , data_train.y.shape))
        weight  = doe_params.sampling_weight()   ## weight function
        pce_model.fit(model_params.fitting, data_train.xi, data_train.y, w=weight, n_jobs=model_params.n_jobs) 
        sparsity= len(pce_model.active_index)
        print('     - {:<32s} : {}'.format('Sparsity'   , sparsity))

        print('   3. Prediction with {} samples '.format(xi_test.shape))
        y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
        data_iteration.model_.append(pce_model)
        data_iteration.rmse_y_.append(uqra.metrics.mean_squared_error(y_test, y_test_hat, squared=False))
        data_iteration.y0_hat_.append(uqra.metrics.mquantiles(np.concatenate([y_test_hat, y_outside], axis=0), 1-model_params.pf))
        data_iteration.score_.append(pce_model.score)
        data_iteration.cv_err_.append(pce_model.cv_error)
        data_iteration.yhat_ecdf_.append(uqra.ECDF(y_test_hat, model_params.pf, compress=True))
        # isOverfitting(data_iteration.cv_err) ## check Overfitting
        print('     - {:<32s} : {:.4e}'.format('y0 test [ PCE ]', np.array(data_iteration.y0_hat_[-1])))
        print('     - {:<32s} : {:.4e}'.format('y0 test [TRUE ]', y0_test))

        isConverge, error_converge = relative_converge(data_iteration.y0_hat_, err=model_params.rel_err)
        is_y0_converge   , y0_converge_err = relative_converge(data_iteration.y0_hat_, err=2*model_params.rel_err)
        is_score_converge, score_converge  = threshold_converge(data_iteration.score_)
        print('   4. Converge check ...')
        print('      - Value : {} [Ref: {:e}]'.format(np.array(data_iteration.y0_hat_), y0_test))
        print('      - Rel Error [%]: {:.2f}, Converge: {}'.format(y0_converge_err*100, is_y0_converge     ))
        print('      - Fit Score [%]: {:.2f}, Converge: {}'.format(score_converge*100 , is_score_converge  ))
        print('   ------------------------------------------------------------')

        data_iteration.is_converge.append(is_y0_converge and is_score_converge)
        i_iteration +=1
        # if is_y0_converge and is_score_converge:
            # print('         !< Model converge for order {:d} >!'.format(deg))
            # break
        if n_samples_deg > model_params.alpha*orth_poly.num_basis:
            print('         !< Number of samples exceeding {:.2f}P >!'.format(model_params.alpha))
            break


    data_iteration.y0_hat    = data_iteration.y0_hat_[-1]
    data_iteration.cv_err    = data_iteration.cv_err_[-1]
    # data_iteration.kappa   = data_iteration.kappa_[-1]
    data_iteration.model     = data_iteration.model_[-1]
    data_iteration.score     = data_iteration.score_[-1]
    data_iteration.yhat_ecdf = data_iteration.yhat_ecdf_[-1]
    data_iteration.rmse_y    = data_iteration.rmse_y_[-1]
    data_iteration.train_data= data_train
    data_iteration.xi_train  = data_train.xi 
    data_iteration.x_train   = data_train.x 
    data_iteration.y_train   = data_train.y 
    print(' ------------------------------------------------------------')
    tqdm.write(' > Final PCE: ndim={:d}, p={:d}, Iteration={:d}'.format(ndim, deg, i_iteration))
    tqdm.write('  - {:<15s} : {:.4e}'.format( 'RMSE y ' , data_iteration.rmse_y))
    tqdm.write('  - {:<15s} : {:.4e}'.format( 'CV MSE'  , data_iteration.cv_err))
    tqdm.write('  - {:<15s} : {:.4f}'.format( 'Score '  , data_iteration.score ))
    tqdm.write('  - {:<15s} : {:.4e} [{:.4e}]'.format( 'QoI[y0] ' , data_iteration.y0_hat, y0_test))
    print(' ------------------------------------------------------------')
    return data_iteration

if __name__ == '__main__':
    ## ------------------------ Displaying set up ------------------- ###
    r, theta= 0, 0
    ith_batch  = 0
    batch_size = 1
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
    model_params.degs    = 10 #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Hem'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLSLAR' 
    model_params.n_splits= 20
    model_params.alpha   = 5
    model_params.num_test= int(1e7)
    model_params.pf      = np.array([0.5/(365.25*24*50)])
    model_params.abs_err = 1e-4
    model_params.rel_err = 2.5e-2
    model_params.n_jobs  = mp.cpu_count()
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


    data_test.y = solver.run(data_test.x) if not hasattr(data_test, 'y') else data_test.y
    try:
        data_test.y = data_test.y[theta]
    except:
        pass
    assert data_test.xi.shape[1] >= model_params.num_test 
    xi_test = data_test.xi[:, :model_params.num_test] 
    y_test  = data_test.y [   :model_params.num_test] 
    y0_test = uqra.metrics.mquantiles(y_test, 1-model_params.pf)

    np.save('FPSO_test_x.npy' , data_test.x)
    np.save('FPSO_test_xi.npy', data_test.xi)
    np.save('FPSO_test_y.npy' , y_test)

    print('U:', data_test.u.shape, np.mean(data_test.u, axis=1), np.std(data_test.u, axis=1))
    print(np.amin(data_test.u, axis=1), np.amax(data_test.u, axis=1))
    print('Xi:', data_test.xi.shape, np.mean(data_test.xi, axis=1), np.std(data_test.xi, axis=1))
    print(np.amin(data_test.xi, axis=1), np.amax(data_test.xi, axis=1))
    print('Xi Test:', xi_test.shape, np.mean(xi_test, axis=1), np.std(xi_test, axis=1))
    print(np.amin(xi_test, axis=1), np.amax(xi_test, axis=1))
    idx_outside = np.linalg.norm(xi_test, axis=0) > np.sqrt(model_params.degs * 2)
    idx_inside  = np.linalg.norm(xi_test, axis=0) <=np.sqrt(model_params.degs * 2)
    xi_test_outside = xi_test[:, idx_outside]
    y_outside = y_test[idx_outside] 
    xi_test   = xi_test[:, idx_inside]
    y_test    = y_test[idx_inside]
    print(xi_test_outside.shape)
    print('X:', data_test.x.shape, np.mean(data_test.x, axis=1), np.std(data_test.x, axis=1))
    print(np.amin(data_test.x, axis=1), np.amax(data_test.x, axis=1))

    res = []
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
    if len(res) == 1:
        res = res[0]
    filename = '{:s}_Adap{:d}{:s}_{:s}E5R{:d}S{:d}_{:d}{:d}'.format(solver.nickname, 
            solver.ndim, model_params.basis, doe_params.doe_nickname(), r, theta,
            batch_size, ith_batch)
    # ## ============ Saving QoIs ============
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))

    # alarm_window = [datetime.time(7,0,0),datetime.time(23,59,59)]
    # current_time = datetime.datetime.now().time()
    # if uqra.time_in_range(alarm_window[0], alarm_window[1], current_time) :
        # os.system('say "{:s} Simulation has finished"'.format(solver.name))
        
