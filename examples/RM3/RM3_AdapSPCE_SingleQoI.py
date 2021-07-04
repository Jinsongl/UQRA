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

def main(model_params, doe_params, solver, r=0, random_state=None, theta=None):
    random.seed(random_state)
    ## ------------------------ Initialize parameters ----------------- ###
    ndim, deg = model_params.ndim, model_params.degs[0]
    max_sparsity  = 6  ## initialize n_samples
    ndim_deg_cases  = np.array(list(itertools.product([model_params.ndim,], model_params.degs)))


    ### data object containing results from intermedia steps
    ## attribute ending with '_' is a collection of variables after each iteration
    data_init = uqra.Data()
    data_init.ndim         = ndim 
    data_init.deg          = deg 
    data_init.y0_hat_      = []
    data_init.cv_err_      = []
    data_init.model_       = []
    data_init.score_       = []
    data_init.DoI_xi_      = []
    data_init.DoI_x_       = []
    data_init.data_train_  = []
    data_init.exploration0 = None## initial exploration sample set
    data_init.exploration_ = []  ## exploration sample sets added later
    data_init.exploitation_= []  ## exploitation sample sets added later
    data_init.deg_converge = False
    data_init.deg_overfit  = False
    data_init.iteration_converge = False
    ## ------------------------ list of Data obj for all QoIs ----------------- ###
    ## data object while building p-order PCE iteratively
    ## attribute ending with '_' is a collection of variables after each iteration
    data_QoIs = [copy.deepcopy(data_init) for _ in range(34)] 
    data_degs_QoIs = [data_QoIs for _ in range(model_params.degs[-1])] ## nested list, [data_ideg_QoIs[data_iqoi_ideg]]   34 outputs in total



    for iqoi in model_params.channel:
        ### object contain all training samples
        data_train = uqra.Data()
        data_train.xi_index = []
        data_train.xi = np.empty((model_params.ndim, 0))
        data_train.x  = np.empty((model_params.ndim, 0))
        data_train.y  = np.empty((0,34)) 
        while deg < model_params.degs[-1]:
            print('\n==================================================================================')
            print('         <<<< UQRA Sparse PCE Model: ndim={:d}, p={:d} >>>>'.format(ndim, deg))
            print('            < QoI: {:s} >'.format(headers[iqoi]))
            print('==================================================================================\n')

            data_ideg_QoIs = data_degs_QoIs[deg] ## list of uqra.Data()
            ## data_ideg_QoIs was assigned before: overfitting occurs for some QoIs
            ## new resutls will be appended to current results for order p = deg
            ## However, for higher order models, results will be cleared

            if data_ideg_QoIs[iqoi].deg_overfit:
                ## clear results for all higher order
                for ideg in range(deg+1, model_params.degs[-1]):
                    ## data_degs_QoIs[ideg] either empty list (not reach this order yet) or list of 34 uqra.Data()
                    data_degs_QoIs[ideg][iqoi] = data_init
            ## ------------------------ Updating DoE parameters ----------------- ###
            idoe_params = copy.deepcopy(doe_params)
            idoe_params.ndim = ndim
            idoe_params.deg  = int(deg)
            ### Specify candidate data filename template function
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
            print('     - {:<23s} : {}'.format(' UQRA DoE '  ,idoe_params.doe_nickname()))

            ## ------------------------ UQRA Surrogate model----------------- ###
            orth_poly = uqra.poly.orthogonal(ndim, deg, model_params.basis)
            dist_xi   = orth_poly.weight

            ### ------------------------ #1: Obtain exploration optimal samples ----------------- ###
            print(' ------------------------------------------------------------')
            print('   Initial exploration optimal samples in global domain based on FULL basis: {:s}'.format(idoe_params.doe_nickname()))
            print(' ------------------------------------------------------------')
            print('   > Adding exploration samples in global domain... ')
            ### optimal samples are available in global_data. Optimal Designs are still process here to confirm those samples are correct
            # samples from optimal design
            n_samples = max(max_sparsity, math.ceil(0.8*orth_poly.num_basis))
            xi_exploration0, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=[], 
                    active_index=None, initialization='RRQR', return_index=True) 
            print('     - {:<32s} : {:d}'.format('Adding exploration optimal samples', n_samples))
            ### some samples are cached for quick access. 
            x_exploration0 = solver.map_domain(xi_exploration0, dist_xi)
            ii = np.where(np.array([iglobal_data.deg for iglobal_data in global_data]) == deg)[0][0]
            iglobal_data = global_data[ii]
            # if samples are evaluated before, use those directly 
            if np.amax(abs(xi_exploration0-iglobal_data.xi_train[:,:n_samples])) > 1e-6 \
                    or np.amax(abs(x_exploration0-iglobal_data.x_train[:,:n_samples])) > 1e-6 :
                print('   - x/xi values in cached data not match, running WEC-Sim ...')
                print(xi_exploration0)
                print(iglobal_data.xi_train[:,:n_samples])
                eng.workspace['deg']       = float(deg)
                eng.workspace['phaseSeed'] = float(theta)
                y_exploration0 = []
                for iHs, iTp in tqdm(x_exploration0.T, ncols=80, desc='     - [WEC-SIM]' ):
                    eng.workspace['Hs'] = float(iHs)
                    eng.workspace['Tp'] = float(iTp)
                    # eng.wecSim(nargout=0)
                    eng.wecSim(nargout=0,stdout=out,stderr=err)
                    y_exploration0.append(np.squeeze(eng.workspace['maxima'])[2:]) ## first two are Hs,Tp
                y_exploration0 = np.array(y_exploration0)
            else:
                print('   - Retrieving cached data n={:d}...'.format(n_samples))
                y_exploration0 = iglobal_data.y_train[:n_samples,:,theta] ## shape (nsample, nQoIs, n_short_term)

            data_exploration0 = uqra.Data()
            data_exploration0.xi= xi_exploration0
            data_exploration0.x = x_exploration0
            data_exploration0.y = y_exploration0

            data_train.xi  = np.concatenate([data_train.xi, xi_exploration0], axis=1)
            data_train.x   = np.concatenate([data_train.x , x_exploration0 ], axis=1)
            data_train.y   = np.concatenate([data_train.y , y_exploration0 ], axis=0)
            data_train.xi_index = uqra.list_union(data_train.xi_index, idx_optimal)
            data_ideg_QoIs[iqoi].exploration0= data_exploration0
            data_ideg_QoIs[iqoi].data_train_.append(copy.deepcopy(data_train))


            print(' ------------------------------------------------------------')
            print('   Build PCE (p={:d}) model with {} '.format(deg, model_params.fitting))
            assert np.array_equal( data_train.x, solver.map_domain(data_train.xi, dist_xi))
            weight    = idoe_params.sampling_weight()   ## weight function
            data_train= data_ideg_QoIs[iqoi].data_train_[-1]
            pce_model = uqra.PCE(orth_poly)
            dist_u    = model_params.dist_u 
            dist_xi   = orth_poly.weight
            dist_x    = solver.distributions
            pce_model.info()
            print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], 
                pce_model.num_basis, data_train.x.shape[1]/pce_model.num_basis))
            print('     - {:<32s} : {}'.format('Y train'    , data_train.y[:,iqoi].shape))
            print('     - {:<32s} : {}'.format('Y test '    , xi_test.shape))

            pce_model.fit(model_params.fitting, data_train.xi, data_train.y[:, iqoi]/model_params.y_scales[iqoi], 
                    w=weight,n_jobs=model_params.n_jobs) 
            data_ideg_QoIs[iqoi].sparsity = len(pce_model.active_index)
            y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
            data_ideg_QoIs[iqoi].model_.append(pce_model)
            data_ideg_QoIs[iqoi].score_.append(pce_model.score)
            data_ideg_QoIs[iqoi].cv_err_.append(pce_model.cv_error)

            max_sparsity = max(max_sparsity, data_ideg_QoIs[iqoi].sparsity)

            data_excd = uqra.Data()
            data_excd.pf     = model_params.pf
            data_excd.y0_hat = uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf)
            data_excd.x0_hat = x_test [:,np.argmin(abs(y_test_hat-data_excd.y0_hat))]
            data_excd.xi0_hat= xi_test[:,np.argmin(abs(y_test_hat-data_excd.y0_hat))]
            eng.workspace['deg']       = float(deg)
            eng.workspace['phaseSeed'] = float(theta)
            eng.workspace['Hs'] = float(data_excd.x0_hat[0])
            eng.workspace['Tp'] = float(data_excd.x0_hat[1])
            eng.wecSim(nargout=0,stdout=out,stderr=err)
            y0 = np.squeeze(eng.workspace['maxima'])[2:] ## first two are Hs,Tp
            data_excd.y0 = y0[iqoi]/model_params.y_scales[iqoi] 
            data_ideg_QoIs[iqoi].y0_hat_.append(data_excd)
            print('     - {:<32s} : {}'.format('Sparsity', data_ideg_QoIs[iqoi].sparsity))
            print('     - {:<32s} : x0={}, y0={:.4e}'.format('Estiamted exceedance value',data_excd.x0_hat, data_excd.y0_hat)) 
            print('     - {:<32s} : y={:.4e}, err={:.2f} %'.format('Response with true system at x0', data_excd.y0,
                (data_excd.y0_hat - data_excd.y0)/data_excd.y0)) 

            ## don't waste data, save this one sample into training set
            data_train.xi  = np.concatenate([data_train.xi, data_excd.xi0_hat.reshape(ndim, 1)], axis=1)
            data_train.x   = np.concatenate([data_train.x , data_excd.x0_hat.reshape(ndim, 1)], axis=1)
            data_train.y   = np.concatenate([data_train.y , y0.reshape(1,-1)], axis=0)
            data_ideg_QoIs[iqoi].data_train_[-1]= copy.deepcopy(data_train)

            #############################################################################
            #############################################################################
            i_iteration = 1
            while i_iteration <= 20:
                # print(' ------------------------------------------------------------')
                print('          Sequential Optimal Design: Iteration # {:d} >'.format(i_iteration))
                # print(' ------------------------------------------------------------')
                n_samples = min(3, max(3,max_sparsity)) 
                print('   > 1. exploration step (FULL basis)... ')
                print('     - {:<32s} : {:d}'.format('Adding exploration optimal samples', n_samples))
                ####-------------------------------------------------------------------------------- ####
                # min(max_sparsity, model_params.alpha *pce_model.num_basis - n_samples_deg, 5)
                # n_samples = min(10, max_sparsity) #len(active_index)
                xi_exploration, idx_optimal = idoe_params.get_samples(data_cand, orth_poly, n_samples, x0=data_train.xi_index, 
                        active_index=None, initialization='RRQR', return_index=True) 
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
                data_exploration = uqra.Data()
                data_exploration.xi= xi_exploration
                data_exploration.x = x_exploration
                data_exploration.y = y_exploration
                data_ideg_QoIs[iqoi].exploration_.append(data_exploration)

                data_train.xi  = np.concatenate([data_train.xi, xi_exploration], axis=1)
                data_train.x   = np.concatenate([data_train.x , x_exploration ], axis=1)
                data_train.y   = np.concatenate([data_train.y , y_exploration ], axis=0)
                data_train.xi_index = uqra.list_union(data_train.xi_index, idx_optimal)
                data_ideg_QoIs[iqoi].data_train_.append(copy.deepcopy(data_train))

                print(' ------------------------------------------------------------')
                print('   Build PCE (p={:d}) model with {} '.format(deg, model_params.fitting))
                assert np.array_equal( data_train.x, solver.map_domain(data_train.xi, dist_xi))
                pce_model = uqra.PCE(orth_poly)
                weight    = doe_params.sampling_weight()   ## weight function
                print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], 
                    pce_model.num_basis, data_train.x.shape[1]/pce_model.num_basis))
                print('     - {:<32s} : {}'.format('Y train'    , data_train.y[:,iqoi].shape))
                print('     - {:<32s} : {}'.format('Y test '    , xi_test.shape))
                print('     {:<20s}, prediction samples: {}'.format(headers[iqoi], xi_test.shape))

                pce_model.fit(model_params.fitting, data_train.xi, data_train.y[:, iqoi]/model_params.y_scales[iqoi], 
                        w=weight,n_jobs=model_params.n_jobs) 
                data_ideg_QoIs[iqoi].sparsity = len(pce_model.active_index)
                y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)
                data_ideg_QoIs[iqoi].model_.append(pce_model)
                data_ideg_QoIs[iqoi].score_.append(pce_model.score)
                data_ideg_QoIs[iqoi].cv_err_.append(pce_model.cv_error)

                max_sparsity = max(max_sparsity, data_ideg_QoIs[iqoi].sparsity)

                data_excd = uqra.Data()
                data_excd.pf     = model_params.pf
                data_excd.y0_hat = uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf)
                data_excd.x0_hat = x_test [:,np.argmin(abs(y_test_hat-data_excd.y0_hat))]
                data_excd.xi0_hat= xi_test[:,np.argmin(abs(y_test_hat-data_excd.y0_hat))]
                eng.workspace['deg']       = float(deg)
                eng.workspace['phaseSeed'] = float(theta)
                eng.workspace['Hs'] = float(data_excd.x0_hat[0])
                eng.workspace['Tp'] = float(data_excd.x0_hat[1])
                eng.wecSim(nargout=0,stdout=out,stderr=err)
                y0 = np.squeeze(eng.workspace['maxima'])[2:] ## first two are Hs,Tp
                data_excd.y0 = y0[iqoi]/model_params.y_scales[iqoi] 
                data_ideg_QoIs[iqoi].y0_hat_.append(data_excd)
                print('     - {:<32s} : {}'.format('Sparsity', data_ideg_QoIs[iqoi].sparsity))
                print('     - {:<32s} : x0={}, y0={:.4e}'.format('Estiamted exceedance value',data_excd.x0_hat, data_excd.y0_hat)) 
                print('     - {:<32s} : y={:.4e}, err={:.2f} %'.format('Response with true system at x0',data_excd.y0,
                    (data_excd.y0_hat - data_excd.y0)/data_excd.y0)) 

                ## don't waste data, save this one sample into training set
                data_train.xi  = np.concatenate([data_train.xi, data_excd.xi0_hat.reshape(ndim, 1)], axis=1)
                data_train.x   = np.concatenate([data_train.x , data_excd.x0_hat.reshape(ndim, 1)], axis=1)
                data_train.y   = np.concatenate([data_train.y , y0.reshape(1,-1)], axis=0)
                data_ideg_QoIs[iqoi].data_train_[-1]= copy.deepcopy(data_train)

                ####-------------------------------------------------------------------------------- ####
                print('   > 2. exploitation step (SIGNIFICANT basis)... ')

                ## obtain DoI candidate samples for each QoI

                print('     - {:<32s} : {}'.format('Iteration Converge', data_ideg_QoIs[iqoi].iteration_converge))
                ## obtain candidate samples for each QoI
                # data_cand_DoI_iqoi, idx_data_cand_DoI = idoe_params.samples_nearby(data_ideg_QoIs[iqoi].y0_hat_[-1], 
                        # xi_test, data_ideg_QoIs[iqoi].y_test_hat, data_cand, deg, n0=10, epsilon=0.1, return_index=True)

                y0_hat_err = abs(data_ideg_QoIs[iqoi].y0_hat_[-1].y0_hat - data_ideg_QoIs[iqoi].y0_hat_[-1].y0)/data_ideg_QoIs[iqoi].y0_hat_[-1].y0
                if y0_hat_err < 0.1:
                    data_cand_DoI_iqoi = idoe_params.domain_of_interest(data_ideg_QoIs[iqoi].y0_hat_[-1].y0_hat, xi_test, 
                            y_test_hat, n_centroid=5, epsilon=0.1)
                else:
                    data_cand_DoI_iqoi = idoe_params.domain_of_interest(data_ideg_QoIs[iqoi].y0_hat_[-1].y0_hat, xi_test, 
                            y_test_hat, n_centroid=1, epsilon=0.2)
                # data_cand_DoI_iqoi = idoe_params.domain_of_interest(data_ideg_QoIs[iqoi].y0_hat_[-1].y0_hat, xi_test, 
                        # data_ideg_QoIs[iqoi].y_test_hat, n_centroid=5, epsilon=0.2)

                # data_cand_DoI_iqoi = idoe_params.domain_of_interest(data_ideg_QoIs[iqoi].y0_hat_[-1], xi_test, 
                        # data_ideg_QoIs[iqoi].y_test_hat, n_centroid=20, epsilon=0.1)

                data_ideg_QoIs[iqoi].DoI_xi_.append(data_cand_DoI_iqoi)
                data_ideg_QoIs[iqoi].DoI_x_.append(solver.map_domain(data_cand_DoI_iqoi, dist_xi ))
                print('     - {:<32s} : {}  '.format('DoI candidate samples', data_cand_DoI_iqoi.shape ))
                ## get optimal samples for each QoI
                print('     - {:<32s} : {:d}'.format('Adding DoI optimal samples', n_samples ))
                xi_exploitation, idx_optimal_DoI = idoe_params.get_samples(data_cand_DoI_iqoi, orth_poly, n_samples, x0=[], 
                        active_index= data_ideg_QoIs[iqoi].model_[-1].active_index, initialization='RRQR', return_index=True) 
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
                data_ideg_QoIs[iqoi].exploitation_.append(data_exploitation)

                ## save all training samples together
                data_train.xi  = np.concatenate([data_train.xi, xi_exploitation], axis=1)
                data_train.x   = np.concatenate([data_train.x , x_exploitation ], axis=1)
                data_train.y   = np.concatenate([data_train.y , y_exploitation ], axis=0)
                data_train.xi_index = uqra.list_union(data_train.xi_index, idx_optimal)
                data_ideg_QoIs[iqoi].data_train_.append(copy.deepcopy(data_train))


     
                print(' ------------------------------------------------------------')
                print('   Build PCE (p={:d}) model with {} '.format(deg, model_params.fitting))
                assert np.array_equal( data_train.x, solver.map_domain(data_train.xi, dist_xi))
                pce_model = uqra.PCE(orth_poly)
                weight    = doe_params.sampling_weight()   ## weight function
                print('     - {:<32s} : ({},{}),    Alpha: {:.2f}'.format('X train', data_train.x.shape[1], 
                    pce_model.num_basis, data_train.x.shape[1]/pce_model.num_basis))
                print('     - {:<32s} : {}'.format('Y train'    , data_train.y[:,iqoi].shape))
                print('     - {:<32s} : {}'.format('Y test '    , xi_test.shape))
                print('     {:<20s}, prediction samples: {}'.format(headers[iqoi], xi_test.shape))

                pce_model.fit(model_params.fitting, data_train.xi, data_train.y[:, iqoi]/model_params.y_scales[iqoi], 
                        w=weight, n_jobs=model_params.n_jobs) 

                data_ideg_QoIs[iqoi].sparsity = len(pce_model.active_index)
                y_test_hat = pce_model.predict(xi_test, n_jobs=model_params.n_jobs)

                data_ideg_QoIs[iqoi].model_.append(pce_model)
                data_ideg_QoIs[iqoi].score_.append(pce_model.score)
                data_ideg_QoIs[iqoi].cv_err_.append(pce_model.cv_error)

                max_sparsity = max(max_sparsity, data_ideg_QoIs[iqoi].sparsity)

                data_excd = uqra.Data()
                data_excd.pf     = model_params.pf
                data_excd.y0_hat = uqra.metrics.mquantiles(y_test_hat, 1-model_params.pf)
                data_excd.x0_hat = x_test [:,np.argmin(abs(y_test_hat-data_excd.y0_hat))]
                data_excd.xi0_hat= xi_test[:,np.argmin(abs(y_test_hat-data_excd.y0_hat))]
                eng.workspace['deg']       = float(deg)
                eng.workspace['phaseSeed'] = float(theta)
                eng.workspace['Hs'] = float(data_excd.x0_hat[0])
                eng.workspace['Tp'] = float(data_excd.x0_hat[1])
                eng.wecSim(nargout=0,stdout=out,stderr=err)
                y0 = np.squeeze(eng.workspace['maxima'])[2:] ## first two are Hs,Tp
                data_excd.y0 = y0[iqoi]/model_params.y_scales[iqoi] 
                data_ideg_QoIs[iqoi].y0_hat_.append(data_excd)
                print('     - {:<32s} : {}'.format('Sparsity', data_ideg_QoIs[iqoi].sparsity))
                print('     - {:<32s} : x0={}, y0={:.4e}'.format('Estiamted exceedance value',data_excd.x0_hat, data_excd.y0_hat)) 
                print('     - {:<32s} : y={:.4e}, err={:.2f} %'.format('Response with true system at x0',data_excd.y0,
                    (data_excd.y0_hat - data_excd.y0)/data_excd.y0)) 

                ## don't waste data, save this one sample into training set
                data_train.xi  = np.concatenate([data_train.xi, data_excd.xi0_hat.reshape(ndim, 1)], axis=1)
                data_train.x   = np.concatenate([data_train.x , data_excd.x0_hat.reshape(ndim, 1)], axis=1)
                data_train.y   = np.concatenate([data_train.y , y0.reshape(1,-1)], axis=0)
                data_ideg_QoIs[iqoi].data_train_[-1]= copy.deepcopy(data_train)


                data_ideg_QoIs[iqoi].cv_err = data_ideg_QoIs[iqoi].cv_err_[-1]
                data_ideg_QoIs[iqoi].score  = data_ideg_QoIs[iqoi].score_ [-1]
                data_ideg_QoIs[iqoi].model  = data_ideg_QoIs[iqoi].model_ [-1]
                data_ideg_QoIs[iqoi].y0_hat = data_ideg_QoIs[iqoi].y0_hat_[-1]

                print('   4. converge check ...')
                is_QoIs_converge = [] 
                y0_hat = [idata.y0_hat for idata in data_QoIs[iqoi].y0_hat_] 
                is_y0_converge   , y0_converge_err = relative_converge(y0_hat, err=2*model_params.rel_err)
                is_score_converge, score_converge  = threshold_converge(data_ideg_QoIs[iqoi].score_)
                is_PCE_accurate = abs(data_QoIs[iqoi].y0_hat_[-1].y0_hat - data_QoIs[iqoi].y0_hat_[-1].y0)/data_QoIs[iqoi].y0_hat_[-1].y0
                data_ideg_QoIs[iqoi].iteration_converge = is_y0_converge and is_score_converge and is_PCE_accurate < 0.1 

                print('  >  QoI: {:<25s}'.format(headers[iqoi]))
                print('     >  Values: {}'.format(np.array(y0_hat)))
                print('     >  Rel Error : {:5.2f} %, Converge: {}'.format(y0_converge_err*100, is_y0_converge     ))
                print('     >  Fit Score : {:5.2f} %, Converge: {}'.format(score_converge *100, is_score_converge  ))
                print('     >  Error of response at x0: {}, {:5.2f} %, y0_hat: {:.2f}, y0: {:.2f}'.format(
                    data_QoIs[iqoi].y0_hat_[-1].x0_hat, is_PCE_accurate*100, 
                    data_QoIs[iqoi].y0_hat_[-1].y0_hat, data_QoIs[iqoi].y0_hat_[-1].y0))
                print('     -------------------------------------------')

                i_iteration +=1
                if data_ideg_QoIs[iqoi].iteration_converge:
                    print('         !< Iteration converge for order {:d} >!'.format(deg))
                    break
                if data_train.x.shape[1] > model_params.alpha*orth_poly.num_basis:
                    print('     PCE(d={:d},p={:d}) !< Number of samples exceeding {:.2f}P >!'.format(
                        ndim, deg, model_params.alpha))
                    break
            #### end sequential sampling inner loop

            ### check the deg outer loop, 
            data_degs_QoIs[deg] = copy.deepcopy(data_ideg_QoIs)
            print('--------------------------------------------------')
            print('     Model Performance up to order p={:d}'.format(deg))
            is_QoIs_converge = [] 
            is_QoIs_overfit  = []
            iheader   = headers[iqoi]
            data_iqoi = [data_ideg_QoIs[iqoi] for data_ideg_QoIs in data_degs_QoIs[model_params.degs[0]: deg+1]]
            cv_err_iqoi_degs = np.array([idata.cv_err for idata in data_iqoi]).T
            y0_hat_iqoi_degs = np.array([idata.y0_hat.y0_hat for idata in data_iqoi]).T
            score_iqoi_degs  = np.array([idata.score  for idata in data_iqoi]).T
            is_overfit       , overfit_vals    = overfitting_check(cv_err_iqoi_degs) ## check Overfitting
            is_y0_converge   , y0_converge_err = relative_converge(y0_hat_iqoi_degs, err=model_params.rel_err)
            is_score_converge, score_converge  = threshold_converge(score_iqoi_degs)

            data_degs_QoIs[deg][iqoi].deg_overfit  = is_overfit 
            data_degs_QoIs[deg][iqoi].deg_converge = is_y0_converge and  is_score_converge
            print('  >  QoI: {:<25s}'.format(iheader))
            print('     >  Values: {}'.format(np.array(y0_hat_iqoi_degs)))
            print('     >  Overfit : {}; CV errors: {}'.format(is_overfit, overfit_vals))
            print('     >  Rel Error [%]: {:5.2f}, Converge: {}'.format(y0_converge_err*100, is_y0_converge     ))
            print('     >  Fit Score [%]: {:5.2f}, Converge: {}'.format(score_converge *100, is_score_converge  ))
            print('--------------------------------------------------')

            if data_degs_QoIs[deg][iqoi].deg_overfit:
                deg = deg - 1
                continue
            else:
                deg = deg + 1
                if data_degs_QoIs[deg][iqoi].deg_converge:
                    break
                else:
                    continue

    return data_degs_QoIs

if __name__ == '__main__':
    ## ------------------------ Displaying set up ------------------- ###
    r, theta   = 0, 9## r is the number of repeated MCS samples, availble in 0 to 9
    ## batch parameters are used to validate the uncertainty due to sampling on same theta and same r
    ## not used for practice, only for benchmark validation
    # ith_batch  = 0
    # batch_size = 1
    random_state = 0
    np.random.seed(random_state)
    random.seed(random_state)
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
    model_params.degs    = np.arange(2,8) #[2,6,10]#
    model_params.ndim    = solver.ndim
    model_params.basis   = 'Heme'
    model_params.dist_u  = stats.uniform(0,1)  #### random CDF values for samples
    model_params.fitting = 'OLSLAR' 
    model_params.n_splits= 10
    model_params.alpha   = 3
    model_params.num_test= int(1e7)
    model_params.pf      = np.array([1.0/(365.25*24*50)])
    model_params.abs_err = 1e-4
    model_params.rel_err = 2.5e-2
    model_params.n_jobs  = mp.cpu_count()
    model_params.channel = [2, 12, 23, 24, 25]
    model_params.y_scales= np.zeros(34)
    model_params.y_scales[model_params.channel]= [1, 1e7, 1e6, 1e7, 1]
    model_params.update_basis()
    model_params.info()
    ## ------------------------ UQRA DOE Parameters ----------------- ###
    doe_params = uqra.ExperimentParameters('MCS', 'S')
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
    xi_test     = data_test.xi[:, :model_params.num_test] 
    x_test      = data_test.x [:, :model_params.num_test] 
    ### 2. Get finished global data
    filename = '{:s}_Adap{:d}{:s}_{:s}E5R{:d}_global.npy'.format(solver.nickname, 
            solver.ndim, model_params.basis[:3], doe_params.doe_nickname(), r)
    global_data = np.load(os.path.join(data_dir_result, filename), allow_pickle=True).tolist()
    headers  = global_data[0].headers

    print('\n#################################################################################')
    print(' >>>  File: ', __file__)
    print(' >>>  Start UQRA : Theta: {:d}'.format(theta))
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
    res = main(model_params, doe_params, solver, r=r, random_state=random_state, theta=theta)
    filename = '{:s}_Adap{:d}{:s}_{:s}E5R{:d}S{:d}'.format(solver.nickname, 
            solver.ndim, model_params.basis, doe_params.doe_nickname(), r, theta)
    eng.quit()
    # ## ============ Saving QoIs ============
    res = np.array(res, dtype=object)
    try:
        np.save(os.path.join(data_dir_result, filename), res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(data_dir_result, filename)))
    except:
        np.save(filename, res, allow_pickle=True)
        print(' >> Simulation Done! Data saved to {:s}'.format(os.path.join(os.getcwd(), filename)))
