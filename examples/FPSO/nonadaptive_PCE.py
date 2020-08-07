#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import uqra, warnings, random, math
import numpy as np, os, sys
import collections
import scipy.stats as stats
import scipy
import scipy.io
from tqdm import tqdm
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

def get_basis(ndim, deg, poly_type):
    if poly_type.lower() == 'leg':
        print(' Legendre polynomial')
        basis = uqra.Legendre(d=ndim,deg=deg)

    elif poly_type.lower() == 'hem':
        print(' Physicist Hermite polynomial')
        basis = uqra.Hermite(d=ndim,deg=deg, hem_type='physicists')

    elif poly_type.lower() == 'heme':
        print(' Probabilists Hermite polynomial')
        basis = uqra.Hermite(d=ndim,deg=deg, hem_type='probabilists')

    else:
        raise NotImplementedError

    return basis 


def main(ST):
    print('------------------------------------------------------------')
    print('>>> Model: FPSO, Short-term simulation (n={:d})  '.format(ST))
    print('------------------------------------------------------------')

    ## ------------------------ Displaying set up ------------------- ###
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    pf       = 0.5/(50*365.25*24)
    radius_surrogate= 3
    Kvitebjorn      = uqra.environment.Kvitebjorn()
    short_term_seeds_applied = np.setdiff1d(np.arange(11), np.array([]))

    ## ------------------------ Simulation Parameters ----------------- ###
    short_term_seeds = np.arange(ST,ST+1)
    assert short_term_seeds.size == 1
    solver    = uqra.FPSO(phase=short_term_seeds)
    simparams = uqra.Parameters()
    simparams.solver     = solver
    simparams.pce_degs   = np.array(range(2,21))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = -1
    simparams.doe_method = 'MCS' ### 'mcs', 'cls1', 'cls2', ..., 'cls5', 'reference'
    simparams.optimality = 'D'# 'D', 'S', None
    simparams.poly_type  = 'heme'
    # simparams.poly_type  = 'hem'
    if simparams.poly_type.lower().startswith('hem'):
        if simparams.doe_method.lower().startswith('cls'):
            assert simparams.poly_type.lower() == 'hem'
        elif simparams.doe_method.lower().startswith('mcs'):
            assert simparams.poly_type.lower() == 'heme'
        else:
            raise ValueError

    simparams.fit_method = 'LASSOLARS'
    simparams.n_splits   = 50
    repeats              = 1 #if simparams.optimality is None else 1
    alphas               = 1.2
    # # simparams.num_samples=np.arange(21+1, 130, 5)
    simparams.update()
    simparams.info()

    ## ------------------------ MCS Benchmark ----------------- ###
    print('------------------------------------------------------------')
    print('>>> Monte Carlo Sampling for Model: FPSO                    ')
    print('------------------------------------------------------------')
    filename = 'DoE_McsE7R0.npy'
    mcs_data = np.load(os.path.join(simparams.data_dir_result, filename))
    mcs_data_ux, mcs_data_y = mcs_data[:4], mcs_data[4+short_term_seeds_applied,:]
    y50_MCS  = uqra.metrics.mquantiles(mcs_data_y.T, 1-pf, multioutput='raw_values')
    y50_MCS_mean  = np.mean(y50_MCS)
    y50_MCS_std   = np.std(y50_MCS)

    mcs_data_mean = np.mean(mcs_data_y, axis=0)
    y50_mean_MCS  = uqra.metrics.mquantiles(mcs_data_mean, 1-pf, multioutput='raw_values')
    y50_mean_MCS_idx = (np.abs(mcs_data_mean-y50_mean_MCS)).argmin()
    y50_mean_MCS_ux  = mcs_data_ux[:,y50_mean_MCS_idx]
    y50_mean_MCS  = np.array(list(y50_mean_MCS_ux)+[y50_mean_MCS,])

    print(' > Extreme reponse from MCS:')
    print('   - {:<25s} : {}'.format('Data set', mcs_data_y.shape))
    print('   - {:<25s} : {}'.format('y50 MCS', y50_MCS))
    print('   - {:<25s} : {}'.format('y50 MCS [mean, std]', np.array([y50_MCS_mean, y50_MCS_std])))
    print('   - {:<25s} : {}'.format('y50 Mean MCS', np.array(y50_mean_MCS[-1])))
    print('   - {:<25s} : {}'.format('Design state (u,x)', y50_mean_MCS[:4]))

    ## ------------------------ Environmental Contour ----------------- ###
    print('------------------------------------------------------------')
    print('>>> Environmental Contour for Model: FPSO                   ')
    print('------------------------------------------------------------')
    filename    = 'FPSO_DoE_EC2D_T50_y.npy' 
    EC2D_data_y = np.load(os.path.join(simparams.data_dir_result, filename))[short_term_seeds_applied,:] 
    filename    = 'FPSO_DoE_EC2D_T50.npy' 
    EC2D_data_ux= np.load(os.path.join(simparams.data_dir_result, filename))[:4]

    EC2D_median = np.median(EC2D_data_y, axis=0)
    EC2D_data   = np.concatenate((EC2D_data_ux,EC2D_median.reshape(1,-1)), axis=0)
    y50_EC      = EC2D_data[:,np.argmax(EC2D_median)]

    print(' > Extreme reponse from EC:')
    print('   - {:<25s} : {}'.format('EC data set', EC2D_data_y.shape))
    print('   - {:<25s} : {}'.format('y0', np.array(y50_EC[-1])))
    print('   - {:<25s} : {}'.format('Design state (u,x)', y50_EC[:4]))

    np.random.seed(100)
    EC2D_y_boots      = uqra.bootstrapping(EC2D_data_y, 100) 
    EC2D_boots_median = np.median(EC2D_y_boots, axis=1)
    y50_EC_boots_idx  = np.argmax(EC2D_boots_median, axis=-1)
    y50_EC_boots_ux   = np.array([EC2D_data_ux[:,i] for i in y50_EC_boots_idx]).T
    y50_EC_boots_y    = np.max(EC2D_boots_median,axis=-1) 
    y50_EC_boots      = np.concatenate((y50_EC_boots_ux, y50_EC_boots_y.reshape(1,-1)), axis=0)
    y50_EC_boots_mean = np.mean(y50_EC_boots, axis=1)
    y50_EC_boots_std  = np.std(y50_EC_boots, axis=1)
    print(' > Extreme reponse from EC (Bootstrap (n={:d})):'.format(EC2D_y_boots.shape[0]))
    print('   - {:<25s} : {}'.format('Bootstrap data set', EC2D_y_boots.shape))
    print('   - {:<25s} : [{:.2f}, {:.2f}]'.format('y50[mean, std]',y50_EC_boots_mean[-1], y50_EC_boots_std[-1]))
    print('   - {:<25s} : {}'.format('Design state (u,x)', y50_EC_boots_mean[:4]))

    u_center = y50_EC_boots_mean[ :2].reshape(-1, 1)
    x_center = y50_EC_boots_mean[2:4].reshape(-1, 1)
    print(' > Important Region based on EC(boots):')
    print('   - {:<25s} : {}'.format('Radius', radius_surrogate))
    print('   - {:<25s} : {}'.format('Center U', np.squeeze(u_center)))
    print('   - {:<25s} : {}'.format('Center X', np.squeeze(x_center)))
    print('================================================================================')


    ## ----------- Candidate and testing data set for DoE ----------- ###
    ## ----- validation data set centered around y50_EC
    # print(' > Getting validation data set...')
    # filename_validate  = 'FPSO_DoE_{:s}E5R0.npy'.format(simparams.doe_method.capitalize())
    # data_vald = np.load(os.path.join(simparams.data_dir_result, filename_validate)) 
    # u_vald    = data_vald[             :  solver.ndim]
    # x_vald    = data_vald[solver.ndim  :2*solver.ndim]
    # y_vald    = np.squeeze(data_vald[2*solver.ndim+short_term_seeds,:])
    # # y_vald    = np.mean(y_vald, axis=0).reshape(1,-1)
    # print('   - {:<25s} : {}, {}, {}'.format('Validation Dataset (U,X,Y)',u_vald.shape, x_vald.shape, y_vald.shape ))
    # print('   - {:<25s} : [{}, {}]'.format('Validation U[mean, std]',np.mean(u_vald, axis=1),np.std (u_vald, axis=1)))
    # print('   - {:<25s} : [{}]'.format('Validation max(U)[U1, U2]',np.amax(abs(u_vald), axis=1)))
    # print('   - {:<25s} : [{}]'.format('Validatoin [min(Y), max(Y)]',np.array([np.amin(y_vald),np.amax(y_vald)])))


    ## ----- Testing data set centered around (0,0)
    print(' > Getting Testing data set...')
    filename    = 'FPSO_DoE_McsE7R0_Test_Radius3.npy'
    data_test   = np.load(os.path.join(simparams.data_dir_result, filename))
    u_test      = data_test[:solver.ndim, :]
    x_test      = data_test[solver.ndim:2*solver.ndim, :]
    y_test      = np.squeeze(data_test[2*solver.ndim+short_term_seeds,:])
    
    # filename    = 'DoE_McsE7R0.npy'
    # u_test      = np.load(os.path.join(simparams.data_dir_sample, 'MCS', 'Norm', filename))[:solver.ndim,:]
    # x_test      = Kvitebjorn.ppf(stats.norm().cdf(u_test))
    # y_test      = -np.inf* np.ones((u_test.shape[1],))
    # idx_in_circle   = np.arange(u_test.shape[1])[np.linalg.norm(u_test-u_center, axis=0) < radius_surrogate]
    # u_test_in_circle= u_test[:,idx_in_circle]
    # x_test_in_circle= x_test[:,idx_in_circle]
    # y_test_in_circle= solver.run(x_test_in_circle, verbose=True)
    # np.save(os.path.join(simparams.data_dir_result, 'y_test_in_circle.npy'), y_test_in_circle)

    print('   - {:<25s} : {}, {}, {}'.format('Test Dataset (U,X,Y)',u_test.shape, x_test.shape, y_test.shape ))
    # print('   - {:<25s} : [{}, {}]'.format('Test U[mean, std]',np.mean(u_test, axis=1),np.std (u_test, axis=1)))
    print('   - {:<25s} : [{}]'.format('Test max(U)[U1, U2]',np.amax(abs(u_test), axis=1)))
    print('   - {:<25s} : [{}]'.format('Test [min(Y), max(Y)]',np.array([np.amin(y_test),np.amax(y_test)])))
    # print('   - {:<25s} : {}'.format('# Test data in Circle({:.2f})'.format(radius_surrogate),u_test_in_circle.shape))


    ## ----------- Candidate and testing data set for DoE ----------- ###
    print(' > Getting candidate data set...')
    # u_cand = modeling.get_candidate_data()

    if simparams.doe_method.lower().startswith('cls2'):
        u_cand = np.load(os.path.join(simparams.data_dir_sample, 'CLS', 'DoE_Cls2E7d2R0.npy'))[:solver.ndim, :simparams.n_cand]
        u_cand = u_cand  * radius_surrogate
    elif simparams.doe_method.lower().startswith('mcs'):
        u_cand = np.load(os.path.join(simparams.data_dir_sample, 'MCS','Norm','DoE_McsE6R0.npy'))
        u_cand = u_cand[:solver.ndim, np.linalg.norm(u_cand[:2], axis=0)<radius_surrogate]
        u_cand = u_cand[:, :simparams.n_cand]
    # u_cand    = pce_deg ** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
    # u_cand    = 2** 0.5 * u_cand if modeling.is_cls_unbounded() else u_cand
    ### ============ Initial Values ============

    res_pce_deg = []
    data_test = []

    for pce_deg in simparams.pce_degs:
        print('\n================================================================================')
        # simparams.info()
        print('   - Sampling and Fitting:')
        print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_method))
        print('     - {:<23s} : {}'.format('Optimality '      , simparams.optimality))
        print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))

        print(' > Building surrogate model ...')
        ## ----------- Define PCE  ----------- ###
        orth_poly = get_basis(solver.ndim, pce_deg, simparams.poly_type)
        pce_model = uqra.PCE(orth_poly)
        modeling  = uqra.Modeling(solver, pce_model, simparams)
        pce_model.info()

        ### ----------- Oversampling ratio ----------- ###
        simparams.update_num_samples(pce_model.num_basis, alphas=alphas)
        print(' > Oversampling ratio: {}'.format(np.around(simparams.alphas,2)))
        n_train  = int(alphas * pce_model.num_basis)  

        ### ============ Get training points ============
        # _, u_train = modeling.get_train_data((repeats,n_train), u_cand, u_train=None, basis=pce_model.basis)

        print(' > Getting Train data set...')
        if simparams.optimality is None:
            doe_idx_u_cand = np.arange(simparams.n_cand) 
            filename= 'Random index'
        else:
            filename = 'DoE_{:s}E5R0_{:s}_{:s}r{:d}.npy'.format(
                        simparams.doe_method.capitalize(), pce_model.tag, simparams.optimality, radius_surrogate)
            doe_idx_u_cand = np.load(os.path.join(simparams.data_dir_sample, 'OED', filename))
            doe_idx_u_cand = np.array(doe_idx_u_cand, ndmin=2)[0]

        ### ============ Build Surrogate Model ============
        u_train = u_cand[:, doe_idx_u_cand[:n_train]]
        x_train = Kvitebjorn.ppf(stats.norm.cdf(u_train + u_center)) 
        y_train = solver.run(x_train)
        U_train = pce_model.basis.vandermonde(u_train)
        if simparams.doe_method.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis, pce_model.active_index)
            U_train = modeling.rescale_data(U_train, w_train) 
        else:
            w_train = None
            U_train = U_train[:, pce_model.active_index]

        pce_model.fit(simparams.fit_method, u_train, y_train.T, w=w_train, n_splits=simparams.n_splits)

        print('   - {:<25s} : {:s}'.format('File', filename))
        print('   - {:<25s} : {}, {}, {}'.format('Train Dataset (U,X,Y)',u_train.shape, x_train.shape, y_train.shape))
        if w_train is None:
            print('   - {:<25s} : {}'.format('Train Dataset W ', 'None'))
        else:
            print('   - {:<25s} : {}'.format('Train Dataset W ', w_train.shape))
        print('   - {:<25s} : [{}, {}]'.format('Train U[mean, std]',np.mean(u_train, axis=1),np.std (u_train, axis=1)))
        print('   - {:<25s} : [{}]'.format('Train max(U)[U1, U2]',np.amax(abs(u_train), axis=1)))
        print('   - {:<25s} : [{}]'.format('Train [min(Y), max(Y)]',np.array([np.amin(y_train),np.amax(y_train)])))

        y_train_hat = pce_model.predict(u_train)
        # y_vald_hat  = pce_model.predict(u_vald-u_center)
        # vald_mse    = uqra.metrics.mean_squared_error(y_vald.T, y_vald_hat.T,multioutput='raw_values')

        y_test_hat = pce_model.predict(u_test - u_center)
        test_error = uqra.metrics.mean_squared_error(y_test, y_test_hat,multioutput='raw_values')

        mcs_u_in_circle = mcs_data_ux[ :2,np.linalg.norm(mcs_data_ux[:2] - u_center, axis=0) < radius_surrogate]
        mcs_x_in_circle = mcs_data_ux[-2:,np.linalg.norm(mcs_data_ux[:2] - u_center, axis=0) < radius_surrogate]
        mcs_y_in_circle_hat = pce_model.predict(mcs_u_in_circle - u_center)
        alpha = (pf * mcs_data_ux.shape[1]) / mcs_y_in_circle_hat.size
        y50_pce_y   = uqra.metrics.mquantiles(mcs_y_in_circle_hat, 1-alpha)
        y50_pce_idx = np.array(abs(mcs_y_in_circle_hat - y50_pce_y)).argmin()
        y50_pce_uxy = np.concatenate((mcs_u_in_circle[:,y50_pce_idx], mcs_x_in_circle[:, y50_pce_idx], y50_pce_y)) 
        data_test.append([pce_deg, n_train, mcs_y_in_circle_hat])
        res = [pce_deg, n_train, pce_model.cv_error, test_error[0]]
        for item in y50_pce_uxy:
            res.append(item)
        res_pce_deg.append(res)

        ### ============ calculating & updating metrics ============
        tqdm.write(' > Summary')
        with np.printoptions(precision=4):
            tqdm.write('     - {:<15s} : {}'.format( 'y50_pce_y' , np.array(res_pce_deg)[-1:,-1]))
            tqdm.write('     - {:<15s} : {}'.format( 'Test MSE ' , np.array(res_pce_deg)[-1:, 3]))
            tqdm.write('     - {:<15s} : {}'.format( 'CV MSE'    , np.array(res_pce_deg)[-1:, 2]))
            tqdm.write('     - {:<15s} : {}'.format( 'Design state', np.array(res_pce_deg)[-1:,6:8]))
            tqdm.write('     ----------------------------------------')
            # tqdm.write('     - {:<15s} : {:.4f}'.format( 'kappa '    , kappa))


    res_pce_deg = np.array(res_pce_deg)
    data_test = np.array(data_test, dtype=object)

    with open(os.path.join(simparams.data_dir_result, 'outlist_name.txt'), "w") as text_file:
        text_file.write('\n'.join(['pce_deg', 'n_train', 'cv_error', 'test mse', 'y50_pce_u', 'y50_pce_x', 'y50_pce_y']))

    ### Saving test data in circle
    filename = '{:s}_{:s}_{:s}_Alpha{}_ST{}_test'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(alphas).replace('.', 'pt'), short_term_seeds[-1])
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), data_test)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), data_test)

    ### Saving QoI data  
    filename = '{:s}_{:s}_{:s}_Alpha{}_ST{}'.format(solver.nickname, pce_model.tag, 
            simparams.tag, str(alphas).replace('.', 'pt'), short_term_seeds[-1])
    try:
        np.save(os.path.join(simparams.data_dir_result, filename), res_pce_deg)
    except:
        print(' Directory not found: {}, file save locally... '.format(simparams.data_dir_result))
        np.save(os.path.join(os.getcwd(), filename), res_pce_deg)


if __name__ == '__main__':
    for s in range(11):
        main(s)
