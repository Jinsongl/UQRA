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
# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

class Data():
    pass

def main(theta):
    ## ------------------------ Displaying set up ------------------- ###
    print('\n'+'#' * 80)
    print(' >>>  Start simulation: {:s}, theta={:d}'.format(__file__, theta))
    print('#' * 80 + '\n')
    np.random.seed(100)
    np.set_printoptions(precision=4)
    np.set_printoptions(threshold=8)
    np.set_printoptions(suppress=True)
    Kvitebjorn = uqra.environment.Kvitebjorn()
    pf = 0.5/(50*365.25*24)

    ## ------------------------ Simulation Parameters ----------------- ###
    solver    = uqra.FPSO(random_state =theta)
    simparams = uqra.Parameters(solver, doe_method=['CLS4', 'D'], fit_method='LASSOLARS')
    simparams.set_udist('norm')
    simparams.x_dist     = Kvitebjorn
    simparams.pce_degs   = np.array(range(2,11))
    simparams.n_cand     = int(1e5)
    simparams.n_test     = int(1e6)
    simparams.n_pred     = int(1e7)
    simparams.n_splits   = 50
    simparams.alphas     = 1
    n_initial = 20
    simparams.info()

    ## ----------- Define U distributions ----------- ###
    sampling_domain = -simparams.u_dist[0].ppf(1e-7)
    sampling_space  = 'u'
    dist_support    = None        ## always in x space

    # sampling_domain = [[-1,1],] * solver.ndim 
    # sampling_space  = 'u'
    # dist_support    = np.array([[0, 18],[0,35]]) ## always in x space

    ## ----------- Predict data set ----------- ###
    print(' > Getting predict data set...')
    filename = 'FPSO_SURGE_McsE7R{:d}_pred.npy'.format(theta)
    data = np.load(os.path.join(simparams.data_dir_result, 'TestData', filename))
    idx_inside, idx_outside = simparams.separate_samples_by_domain(data[:simparams.ndim], sampling_domain)
    data_pred = data[:, idx_inside ]
    data_pred_outside = data[:, idx_outside]
    u_pred = data_pred[:simparams.ndim]
    x_pred = data_pred[simparams.ndim:2*simparams.ndim]
    u_pred_outside = data_pred_outside[:simparams.ndim]
    x_pred_outside = data_pred_outside[simparams.ndim:2*simparams.ndim]
    print('   - {:<25s} : {}'.format(' Sample domain', sampling_domain))
    print('   - {:<25s} : {}'.format(' dist Support ', dist_support if dist_support is None else dist_support.reshape(1,-1)))
    print('   - {:<25s} '.format(' Samples inside domain...'))
    print('   - {:<25s} : {}, {}'.format(' U,X', u_pred.shape, x_pred.shape))
    print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_pred, axis=1), np.amax(u_pred, axis=1)))
    print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_pred, axis=1), np.amax(x_pred, axis=1)))
    print('   - {:<25s} '.format(' Samples outside domain...'))
    print('   - {:<25s} : {}'.format('X ', x_pred_outside.shape))

    y_pred_outside = np.array(solver.run(x_pred_outside), ndmin=1)
    print('   - {:<25s} : {}, {}'.format('(X,Y) ', x_pred_outside.shape, y_pred_outside.shape))

    ## ----------- Test data set ----------- ###
    print(' > Getting Test data set...')
    filename = '{:s}_McsE6R{:d}_test.npy'.format(solver.nickname,theta)
    data = np.load(os.path.join(simparams.data_dir_result, 'TestData', filename))
    u_test, x_test, y_test = data[:simparams.ndim],data[simparams.ndim:2*simparams.ndim], data[2*simparams.ndim] 
    print('   - {:<25s} : {}, {}, {}'.format(' U,X,Y.shape', u_test.shape, x_test.shape, y_test.shape ))
    print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_test, axis=1), np.amax(u_test, axis=1)))
    print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_test, axis=1), np.amax(x_test, axis=1)))
    print('   - {:<25s} : [{}]'.format(' Y [min(Y), max(Y)]', np.array([np.amin(y_test),np.amax(y_test)])))


    output_each_deg = []
    print(' > Train data initialization ...')
    ## Initialize u_train with LHS 
    u_train = simparams.get_init_samples(n_initial, doe_candidate='lhs', random_state=100)
    ## mapping points to the square in X space
    x_train = uqra.inverse_rosenblatt(simparams.x_dist, u_train, simparams.u_dist, support=dist_support)
    ## mapping points to physical space
    y_train = solver.run(x_train)

    print('   - {:<25s} : {}, {}, {}'.format(' Dataset (U,X,Y)',u_train.shape, x_train.shape, y_train.shape))
    print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_train, axis=1), np.amax(u_train, axis=1)))
    print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_train, axis=1), np.amax(x_train, axis=1)))
    print('   - {:<25s} : [{}]'.format(' Y [min(Y), max(Y)]',np.array([np.amin(y_train),np.amax(y_train)])))

    for deg in simparams.pce_degs:
        print('\n================================================================================')
        print('   - Sampling and Fitting:')
        print('     - {:<23s} : {}'.format('Sampling method'  , simparams.doe_candidate))
        print('     - {:<23s} : {}'.format('Optimality '      , simparams.doe_optimality))
        print('     - {:<23s} : {}'.format('Fitting method'   , simparams.fit_method))

        ## ----------- Candidate data set for DoE ----------- ###
        print(' > Getting candidate data set...')
        if simparams.doe_candidate in ['uniform', 'unf']:
            fname_cand = 'FPSO_SURGE_DoE_UnfE5_Norm.npy'
            data = np.load(os.path.join(simparams.data_dir_result, 'TestData', fname_cand))
            u_cand, x_cand = data[:solver.ndim], data[solver.ndim:]
        elif simparams.doe_candidate.startswith('cls'):
            fname_cand = 'FPSO_SURGE_DoE_Cls4E5.npy'
            u_cand = np.load(os.path.join(simparams.data_dir_result, 'TestData', fname_cand)) 
            u_cand = u_cand * np.sqrt(deg)
            x_cand = uqra.inverse_rosenblatt(simparams.x_dist, u_cand, simparams.u_dist, support=dist_support)
        else:
            raise ValueError

        print('   - {:<25s} : {}'.format(' Dataset (U)', u_cand.shape))
        print('   - {:<25s} : {}, {}'.format(' U [min(U), max(U)]', np.amin(u_cand, axis=1), np.amax(u_cand, axis=1)))
        print('   - {:<25s} : {}, {}'.format(' X [min(X), max(X)]', np.amin(x_cand, axis=1), np.amax(x_cand, axis=1)))

        print(' > Building surrogate model ...')
        ## ----------- Define PCE  ----------- ###
        basis     = simparams.get_basis(deg)
        pce_model = uqra.PCE(basis)
        modeling  = uqra.Modeling(solver, pce_model, simparams)
        pce_model.info()

        #================================================================================
        # New train samples 
        #================================================================================
        ### global new train samples 
        print('   => Getting training data based on sparsity...')
        U_train = pce_model.basis.vandermonde(u_train)
        if simparams.doe_candidate.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis, pce_model.active_index)
            U_train = U_train[:, pce_model.active_index]
            U_train = modeling.rescale_data(U_train, w_train) 
        else:
            w_train = None
            U_train = U_train[:, pce_model.active_index]

        _, sig_value, _ = np.linalg.svd(U_train)

        pce_model.fit('LASSOLARS', u_train, y_train.T, w=w_train,epsilon=1e-3)
        print('     - {:<23s} : {}'.format('Active Index'   , pce_model.active_index))
        active_basis= pce_model.active_basis
        sparsity    = pce_model.sparsity 
        n_train_new = int(simparams.alphas*sparsity)
        tqdm.write('    > {}:{}; Sparsity: {}/{}; #samples = {:d}'.format('DoE', simparams.doe_optimality.upper(), sparsity, 
            pce_model.num_basis, n_train_new ))

        u_train_new, _ = modeling.get_train_data(n_train_new, u_cand, u_train=u_train, active_basis=active_basis)
        x_train_new = uqra.inverse_rosenblatt(simparams.x_dist, u_train_new, simparams.u_dist, support=dist_support)
        y_train_new = solver.run(x_train_new)
        u_train = np.hstack((u_train, u_train_new)) 
        x_train = np.hstack((x_train, x_train_new)) 
        y_train = np.hstack((y_train, y_train_new)) 

        ### ============ Build 2nd Surrogate Model ============
        U_train = pce_model.basis.vandermonde(u_train)
        if simparams.doe_candidate.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis, active_index=pce_model.active_index)
            U_train = U_train[:, pce_model.active_index]
            U_train = modeling.rescale_data(U_train, w_train) 
        else:
            w_train = None
            U_train = U_train[:, pce_model.active_index]

        pce_model.fit('OLS', u_train, y_train.T, w_train, active_basis=active_basis)

        ### localized new train samples 
        print('   => Getting training data based on top y ellipsoid...')
        y_pred = pce_model.predict(u_pred)
        if u_pred_outside is None or len(u_pred_outside) == 0:
            y_pred_all = y_pred
            x_pred_all = x_pred
            u_pred_all = u_pred
        else:
            y_pred_all = np.concatenate((y_pred, y_pred_outside)) 
            x_pred_all = np.concatenate((x_pred, x_pred_outside), axis=1) 
            u_pred_all = np.concatenate((u_pred, u_pred_outside), axis=1) ## could have nan in u
        if y_pred_all.size != simparams.n_pred:
            raise ValueError (' Expecting {:d} predict samples but only have {:d}'.format(simparams.n_pred, y_pred_all.size))
        y_pred_top_idx = np.argsort(y_pred_all)[-2*int(pf*simparams.n_pred):]
        y_pred_top = y_pred_all[   y_pred_top_idx]
        u_pred_top = u_pred_all[:, y_pred_top_idx]
        x_pred_top = x_pred_all[:, y_pred_top_idx]
        ## remove nan values in u
        if np.isnan(u_pred_top).any():
            raise ValueError('nan in u_pred_top')

        center, radii, rotation = uqra.EllipsoidTool().getMinVolEllipse(u_pred_top.T)
        # if simparams.u_distname == 'norm':
            # distance = max(np.linalg.norm(u_pred_top-center, axis=0))
        # elif simparams.u_distname == 'uniform':
            # distance = [[min(iu), max(iu)] for iu in u_pred_top-center]
        simparams.topy_center.append(center)
        simparams.topy_radii.append(radii)
        simparams.topy_rotation.append(rotation)

        
        idx_within_ellipse, _ = uqra.samples_within_ellipse(u_cand, center, radii, rotation)
        u_cand_topy = u_cand[:, idx_within_ellipse]

        tqdm.write('     - Min Vol ellipsoid with {:d} samples'.format(y_pred_top.size))
        tqdm.write('     - {:<15s} : {}'.format( 'center'   , center))
        tqdm.write('     - {:<15s} : {}'.format( 'radii'    , radii))
        tqdm.write('     - {:<15s} : {}'.format( 'candidate', u_cand_topy.shape[1]))
        u_train_topy, _ = modeling.get_train_data(sparsity, u_cand_topy, u_train=u_train, 
                active_basis=active_basis)

        x_train_topy = uqra.inverse_rosenblatt(simparams.x_dist, u_train_topy, simparams.u_dist, support=dist_support)
        y_train_topy = solver.run(x_train_topy)
        u_train = np.hstack((u_train, u_train_topy)) 
        x_train = np.hstack((x_train, x_train_topy)) 
        y_train = np.hstack((y_train, y_train_topy)) 

        ### ============ Build 3rd Surrogate Model ============
        # print(bias_weight)
        U_train = pce_model.basis.vandermonde(u_train)
        if simparams.doe_candidate.lower().startswith('cls'):
            w_train = modeling.cal_cls_weight(u_train, pce_model.basis, active_index=pce_model.active_index)
            U_train = U_train[:, pce_model.active_index]
            U_train = modeling.rescale_data(U_train, w_train) 
        else:
            w_train = None
            U_train = U_train[:, pce_model.active_index]
        _, sig_value, _ = np.linalg.svd(U_train)
        kappa = max(abs(sig_value)) / min(abs(sig_value)) 

        pce_model.fit('OLS', u_train, y_train.T, w_train, active_basis=active_basis)

        print(' > Train data ...')
        print('   - {:<25s} : {}, {}, {}'.format(' Dataset (U,X,Y)',u_train.shape, x_train.shape, y_train.shape))
        if w_train is None:
            print('   - {:<25s} : {}'.format(' weight ', 'None'))
        else:
            print('   - {:<25s} : {}'.format(' weight ', w_train.shape))
        print('   - {:<25s} : [{}]'.format(' max(U)[U1, U2]',np.amax(abs(u_train), axis=1)))
        print('   - {:<25s} : {}, {}'.format(' U support', np.amin(u_train, axis=1), np.amax(u_train, axis=1)))
        print('   - {:<25s} : {}, {}'.format(' X support', np.amin(x_train, axis=1), np.amax(x_train, axis=1)))
        print('   - {:<25s} : [{}]'.format(' Y [min(Y), max(Y)]',np.array([np.amin(y_train),np.amax(y_train)])))

        y_train_hat = pce_model.predict(u_train)
        y_test_hat  = pce_model.predict(u_test)
        train_error = uqra.metrics.mean_squared_error(y_train, y_train_hat, squared=False)
        test_error  = uqra.metrics.mean_squared_error(y_test , y_test_hat , squared=False)

        y_pred = pce_model.predict(u_pred)
        if u_pred_outside is None or len(u_pred_outside) == 0:
            y_pred_all = y_pred
            x_pred_all = x_pred
            u_pred_all = u_pred
        else:
            y_pred_all = np.concatenate((y_pred_outside, y_pred)) 
            x_pred_all = np.concatenate((x_pred_outside, x_pred), axis=1) 
            u_pred_all = np.concatenate((u_pred_outside, u_pred), axis=1) ## could have nan in u

        if y_pred_all.size != simparams.n_pred:
            raise ValueError (' Expecting {:d} predict samples but only have {:d}'.format(simparams.n_pred, y_pred_all.size))
        y_pred_ecdf = uqra.utilities.helpers.ECDF(y_pred_all, alpha=pf, compress=True)
        y50_pce_y   = uqra.metrics.mquantiles(y_pred_all, 1-pf)
        y50_pce_u   = u_pred_all[:, np.array(abs(y_pred_all - y50_pce_y)).argmin()]
        y50_pce_x   = x_pred_all[:, np.array(abs(y_pred_all - y50_pce_y)).argmin()]

        data = Data()
        data.simparams = simparams
        data.deg     = deg
        data.nsamples= u_train.shape[1]
        data.model   = pce_model
        data.metrics = [train_error, pce_model.cv_error, test_error, kappa, y50_pce_y.item()]
        data.w_train = w_train
        data.y_ecdf  = y_pred_ecdf
        data.y0      = np.concatenate([y50_pce_u, y50_pce_x, y50_pce_y]).reshape(-1,1)
        data.topy_data = np.concatenate((u_pred_top, x_pred_top, y_pred_top.reshape(1,-1)), axis=0)
        data.topy_ellipsoid = [center, radii, rotation]
        data.train_data = np.concatenate((u_train, x_train, y_train.reshape(1,-1)), axis=0)
        data.outside = np.concatenate((u_pred_outside, x_pred_outside, y_pred_outside.reshape(1,-1)), axis=0)

        ### ============ calculating & updating metrics ============
        tqdm.write(' > Summary')
        with np.printoptions(precision=4):
            tqdm.write('     - {:<15s} : {}'.format( 'y50 PCE Y:', y50_pce_y))
            tqdm.write('     - {:<15s} : {:.4f}'.format( 'Train MSE' , train_error))
            tqdm.write('     - {:<15s} : {:.4f}'.format( 'CV MSE'    , pce_model.cv_error))
            tqdm.write('     - {:<15s} : {:.4f}'.format( 'Test MSE ' , test_error))
            tqdm.write('     ----------------------------------------')

        output_each_deg.append(data)
    
    filename = '{:s}_EmAdap{:s}_{:s}_ST{}.npy'.format(solver.nickname, pce_model.tag[:4], simparams.tag, theta)
    output   = np.array(output_each_deg, dtype=object)
    np.save(os.path.join(simparams.data_dir_result, filename), output)

if __name__ == '__main__':
    for s in range(10):
        main(s)
