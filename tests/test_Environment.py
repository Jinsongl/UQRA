# -*- coding: utf-8 -*-

import uqra, unittest,warnings,os, sys 
from tqdm import tqdm
import numpy as np, scipy as sp 
import scipy.stats as stats
from uqra.environment import Kvitebjorn as Kvitebjorn

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = uqra.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/uqra/examples/JupyterNotebook'
class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_Kvitebjorn(self):
        print('========================TESTING: Kvitebjorn =======================')

        # data_dir = '/Users/jinsongliu/BoxSync/MUSELab/uqra/uqra/environment'
        # hs1     = np.linspace(0,2.9,291)
        # hs2     = np.linspace(2.90,20, 1711)
        # hs      = np.hstack((hs1, hs2))
        # hs_pdf  = Kvitebjorn.hs_pdf(hs) 
        # np.save(os.path.join(data_dir, 'Kvitebjorn_hs'), np.vstack((hs, hs_pdf)))
        for r in range(10):
            data_dir    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/MCS/Norm'
            filename = 'DoE_McsE6R{:d}.npy'.format(r)
            mcs_sampels = np.load(os.path.join(data_dir, filename))
            mcs_sampels = stats.norm().cdf(mcs_sampels)
            print(dir(Kvitebjorn))
            samples_x   = Kvitebjorn.ppf(mcs_sampels[:2,:])
            data_dir    = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/Samples/Kvitebjorn/Norm'
            # np.save(os.path.join(data_dir, filename), samples_x)

        # return EC from Kvitebjorn
        # P = 10
        # EC_samples = Kvitebjorn.EC(P)
        # np.save(os.path.join(data_dir, 'Kvitebjorn_EC_P{:d}'.format(P)), EC_samples)

        # ## test cdf method for Kvitebjørn
        # u = np.array([np.linspace(0,0.99999,11), np.linspace(0,0.99999,11)])
        # x = Kvitebjorn.samples(u)
        # u_= Kvitebjorn.cdf(x)
        # print(np.around(u,2))
        # print(np.around(x,2))
        # print(np.around(u_,2))



if __name__ == '__main__':
    unittest.main()
