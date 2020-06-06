# -*- coding: utf-8 -*-

import museuq, unittest,warnings,os, sys 
from tqdm import tqdm
import numpy as np, scipy as sp 
from museuq.solver.PowerSpectrum import PowerSpectrum
from museuq.environment import Kvitebjorn as Kvitebjorn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pickle
import scipy.stats as stats

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
sys.stdout  = museuq.utilities.classes.Logger()

data_dir = '/Users/jinsongliu/BoxSync/MUSELab/museuq/examples/JupyterNotebook'

def nextPowerOf2(n): 
	count = 0; 

	# First n in the below 
	# condition is for the 
	# case where n is 0 
	if (n and not(n & (n - 1))): 
		return n 
	
	while( n != 0): 
		n >>= 1
		count += 1
	
	return 1 << count; 


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""


    def test_ifft(self):
        signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
        fourier = np.fft.fft(signal)
        n = signal.size
        timestep = 0.1
        freq = np.fft.fftfreq(n, d=timestep)
        print(freq)
        freq = np.fft.fftfreq(n, d=1)
        print(freq)

        np.random.seed(100)
        env = museuq.Environment('JONSWAP')
        x   = [2.8,13.4]
        w_rad   = np.arange(0.2, 1.4,0.001)
        dw      = w_rad[1] - w_rad[0]
        print(w_rad[-1]/dw)
        A_size  = nextPowerOf2(int(w_rad[-1]/dw))
        A       = np.zeros(A_size, dtype=complex)
        spectrum= PowerSpectrum(env.spectrum, *x)
        density = spectrum.cal_density(w_rad)
        theta   = stats.uniform.rvs(-np.pi, 2*np.pi, size=np.size(w_rad))
        env_c1  = np.sqrt(spectrum.dw * spectrum.pxx) * np.exp(1j*theta) 
        A[int(w_rad[0]/dw):int(w_rad[-1]/dw)] = env_c1
        A_conj  = np.conj(A)
        A       = np.append(A, np.flip(A_conj[1:]))
        eta     = np.fft.ifft(A).real * np.size(A)
        np.save(os.path.join(data_dir, 'eta'), eta)
        print(np.size(eta))
        print(eta[:5])
        print(np.std(eta))
        sigma = np.std(eta)
        print('4*std: {}'.format(4 * sigma))



        Re, Im  = stats.norm.rvs(0,1, size=(2, np.size(w_rad)))
        env_c   = np.sqrt(spectrum.dw * spectrum.pxx)/2.0 * (Re + 1j*Im)
        A[int(w_rad[0]/dw):int(w_rad[-1]/dw)] = env_c
        A_conj  = np.conj(A)
        A       = np.append(A, np.flip(A_conj[1:]))

        eta = np.fft.ifft(A).real * np.size(A)
        print(np.size(eta))
        print(eta[:5])
        print(np.std(eta))
        sigma = np.std(eta)
        print('4*std: {}'.format(4 * sigma))

if __name__ == '__main__':
    unittest.main()
