#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Jinsong Liu <jinsongliu@utexas.edu>
#
# Distributed under terms of the GNU-License license.

"""

"""
import numpy as np
import matplotlib.pyplot as plt

def sff(f):
    """
    input f is in Hz
    """
    f = 2 * np.pi *f
    c = 2
    sf = 2*c/(c**2 + f**2)
    df = f[1] - f[0]
    sa = np.sum(sf*df) 
    return sf, sa
def corrfunc(t):
    c = 2
    return np.exp(-c*abs(t))

# f, axes = plt.subplots(1,2)
# t_max = 20
# dt = 0.01
# # t = np.arange(-t_max,t_max,dt)
# t = np.arange(-t_max,t_max,dt)
# t = np.roll(t, int(t_max / dt))
# print(t[0])
# rt = corrfunc(t)
# sf = np.fft.fft(rt)
# freq = np.fft.fftfreq(t.shape[-1],d=dt)
# df = freq[1] - freq[0]
# amp = np.sqrt(sf.real**2 + sf.imag**2)  * dt 

# amp = np.array([x for _,x in sorted(zip(freq,amp))])
# freq  = np.array([x for _,x in sorted(zip(freq,freq))])
# axes[0].plot(freq, sff(freq*6.28)[0])
# axes[0].plot(freq, amp)


# t = np.arange(0,t_max,dt)
# rt = corrfunc(t)
# rt = np.hstack((rt, np.flip(rt[0:-1], axis=0)))
# sf = np.fft.fft(rt)
# freq = np.fft.fftfreq(t.shape[-1],d=dt)
# df = freq[1] - freq[0]
# amp = np.sqrt(sf.real**2 + sf.imag**2)  * dt 

# amp = np.array([x for _,x in sorted(zip(freq,amp))])
# freq  = np.array([x for _,x in sorted(zip(freq,freq))])
# axes[1].plot(freq, sff(freq*6.28)[0])
# axes[1].plot(freq, amp)

# plt.show()




# dt = .1
# t = np.arange(-10,10,dt)
# tr = corrfunc(t)
# sf = np.fft.fft(tr)
# freq = np.fft.fftfreq(t.shape[-1],d=dt)
# print(t)
# print(tr)
# # print(sf.real)
# amp = np.sqrt(sf.real**2 + sf.imag**2) *dt
# # print(amp - sff(freq)[0])
# ts = np.fft.ifft(sf)
# # print(freq)
# # print(tr)
# print(ts.real)

# # plt.plot(freq, amp)
# # plt.plot(freq, sff(freq)[0])
# plt.plot(t,tr,t,ts)
# plt.show()
# # plt.plot(freq, sf.real)
# # plt.show()

f, axes = plt.subplots(1,2)
T = 10
dt = 0.1
df = 1/(2*T)
fmax = 1/(2*dt)
N = fmax/df
f1 = np.arange(-N, N) *df
f1 = np.roll(f1,int(fmax/df))
sf1,sa1 = sff(f1)
ts1 = np.fft.ifft(sf1)/dt
t = np.arange(0, N) *dt

amp = np.sqrt(ts1.real **2 + ts1.imag**2)[0:len(t)] 
# axes[0].plot(t,amp,'-o')
axes[0].plot(t,ts1.real[0:len(t)],'-o')

axes[0].plot(t,corrfunc(t),'-*')

plt.show()

# print(f1[0])
# dt1 = 1/(2*fmax)
# dt1 = 1/(2*fmax)
# print(f1)
# print('sigma_x ~=:{:f}'.format(sa1))
# t = np.arange(len(ts1)) *dt1
# # t = np.arange(int(len(ts1)/2)+1) *dt1
# amp = np.sqrt(ts1.real **2 + ts1.imag**2)[0:len(t)] 
# # amp = ts1.real[0:len(t)]
# # axes[0].plot(ts1.real - amp)
# axes[0].plot(t,amp,'-o')
# # axes[0].set_xlim(0,1/df)
# axes[0].plot(t,corrfunc(t),'-*')
# # print(corrfunc(t)/ts1)
# plt.show()

# fmax = 100
# num_points = fmax/df * 2 + 1
# dt1 = 1/fmax
# f1 = np.linspace(-fmax, fmax, num_points)
# sf1,sa1 = sff(f1)
# ts1 = np.fft.ifft(sf1).real
# axes[1].plot(np.arange(len(ts1)) *dt1, ts1)
# axes[1].set_xlim(0,1/df)
# plt.show()


# fmax = 100
# dt2 = 1/fmax
# f2 = np.arange(-fmax,fmax,df)
# sf2,sa2 = sff(f2)
# ts2 = np.fft.ifft(sf2).real

# t1 = np.arange(dt1,1/df, dt1)
# # t2 = np.arange(dt2,1/df, dt2)
# plt.figure()
# plt.plot(t1,ts1[1:len(t1)+1],label='f1')
# plt.plot(t2,corrfunc(t2),label='Rtau')
# # plt.plot(t2,ts2[1:len(t2)+1],label='f2')
# # plt.plot(t2,corrfunc(t2),label='Rtau')
# # f, axes = plt.subplots(1,2,sharey=True)
# # axes[0].plot(t1, ts1[0:len(t1)])
# # axes[1].plot(t1, ts2[0:len(t1)])
# plt.legend()
# plt.show()
# plt.figure()
# plt.plot(t2, corrfunc(t2)/ts2[1:len(t2)])
# plt.show()

# t = np.linspace(0,10,200)
# rt = corrfunc(t)
# print(sa)

# plt.figure()
# plt.plot(f,sf)
# plt.show()

# plt.figure()
# plt.plot(ts)
# # plt.plot(t,ts[0:200])
# # plt.plot(t,rt)
# plt.show()

