#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:50:37 2019

@author: gechen

Beam_forming_exercise_V1.py

Compute the power response of an antenna array. 
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import cmath

N_ant = 64 # dish number 
d_ant = 4.75 # dish diameter 
b = 1.5 * d_ant # dish distance in m.

BW = 250 # MHz, bandwidth 
N_ch = 8 # channel number 
nu_c = 1405 # MHz, central frequency 

c = 3e8 # speed of light in m.

# Fringes 
theta_c = 3.0 # in degree, source angle with respect to upwards 
theta_range = 5 # degree, range of angle to show 
# Note: must include 0 in array theta_rel, so num=even number.
theta_num = 1000 
theta_rel = np.linspace(-1*theta_range, theta_range, num=2 * theta_num + 1) # in degree, relative angle from theta_c
theta_abs = theta_c + theta_rel # in degree, angle from the upwards direction 

nu_ch = np.linspace(nu_c-0.5*BW, nu_c+0.5*BW, num=N_ch) # Mean frequency of each channel


tau_geometry = np.array([])
E_arr = np.ones(len(theta_rel)) # the leading array 
n_ant = 0 # Initial antenna number 
t = 0.0 # reference time
while n_ant < N_ant-1:
    n_ant += 1
    tau_geometry = n_ant * b / c * np.sin(np.deg2rad(theta_rel)) # in s
    
    for nu in nu_ch: # MHz 
        fringe_phases = 2 * np.pi * 1e6 * nu * (t - tau_geometry) # in radian 
        E = np.array([cmath.exp(-1j*phi) for phi in fringe_phases]) 
        E_arr = np.vstack((E_arr, E))
        
        #p = stats.norm.pdf(theta_rel, 0, nu/d_ant) # single dish power response function 

E_sum = [np.sum(E_arr[:,i]) for i in np.arange(len(theta_rel))] 
I_sum_sky = [abs(x * x.conjugate()) for x in E_sum] # normalize by /max(I_sum) 
I_sum_sky_norm = I_sum_sky/max(I_sum_sky)


fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.)
ax1.plot(theta_abs, I_sum_sky_norm) 
ax1.vlines(theta_c, 0, 1, linestyle='dashed')
ax1.set_ylim(bottom = 0)
ax1.set_xlabel('Angle from Zenith [degree]', fontsize = 12)
ax1.set_ylabel('Power Response', fontsize = 12)
ax1.set_title(r'Sky Fringes ($N_{channel}$=%d, $N_{antenna}$=%d)'%(N_ch, N_ant), fontsize = 16) 
fig1.savefig('Sky_fringes_Nch%d_Nant%d_range%d.pdf'%(N_ch, N_ant, theta_range)) 


# For DSA, individual antenna power response always peak at zenith (the whole RA)
# How to use frequency-dependent power response? Cannot get sky power function before summing N_ant? 
# sigma = 3.3/2, or 1.22 * lambda /(2*d) ?
p_ant = stats.norm.pdf(theta_abs, 0, 3.3/2)  # single dish power response at central frequency 
p_ant_norm = p_ant / max(p_ant) 

I_tot = p_ant * I_sum_sky # Total power response 
I_tot_norm = I_tot / max(I_tot)

fig2, ax2 = plt.subplots() 
fig2.set_size_inches(8., 6.)
ax2.plot(theta_abs, I_tot_norm, 'k', label='Total power response')
ax2.plot(theta_abs, I_sum_sky_norm, 'c:', label='Sky power fringes')
ax2.plot(theta_abs, p_ant_norm, 'm--', label='Single antenna response')
ax2.legend(loc='best')
ax2.set_ylim(bottom = 0)
ax2.set_xlabel('Angle from Zenith [degree]', fontsize = 12)
ax2.set_ylabel('Power Response', fontsize = 12)
ax2.set_title(r'Total power reaponse ($N_{channel}$=%d, $N_{antenna}$=%d)'%(N_ch, N_ant), fontsize = 14) 
fig2.savefig('Total_Nch%d_Nant%d_range%.1f_centered%d.pdf'%(N_ch, N_ant, theta_range, theta_c))



