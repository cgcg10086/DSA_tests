#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:53:54 2019

@author: gechen

FRB_rate_single_v1.py: computes the detection rate of a single FRB.
"""
import numpy as np 
from scipy import stats 
from scipy import integrate
import matplotlib.pyplot as plt


def DM_pdf(DM, mu=544, sigma=406):
    '''Gaussian distributions from fitting results (FRB_population_v1.py)'''
    return stats.norm.pdf(DM, mu, sigma) # gaussian 

def Width_intrinsic_pdf(w_int, mu=1.85, sigma=2.58):
    '''
    Gaussian distributions from fitting results (FRB_population_v1.py)
    in ms 
    '''
    return stats.norm.pdf(w_int, mu, sigma)

def Power_law_cdf(a, x_low, x_0): 
    '''
    Compute the probability that x > x_0 for a power law pdf: y = x^a 
    # x_low: lower limit of the pdf 
    '''
    return 1./(a+1) * (x_0 ** (a + 1) - x_low ** (a + 1))


def Fluence_cdf(F_0, F_b=15, F_low=0.5, F_up = 1e5, alpha = -1.18, beta = -2.2):
    '''
    Compute the probability to have a FRB whose fluence > F_0 
    fluence pdf: broken power law from James et al. 2019 
    'The slope of the source-count distribution for fast radio bursts' 
    #F_b: fluence turning point is between 5 tp 40 Jy ms  
    #F_low: fluence lower limit 
    #F_up: fluence upper limit     
    #alpha: Parks pdf power index  
    #beta: Askap pdf power index 
    '''
    if F_0 < F_low: 
        fluence_cdf = 0 
    elif F_0 < F_b:
        fluence_cdf = Power_law_cdf(alpha, F_low, F_0) # 1./(alpha+1) * (x ** (alpha + 1) - F_low ** (alpha + 1))
    else:
        fluence_cdf = Power_law_cdf(alpha, F_low, F_b) + Power_law_cdf(beta, F_b, F_0)  
    
    norm_scale = 1./(Power_law_cdf(alpha, F_low, F_b) + Power_law_cdf(beta, F_b, F_up)) # normalize the pdf with this scaling factor 
    
    return norm_scale * fluence_cdf 


def Compute_dispersion_smearing(DM, frequency_central=1405, channel_width=0.122):
    '''
    Using the exact DM, i.e. infinite DM samples
    frequency_central: central frequency in MHz (1280-1530 MHz)
    channel_width: in MHz (bandwidth 250 MHz/ 2048 channels)
    returns dispersion broadening in ms.
    '''
    return 8.3 *1e6 * DM * channel_width * frequency_central ** (-3) 
    

def Compute_width_obs(w_int, DM):
    '''
    Compute the observed width from the dispersion broadening and the intrinsic width w_int.
    w_DM, w_int and w_obs: all in ms.  
    '''
    w_DM = Compute_dispersion_smearing(DM) 
    w_obs = np.sqrt(w_int ** 2 + w_DM ** 2) 
    return w_obs 


def Compute_flux_noise(w_obs, time_resolution, T_sys = 35, bandwidth = 250):
    '''
    Compute the flux noise level of one reading (width < time resolution) in Jy.
    # w_obs: in ms 
    # time_resolution: in ms 
    # T_sys: in Kelvin
    # bandwidth: in MHz
    '''
    Kb = 1381 # in m^2*Jy/Kelvin 
    efficiency = 0.65 
    n = 85 # what is this? 
    D = 4.65 # diameter in m 
    A_eff = efficiency * n * np.pi * (D/2.)**2 # Effective area 
    gain = A_eff / (2 * Kb) # in K/Jy 
    SEFD = T_sys / gain # 103 Jy for the default argument values   
    polarization = 2 
    noise_single = SEFD / np.sqrt(polarization * bandwidth * 1e3 * time_resolution) #Flux noise of a single measurement. 
    n = int(w_obs / time_resolution) + 1 # number of flux samples 
    flux_noise = noise_single / np.sqrt(n) # noise reduced by muiltiple samples 
    return flux_noise # single sample noise = 0.40234375 Jy, for the default values 


def Compute_S2N(fluence, w_obs, time_resolution):
    '''
    Compute the observed mean flux from the fluence, and the S/N. 
    fluence: Jin y ms
    w_obs, time_resolution: in ms. 
    '''
    if w_obs <= time_resolution:
        t_burst = time_resolution 
    else:
        t_burst = w_obs # moving box convolution box can read the exact width 
        #t_burst = time_resolution * (int(w_obs / time_resolution) + 1) # rescale the width to integer number * time resolution.   
    
    flux_obs = fluence / t_burst
    noise = Compute_flux_noise(w_obs, time_resolution) 
        
    return flux_obs / noise 

def Compute_F0(w_int, DM, time_resolution, S2N_min=8):
    '''
    compute fluence threshold for a given time resolution and width.
    '''
    w_obs = Compute_width_obs(w_int, DM) 
    flux_noise = Compute_flux_noise(w_obs, time_resolution) 
    
    if w_obs <= time_resolution: 
        F0 = flux_noise * S2N_min * time_resolution 
    else: 
        F0 = flux_noise * S2N_min * w_obs    

    return F0  


def Rate_integrand(w_int, DM, time_resolution):
    '''
    Integrate this function to get the detection rate.
    '''    
    F_0 = Compute_F0(w_int, DM, time_resolution) # fluence threshold for a given width     
    event_rate_above_F_0 = 1 - Fluence_cdf(F_0) # N[F>F0(w, DM)]  
    
    return event_rate_above_F_0 * Width_intrinsic_pdf(w_int) * DM_pdf(DM) 

    
def Compute_detection_rate(time_resolution, f=Rate_integrand, w_low=0, w_up=100, DM_low=0, DM_up=2000): 
    '''
    Total number of detectable events per day for the instrument. 
    '''
    return integrate.dblquad(f, 0, np.inf, lambda x: 0, lambda x: np.inf, args=[time_resolution]) 
    #return integrate.dblquad(f, w_low, w_up, lambda x: 0, lambda x: 2000, args=[time_resolution]) 


# -- main -- 

rate = np.array([]) 
rate_err = np.array([]) 
time_resolution = np.logspace(-3, 0, num=11) # 1 microsec to 1 millisec
# Use some typical width, fluence, and DM values to test the noise and S/N functions.
my_w_int = 0.5 # width in ms
my_F = 1 # fluence 
my_DM = 100
S2N = np.array([])
F0 = np.array([]) 

for t in time_resolution:  # in ms 
    print 'time resol=%.3f ms,'%t, 'noise=%.2f Jy ms,'%Compute_flux_noise(my_w_int, t), \
    'S/N=%.1f,'%Compute_S2N(my_F, my_w_int, t), \
    'rate=', Compute_detection_rate(t, Rate_integrand) # (integral result, error)
    S2N = np.append(S2N, Compute_S2N(my_F, my_w_int, t))
    F0 = np.append(F0, Compute_F0(my_w_int, my_DM, t))
    rate = np.append(rate, Compute_detection_rate(t, Rate_integrand)[0]) 
    rate_err = np.append(rate_err, Compute_detection_rate(t, Rate_integrand)[1]) 
    

    
fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.)
ax1.set_xscale('log') 
ax1.set_yscale('log') 
ax1.scatter(time_resolution, rate) 
ax1.set_xlabel(r'Time resolution [ms]', fontsize = 12) 
ax1.set_ylabel(r'Detection rate', fontsize = 12) 
fig1.savefig('Rate_vs_time_resol.pdf') 
#plt.close() 


fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.)
ax1.set_xscale('log') 
ax1.set_yscale('log') 
#ax1.set_ylim(top = 15)
ax1.scatter(time_resolution, S2N, s=5)  
ax1.set_xlabel('Time Resolution', fontsize = 12)
ax1.set_ylabel('S/N', fontsize = 12)
ax1.set_title('width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
#fig1.savefig('S2N_vs_time_resol.pdf') 



fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.)
ax1.set_xscale('log') 
#ax1.set_yscale('log') 
ax1.scatter(time_resolution, F0, s=5)  
ax1.set_xlabel('Time Resolution', fontsize = 12)
ax1.set_ylabel('Fluence Threshold', fontsize = 12)
ax1.set_title('width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
#fig1.savefig('F0_vs_time_resol.pdf') 
#plt.close() 

# to be done: move some of the optional argument from the function to the main.
print 'all time should be in ms.'

'''
my_w_obs = compute_width_obs(my_DM, my_w_int, my_frequency, my_delta_frequency)
my_flux_obs = compute_flux_obs(my_flux_int, my_w_obs, my_time_resolution) 
my_S2N = S2N(my_flux_obs, my_noise_1, my_w_obs, my_time_resolution) 
my_rate = rate(my_flux_int, my_DM, my_w_int, my_S2N, my_S2N_min, my_events_rate)
'''



