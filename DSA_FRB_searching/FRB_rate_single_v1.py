#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:53:54 2019

@author: gechen

FRB_rate_single_v1.py: computes the detection rate of a single FRB. 
Search after de-dispersion trails. 
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


def Compute_w_DM(DM, frequency_central=1405, channel_width=0.122):
    '''
    Compute dispersion smearing using the exact DM, i.e. infinite DM samples
    frequency_central: central frequency in MHz (1280-1530 MHz)
    channel_width: in MHz (bandwidth 250 MHz/ 2048 channels = 0.122 MHz)
    returns dispersion broadening in ms.
    '''
    w_DM = 8.3 *1e6 * DM * channel_width * frequency_central ** (-3) # single channel 
    return w_DM 


def Compute_w_eff(w_int, DM):
    '''
    Effective width: width after de-dispersion. 
    All in ms. 
    '''
    w_DM = Compute_w_DM(DM) 
    w_eff = np.sqrt(w_int ** 2 + w_DM ** 2)
    return w_eff 
    

def Compute_flux_noise(w_int, DM, time_resolution, T_sys = 35, bandwidth = 250):
    '''
    Compute the flux noise level of one reading (width < time resolution) in Jy.
    # w_int: in ms 
    # time_resolution: in ms 
    # T_sys: in Kelvin
    # bandwidth: in MHz
    '''
    w_eff = Compute_w_eff(w_int, DM) 
    Kb = 1381 # in m^2*Jy/Kelvin 
    efficiency = 0.65 
    n = 85 # what is this? 
    D = 4.65 # diameter in m 
    A_eff = efficiency * n * np.pi * (D/2.)**2 # Effective area 
    gain = A_eff / (2 * Kb) # in K/Jy 
    SEFD = T_sys / gain # 103 Jy for the default argument values   
    polarization = 2 
    noise_single = SEFD / np.sqrt(polarization * bandwidth * 1e3 * time_resolution) #Flux noise of a single measurement. 
    #n = int(w_dedispersed / time_resolution) + 1 # number of flux samples 
    n = w_eff / time_resolution # see review paper 2 
    flux_noise = noise_single / np.sqrt(n) # noise reduced by muiltiple samples 
    return flux_noise # single sample noise = 0.40234375 Jy, for the default values 


def Compute_S2N(fluence, DM, w_int, time_resolution):
    '''
    Compute the observed mean flux from the fluence, and the S/N. 
    fluence: Jin y ms
    w_int, time_resolution: in ms. 
    '''
    w_eff = Compute_w_eff(w_int, DM) 
    if w_eff <= time_resolution:
        t_burst = time_resolution 
    else:
        t_burst = w_eff # matched filter: moving box convolves with signal. Box width should maximize signal?  
        #t_burst = time_resolution * (int(w_obs / time_resolution) + 1) # rescale the width to integer number * time resolution.   
    
    flux_obs = fluence / t_burst
    noise = Compute_flux_noise(w_int, DM, time_resolution) 
        
    return flux_obs / noise 


def Compute_F0(w_int, DM, time_resolution, S2N_min=8):
    '''
    compute fluence threshold for a given time resolution and width.
    '''
    w_eff = Compute_w_eff(w_int, DM) 
    flux_noise = Compute_flux_noise(w_int, DM, time_resolution) 
    
    if w_eff <= time_resolution: 
        F0 = flux_noise * S2N_min * time_resolution 
    else: 
        F0 = flux_noise * S2N_min * w_eff  

    return F0  


def Rate_integrand(w_int, DM, time_resolution, DM_stepsize):
    '''
    Integrate this function to get the detection rate.
    '''    
    F_0 = Compute_F0(w_int, DM, time_resolution) # fluence threshold for a given width     
    event_rate_above_F_0 = 1 - Fluence_cdf(F_0) # N[F>F0(w, DM)]  
    
    return event_rate_above_F_0 * Width_intrinsic_pdf(w_int) * DM_pdf(DM) 

    
def Compute_detection_rate(time_resolution, DM_stepsize=0, f=Rate_integrand): 
    '''
    Total number of detectable events per day for the instrument. 
    '''
    return integrate.dblquad(f, 0, np.inf, lambda x: 0, lambda x: np.inf, args=[time_resolution, DM_stepsize], epsabs=1e-4, epsrel=1e-4) 
    

# test integration speed
Compute_detection_rate(1e-3, Rate_integrand) 
# -- main -- 
# FRB population: DM mu=544, sigma=406, w_int mu=1.85, sigma=2.58 
rate = np.array([]) 
rate_err = np.array([]) 
time_resolution = np.logspace(-3, 0, num=20) # 1 microsec to 1 millisec
# Use some typical width, fluence, and DM values to test the noise and S/N functions.
my_w_int = 0.5 # width in ms
my_F = 1 # fluence 
my_DM = 100
S2N = np.array([])
F0 = np.array([]) 


for t in time_resolution:  # in ms 
    print 'time resol=%.3f ms,'%t, 'noise=%.2f Jy ms,'%Compute_flux_noise(my_w_int, my_DM, t), \
    'S/N=%.1f,'%Compute_S2N(my_F, my_DM, my_w_int, t), \
    'rate=', Compute_detection_rate(t, Rate_integrand) # (integral result, error)
    S2N = np.append(S2N, Compute_S2N(my_F, my_DM, my_w_int, t))
    F0 = np.append(F0, Compute_F0(my_w_int, my_DM, t))
    rate = np.append(rate, Compute_detection_rate(t, Rate_integrand)[0]) 
    rate_err = np.append(rate_err, Compute_detection_rate(t, Rate_integrand)[1]) 
    

    
fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.)
ax1.set_xscale('log') 
ax1.set_yscale('log') 
ax1.scatter(time_resolution, rate, s=5) 
ax1.set_xlabel(r'Time resolution [ms]', fontsize = 12) 
ax1.set_ylabel(r'Detection rate [%]', fontsize = 12) 
ax1.set_title('Rate vs Time Resolution', fontsize = 12) 
fig1.savefig('Rate_vs_time_resol.pdf') 
#plt.close() 


fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.)
ax1.set_xscale('log') 
ax1.set_yscale('log') 
#ax1.set_ylim(top = 15)
ax1.scatter(time_resolution, S2N, s=5)  
ax1.set_xlabel('Time Resolution [ms]', fontsize = 12)
ax1.set_ylabel('S/N', fontsize = 12)
ax1.set_title('S/N vs Time Resolution \n width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
fig1.savefig('S2N_vs_time_resol.pdf') 



fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.)
ax1.set_xscale('log') 
#ax1.set_yscale('log') 
ax1.scatter(time_resolution, F0, s=5)  
ax1.set_xlabel('Time Resolution [ms]', fontsize = 12)
ax1.set_ylabel('Fluence Threshold [Jy ms]', fontsize = 12)
ax1.set_title('Fluence Threshold vs Time Resolution \n width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
fig1.savefig('F0_vs_time_resol.pdf') 
#plt.close() 

print 'all time should be in ms.'


'''
my_w_obs = compute_width_obs(my_DM, my_w_int, my_frequency, my_delta_frequency)
my_flux_obs = compute_flux_obs(my_flux_int, my_w_obs, my_time_resolution) 
my_S2N = S2N(my_flux_obs, my_noise_1, my_w_obs, my_time_resolution) 
my_rate = rate(my_flux_int, my_DM, my_w_int, my_S2N, my_S2N_min, my_events_rate)

def Dedisperse(w_obs, DM):
    w_DM = min(w_obs, Compute_dispersion_smearing(DM))
    w_dedispersed = np.sqrt(w_obs ** 2 - w_DM ** 2) 
    return w_dedispersed 


def Compute_w_DM_stepsize(DM, DM_stepsize):
    return 0 

def Compute_w_intrinsic(w_obs, DM, DM_stepsize):
    w_DM = Compute_dispersion_smearing(DM) 
    w_DM_stepsize = 0
    w_int = np.sqrt(w_obs ** 2 - w_DM ** 2 - w_DM_stepsize ** 2) 
    return w_int 
    
'''

