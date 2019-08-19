#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:53:54 2019

@author: gechen

FRB_rate_single_v2.py: computes the detection rate of a single FRB. 
Search after de-dispersion. 
V2: Adds grid plot for rate(time resolution, channel width)
"""
import numpy as np 
from scipy import stats 
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
import time


np.set_printoptions(threshold=np.nan) #print full array 
# Note that matplotlib v2.0.0 has smaller default figure size... 
plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 


def DM_pdf(DM, mu=563, sigma=442):
    '''Gaussian distributions from fitting results (FRB_population_v1.py)'''
    return stats.norm.pdf(DM, mu, sigma) # gaussian 

def Width_intrinsic_pdf(w_int, mu=1.85, sigma=3.03):
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


def Compute_w_DM(DM, channel_width, frequency_central=1405):
    '''
    Compute dispersion smearing using the exact DM, i.e. infinite DM samples
    frequency_central: current central frequency in MHz (1280-1530 MHz)
    channel_width: in MHz (current channel width 250 MHz/ 2048 channels = 0.122 MHz)
    returns dispersion broadening in ms.
    '''
    w_DM = 8.3 *1e6 * DM * channel_width * frequency_central ** (-3) # single channel 
    return w_DM 


def Compute_w_eff(w_int, DM, channel_width):
    '''
    Effective width: width after de-dispersion. 
    All in ms. 
    '''
    w_DM = Compute_w_DM(DM, channel_width) 
    w_eff = np.sqrt(w_int ** 2 + w_DM ** 2)
    return w_eff 
    

def Compute_flux_noise(w_int, DM, time_resolution, channel_width, T_sys = 35, bandwidth = 250):
    '''
    Compute the flux noise level of one reading (width < time resolution) in Jy.
    # w_int: in ms 
    # time_resolution: in ms 
    # T_sys: in Kelvin
    # bandwidth: in MHz
    '''
    w_eff = Compute_w_eff(w_int, DM, channel_width) 
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


def Compute_S2N(fluence, DM, w_int, time_resolution, channel_width):
    '''
    Compute the observed mean flux from the fluence, and the S/N. 
    fluence: Jin y ms
    w_int, time_resolution: in ms. 
    '''
    w_eff = Compute_w_eff(w_int, DM, channel_width) 
    if w_eff <= time_resolution:
        t_burst = time_resolution 
    else:
        t_burst = w_eff # matched filter: moving box convolves with signal. Box width should maximize signal?  
        #t_burst = time_resolution * (int(w_obs / time_resolution) + 1) # rescale the width to integer number * time resolution.   
    
    flux_obs = fluence / t_burst
    noise = Compute_flux_noise(w_int, DM, time_resolution, channel_width) 
        
    return flux_obs / noise 


def Compute_F0(w_int, DM, time_resolution, channel_width, S2N_min=8):
    '''
    compute fluence threshold for a given time resolution and width.
    '''
    w_eff = Compute_w_eff(w_int, DM, channel_width) 
    flux_noise = Compute_flux_noise(w_int, DM, time_resolution, channel_width) 
    
    if w_eff <= time_resolution: 
        F0 = flux_noise * S2N_min * time_resolution 
    else: 
        F0 = flux_noise * S2N_min * w_eff  
    
    with open('../../outputs_txt/Integration_track_v2.txt', 'a') as f: 
        print >>f, '#### Discrete DM sampling ######'
        print >>f, time.strftime("%Y-%m-%d %H:%M:%S") # print date and time 
        print >>f, 'w_int=',w_int, 'DM=',DM, 'F0=',F0

    return F0  


def Rate_integrand(w_int, DM, time_resolution, channel_width):
    '''
    Integrate this function to get the detection rate.
    '''    
    F_0 = Compute_F0(w_int, DM, time_resolution, channel_width) # fluence threshold for a given width     
    event_rate_above_F_0 = 1 - Fluence_cdf(F_0) # N[F>F0(w, DM)]  
    
    return event_rate_above_F_0 * Width_intrinsic_pdf(w_int) * DM_pdf(DM) 

    
def Compute_detection_rate(time_resolution, channel_width, f=Rate_integrand): 
    '''
    Total number of detectable events per day for the instrument. 
    Note that the order of arguments are counter-intuitive. 
    '''
    return integrate.dblquad(f, 0, np.inf, lambda x: 0, lambda x: np.inf, args=[time_resolution, channel_width], epsabs=1e-4, epsrel=1e-4) 
    

# test integration speed
Compute_detection_rate(1e-3, 0.122, Rate_integrand) 
# -- main -- 
# FRB population: DM mu=544, sigma=406, w_int mu=1.85, sigma=2.58 
rate = np.array([]) 
rate_err = np.array([]) 
time_resolution_edges = np.logspace(-3, 0, num=25) # 1 microsec to 1 millisec
#time_resolution = [0.1]
my_bandwidth = 250.0 # MHz 
#my_channel_number = np.linspace(1e2, 1e4, num=9) # any rules? Currently 2048 channels. 
my_channel_number_edges = np.array([int(i) for i in np.logspace(4, 2, num=26)])
my_channel_width_edges = my_bandwidth / my_channel_number_edges
#my_channel_width = [250.0/2048] # current channel width 

time_resolution = 0.5*(time_resolution_edges[0:-1] + time_resolution_edges[1:])
my_channel_width = 0.5*(my_channel_width_edges[0:-1] + my_channel_width_edges[1:])
my_channel_number = 0.5*(my_channel_number_edges[0:-1] + my_channel_number_edges[1:])

# Use some typical width, fluence, and DM values to test the noise and S/N functions.
my_w_int = 0.5 # width in ms
my_F = 1 # fluence 
my_DM = 300
S2N = np.array([])
F0 = np.array([]) 


for t in time_resolution:  # in ms 
    for w_ch in my_channel_width: # in MHz 
        with open('../../outputs_txt/rate_outputs_v2.txt', 'a') as f: 
            print >>f, '#### Rate with continuous DM sampling ######'
            print >>f, time.strftime("%Y-%m-%d %H:%M:%S") # print date and time 
            print >>f, 'Time resol=%.3f ms,'%t, 'channel width=%.4f MHz,'%w_ch, 'noise=%.2f Jy ms,'%Compute_flux_noise(my_w_int, my_DM, t, w_ch), \
            'S/N=%.1f,'%Compute_S2N(my_F, my_DM, my_w_int, t, w_ch), \
            'rate=', Compute_detection_rate(t, w_ch, Rate_integrand) # (integral result, error)
        S2N = np.append(S2N, Compute_S2N(my_F, my_DM, my_w_int, t, w_ch))
        F0 = np.append(F0, Compute_F0(my_w_int, my_DM, t, w_ch))
        rate = np.append(rate, Compute_detection_rate(t, w_ch, Rate_integrand)[0]) 
        rate_err = np.append(rate_err, Compute_detection_rate(t, w_ch, Rate_integrand)[1]) 
        
# only changes time resolution
if len(my_channel_width)==1 and len(time_resolution)>1:
    print "Fixed channel width=", my_channel_width[0] 
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.)
    ax1.set_xscale('log') 
    ax1.set_yscale('log') 
    ax1.scatter(time_resolution, rate, s=5) 
    ax1.set_xlabel(r'Time resolution [ms]', fontsize = 12) 
    ax1.set_ylabel(r'Detection rate [%]', fontsize = 12) 
    ax1.set_title('Rate vs Time Resolution', fontsize = 12) 
    #fig1.savefig('Rate_vs_time_resol.pdf') 
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
    #fig1.savefig('S2N_vs_time_resol.pdf') 
    
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.)
    ax1.set_xscale('log') 
    #ax1.set_yscale('log') 
    ax1.scatter(time_resolution, F0, s=5)  
    ax1.set_xlabel('Time Resolution [ms]', fontsize = 12)
    ax1.set_ylabel('Fluence Threshold [Jy ms]', fontsize = 12)
    ax1.set_title('Fluence Threshold vs Time Resolution \n width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
    #fig1.savefig('F0_vs_time_resol.pdf') 
    #plt.close() 

# only changes channel width
elif len(my_channel_width)>1 and len(time_resolution)==1:
    print "Fixed time resolution =", time_resolution
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.)
    ax1.set_xscale('log') 
    ax1.set_yscale('log') 
    ax1.scatter(my_channel_width, rate, s=5) 
    ax1.set_xlabel(r'Channel width [MHz]', fontsize = 12) 
    ax1.set_ylabel(r'Detection rate [%]', fontsize = 12) 
    ax1.set_title('Rate vs Channel width', fontsize = 12) 
    fig1.savefig('Rate_vs_channel_width.pdf') 
    #plt.close() 
    
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.)
    ax1.set_xscale('log') 
    ax1.set_yscale('log') 
    ax1.scatter(my_channel_width, F0, s=5)  
    ax1.set_xlabel('Channel_width [MHz]', fontsize = 12)
    ax1.set_ylabel('Fluence Threshold [Jy ms]', fontsize = 12)
    ax1.set_title('F_0 vs channel width \n width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
    fig1.savefig('F0_vs_channel_width.pdf') 
    #plt.close() 
        
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.)
    ax1.set_xscale('log') 
    ax1.set_yscale('log') 
    ax1.scatter(my_channel_width, S2N, s=5)  
    ax1.set_xlabel('Channel width [MHz]', fontsize = 12)
    ax1.set_ylabel('S/N', fontsize = 12)
    ax1.set_title('S/N vs Channel width \n width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
    fig1.savefig('S2N_vs_channel_width.pdf') 

# grid plot 
elif len(my_channel_width)>1 and len(time_resolution)>1:
    #xg,yg=np.meshgrid(my_channel_width,time_resolution);
    #zg = np.reshape(rate, (len(my_channel_width),len(time_resolution))) #row=len(my_channel_width) 
    zg = np.reshape(rate, (len(time_resolution), len(my_channel_width)))
    
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.) 
    ax1.tick_params(labelsize=12) 
    ax1.set_xscale('log') 
    ax1.set_yscale('log')
    #img = ax1.contourf(my_channel_width, time_resolution, zg) # len(X) == M is the number of columns in Z and len(Y) == N is the number of rows in Z
    #img = ax1.imshow(zg, alpha=0.7, interpolation='bicubic',cmap='terrain') 
    #img = ax1.imshow(zg, extent=(np.amin(my_channel_width), np.amax(my_channel_width), np.amin(time_resolution), np.amax(time_resolution)), origin='lower', aspect='auto') #norm=LogNorm()
    img = ax1.pcolormesh(my_channel_width_edges, time_resolution_edges, zg) #len(x) = row(z)+1 
    char = fig1.colorbar(img, ax=ax1) 
    char.set_label('Rate [%]', fontsize = 12) # colorbar label.
    ax1.set_xlabel(r'Channel width [MHz]', fontsize = 12) 
    ax1.set_ylabel(r'Time resolution [ms]', fontsize = 12) 
    ax1.set_title('Rate vs. (channel width and time resolution)', fontsize = 14)   
    fig1.savefig('Rate_grid.pdf')  
    
else: 
    print 'Check channel width and time resolution arrays.'


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
# do not need to interpolate here. Already a grid. 
dg=interpolate.griddata((coordinates[:, 0], coordinates[:, 1]), y, (xg,yg), method='nearest');
    
'''

