#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:18:16 2019
FRB_rate_single_v5.py: computes the detection rate of a single FRB in this hemisphere. 
@author: gechen

V1: rate vs. time resolution 
V2: adds grid plot for rate(time resolution, channel width)
V3: adds dispersion smearing due to the DM trail step size. 
v4: adds single dish sky power response distribution into the rate integral. 
v5: adds beam forming sky distribution into the rate integral.

Seperate the power resoponse integral and the rest, which is space-independent. 
"""
import numpy as np 
from scipy import stats 
from scipy import integrate
import matplotlib.pyplot as plt
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


def Events_above_F0(F_0, F_b=15, F_low=0.5, F_up = 1e5, alpha = -1.18, beta = -2.2):
    '''
    Compute R(>F0) 
    model: James et al. 2019 'The slope of the source-count distribution for fast radio bursts' 
    
    Check F_low and find a good reference. 
    
    #F_b: fluence turning point is between 5 tp 40 Jy ms  
    #F_low: fluence lower limit assuming w= 1 ms (rate = 1000 per day per sky)
    #F_up: fluence upper limit, some arbitery large number      
    #alpha: Parks pdf power index  
    #beta: Askap pdf power index 
    '''
    if F_0 < F_low: 
        R = 0 
    elif F_0 < F_b:
        R = 1000 * (F_0 / F_low) ** alpha 
    else:
        R_b = 1000 * (F_b / F_low) ** alpha
        R = R_b * (F_0 / F_b) ** beta 
    
    return R  


def Beam_forming_shape(theta, sky_angle, beam_center_space=0.04, sigma_prime=0.012, mu_envelop=0.0, sigma_envelop=1.65):
    '''
    Beam forming in 1D. 
    Normalized to peak at 100% at the zenith. 
    Approximate each prime beams as narrow Gaussians (mu_prime is a freee parameter, sigma_prime=0.012)
    Approximate the envelop using the single dish response (mu_envelop=0, sigma_envelop=1.65)
    
    sky_angle: sky region radius from the zenith. 
    beam_center_space: space between adjacent prime beams. Any rule? 
    beam_center_space=0.04, sigma_prime=0.012, mu_envelop=0.0, sigma_envelop=1.65, all in degrees
    '''
    beam_num = 2 * sky_angle / beam_center_space # number of prime beams to form in the resion of interest. 
    mu_prime_repeat = np.linspace(-1*sky_angle, sky_angle, num=int(beam_num)+1) 
    mu_prime = mu_prime_repeat[np.argmin(abs(theta-mu_prime_repeat))] 
    p_prime_repeat = stats.norm.pdf(theta, mu_prime, sigma_prime) / stats.norm.pdf(mu_prime, mu_prime, sigma_prime) 
    
    p_envelop = stats.norm.pdf(theta, mu_envelop, sigma_envelop) / stats.norm.pdf(mu_envelop, mu_envelop, sigma_envelop) # envelop 
    p_beam_forming = p_prime_repeat * p_envelop  
    
    return p_beam_forming 

    

def Compute_w_DM(DM, channel_number, frequency_central=1405):
    '''
    Compute dispersion smearing using the exact DM, i.e. infinite DM samples
    frequency_central: current central frequency in MHz (1280-1530 MHz)
    channel_width: in MHz (current channel width 250 MHz/ 2048 channels = 0.122 MHz)
    returns dispersion broadening in ms.
    '''
    channel_width = 250.0 / channel_number
    w_DM = 8.3 *1e6 * DM * channel_width * frequency_central ** (-3) # single channel 
    return w_DM 


def Compute_DM_grid(channel_number, time_resolution, DM_min, DM_max, t_i = 0.064, epsilon = 1.25, frequency_central_GHz=1.405):
    '''
    Compute the DM values to use in the de-dispersion trail.
    time_resolution: in ms 
    t_i: typical intrinsic width (ms). Use a small number to get small enough spacing. 
    frequency_central=1405 MHz.
    epsilon: tolerence. 
    '''
    DM = DM_min 
    DM_grid = np.array([DM_min]) 
    t_samp = time_resolution  
    alpha = 1./(16 + channel_number ** 2) 
    beta = (t_i ** 2 + t_samp ** 2) * 1e6
    B = 250.0 / channel_number # MHz, channel bandwidth 
    
    while DM <= DM_max:
        DM = channel_number ** 2 * alpha * DM + np.sqrt(16 * alpha * (epsilon ** 2 - channel_number ** 2 * alpha) * DM ** 2 + 16 * alpha * beta * (epsilon ** 2 - 1) * (frequency_central_GHz ** 3 / (8.3 * B)) ** 2) # Lina Levin Thesis Eqn. 2.5 
        DM_grid = np.append(DM_grid, DM)
        
    return DM_grid
    
    

def Compute_w_delta_DM(DM_real, channel_number, time_resolution, DM_min, DM_max, frequency_central=1405): 
    '''
    Compute the dispersion smearing due to the finite DM trail values.
    frequency_central in MHz 
    DM_real: one number 
    Returns an array. 
    '''
    DM_grid = Compute_DM_grid(channel_number, time_resolution, DM_min, DM_max)
    B = 250.0 / channel_number # channel bandwidth in MHz 
    delta_DM = min(abs(DM_real - DM_grid))
    w_delta_DM = 1e6 * 8.3 * channel_number * B * delta_DM / (4 * frequency_central ** 3) # in ms 
    return w_delta_DM
    

def Compute_w_eff(w_int, DM, channel_number, time_resolution, DM_min, DM_max, include_w_delta_DM = False):
    '''
    Effective width: width after de-dispersion. 
    All width in ms. 
    '''
    w_DM = Compute_w_DM(DM, channel_number) 
    
    if include_w_delta_DM == True: 
        w_delta_DM = Compute_w_delta_DM(DM, channel_number, time_resolution, DM_min, DM_max) 
    else: 
        w_delta_DM = 0 
    
    w_eff = np.sqrt(w_int ** 2 + time_resolution ** 2 + w_DM ** 2 + w_delta_DM ** 2) 
    return w_eff 
    

def Compute_flux_noise(w_int, DM, time_resolution, channel_number, DM_min, DM_max, T_sys = 35, bandwidth = 250):
    '''
    Compute the flux noise level of one reading (width < time resolution) in Jy.
    # w_int: in ms 
    # time_resolution: in ms 
    # T_sys: in Kelvin
    # bandwidth: in MHz 
    '''
    w_eff = Compute_w_eff(w_int, DM, channel_number, time_resolution, DM_min, DM_max) 
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


def Compute_S2N(fluence, DM, w_int, time_resolution, channel_number, DM_min, DM_max):
    '''
    Compute the observed mean flux from the fluence, and the S/N. 
    fluence: Jin y ms
    w_int, time_resolution: in ms. 
    '''
    w_eff = Compute_w_eff(w_int, DM, channel_number, time_resolution, DM_min, DM_max) 
    if w_eff <= time_resolution:
        t_burst = time_resolution 
    else:
        t_burst = w_eff # matched filter: moving box convolves with signal. Box width should maximize signal?  
        #t_burst = time_resolution * (int(w_obs / time_resolution) + 1) # rescale the width to integer number * time resolution.   
    
    flux_obs = fluence / t_burst
    noise = Compute_flux_noise(w_int, DM, time_resolution, channel_number, DM_min, DM_max) 
        
    return flux_obs / noise 


def Compute_F0(w_int, DM, time_resolution, channel_number, DM_min, DM_max, S2N_min=8):
    '''
    compute fluence threshold for a given time resolution and width.
    '''
    w_eff = Compute_w_eff(w_int, DM, channel_number, time_resolution, DM_min, DM_max) 
    flux_noise = Compute_flux_noise(w_int, DM, time_resolution, channel_number, DM_min, DM_max) 
    
    if w_eff <= time_resolution: 
        F0 = flux_noise * S2N_min * time_resolution 
    else: 
        F0 = flux_noise * S2N_min * w_eff 

    return F0  


def Rate_integrand(w_int, DM, time_resolution, channel_number, DM_min, DM_max):
    '''
    Integrate this function to get the detection rate.
    '''    
    F_0 = Compute_F0(w_int, DM, time_resolution, channel_number, DM_min, DM_max) # fluence threshold for a given width     
    event_rate_above_F_0 = Events_above_F0(F_0) # N[F>F0(w, DM)]  
    
    # for debug or record
    with open('../../outputs_txt/Integration_track_'+version+'.txt', 'a') as f: 
        print >>f, '#### Discrete DM sampling ######'
        print >>f, time.strftime("%Y-%m-%d %H:%M:%S") # print date and time 
        print >>f, 'w_int=',w_int, 'DM=',DM, 'F0=', F0 
        #print >>f, 'w_int=',w_int, 'DM=',DM, 'theta=',theta, 'F0=', F0 
    
    return event_rate_above_F_0 * Width_intrinsic_pdf(w_int) * DM_pdf(DM) 



def Compute_detection_rate(time_resolution, channel_number, DM_min, DM_max, beam_forming_integral, f=Rate_integrand): 
    '''
    Total number of detectable events per day for the instrument. 
    Return the double (definite) integral of func(y, x) from x = a..b and y = gfun(x)..hfun(x).
    Note that the order of arguments are counter-intuitive: (y, x)
    '''
    rate_for_perfect_response = integrate.dblquad(f, DM_min, DM_max, lambda x: 0, lambda x: np.inf, args=[time_resolution, channel_number, DM_min, DM_max], epsabs=1e-4, epsrel=1e-4) 
    return beam_forming_integral * rate_for_perfect_response[0],  beam_forming_integral * rate_for_perfect_response[1]


'''
def Compute_detection_rate_riemann_sum(time_resolution, channel_number, DM_min, DM_max, beam_forming_integral):
    DM_arr = np.linspace(DM_min, DM_max, num=50) 
    w_arr = np.logspace(w_order_min, w_order_max, num=50)
    DM_mid_arr = 0.5 * (DM_arr[0:-1] + DM_arr[1:])
    w_mid_arr = 0.5 * (w_arr[0:-1] + w_arr[1:])
    rate = 0.0 
    for i in range(len(w_arr)):
        delta_w = w_arr[i+1] - w_arr[i]
        for j in range(len(DM_arr):
            delta_DM = DM_arr[j+1] - DM_arr[j] 
            rate += delta_w * delta_DM * Rate_integrand(w_mid[i], DM_mid[j], time_resolution, channel_number, DM_min, DM_max)
    
    return rate * beam_forming_integral 
'''    
    

# -- main -- 
version = 'v5' # code version 
# FRB population: DM mu=544, sigma=406, w_int mu=1.85, sigma=2.58 
my_bandwidth = 250.0 # MHz 
my_DM_min = 0.0 
my_DM_max = 5000.0
my_sky_angle = 5.0 
my_beam_forming_integral = integrate.quad(Beam_forming_shape, -1*my_sky_angle,  my_sky_angle, args=my_sky_angle)[0] / 180 

# test integration speed
#Compute_detection_rate(1e-3, 2048, my_DM_min, my_DM_max, Rate_integrand) 


time_resolution_edges = np.logspace(-3.0, 0.0, num=10) # 1 microsec to 1 millisec
my_channel_number_edges = np.logspace(11, 6, num=12, base=2.0) # N <= 2048
my_channel_width_edges = my_bandwidth / my_channel_number_edges

time_resolution = 0.5*(time_resolution_edges[0:-1] + time_resolution_edges[1:])
my_channel_width = 0.5*(my_channel_width_edges[0:-1] + my_channel_width_edges[1:])
my_channel_number = 0.5*(my_channel_number_edges[0:-1] + my_channel_number_edges[1:])


# Use some typical width, fluence, and DM values to test the noise and S/N functions.
my_w_int = 0.5 # width in ms
my_F = 1 # fluence 
my_DM = 300

rate = np.array([]) 
rate_err = np.array([]) 
S2N = np.array([])
F0 = np.array([]) 


for t in time_resolution:  # in ms 
    for n_ch in my_channel_number: 
        noise_element = Compute_flux_noise(my_w_int, my_DM, t, n_ch, my_DM_min, my_DM_max) 
        S2N_element = Compute_S2N(my_F, my_DM, my_w_int, t, n_ch, my_DM_min, my_DM_max)
        F0_element = Compute_F0(my_w_int, my_DM, t, n_ch, my_DM_min, my_DM_max)
        rate_element = Compute_detection_rate(t, n_ch, my_DM_min, my_DM_max, my_beam_forming_integral, Rate_integrand)
        
        S2N = np.append(S2N, S2N_element)
        F0 = np.append(F0, F0_element)
        rate = np.append(rate, rate_element[0]) 
        rate_err = np.append(rate_err, rate_element[1]) 
        
        with open('../../outputs_txt/rate_results_'+version+'.txt', 'a') as f: 
            print >>f, '#### Rate with Discrete DM sampling ######'
            print >>f, time.strftime("%Y-%m-%d %H:%M:%S") # print date and time 
            print >>f, 'Time resol=%.3f ms,'%t, 'channel width=%.4f MHz,'%n_ch, 'noise=%.2f Jy ms,'%noise_element, \
            'S/N=%.1f,'%S2N_element, \
            'rate=', rate_element # (integral result, error)
        
        print 'Time resol=%.3f ms,'%t, 'channel width=%.4f MHz,'%n_ch, 'noise=%.2f Jy ms,'%noise_element, \
            'S/N=%.1f,'%S2N_element, \
            'rate=', rate_element # (integral result, error)
        
    
    
    
# only changes time resolution
if len(my_channel_width)==1 and len(time_resolution)>1:
    print "Fixed channel width=", my_channel_width[0] 
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.)
    ax1.set_xscale('log') 
    ax1.set_yscale('log') 
    ax1.scatter(time_resolution, rate, s=5) 
    ax1.set_xlabel(r'Time resolution [ms]', fontsize = 12) 
    ax1.set_ylabel(r'Detection rate', fontsize = 12) 
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
    #fig1.savefig('F0_vs_time_resol_beam_forming.pdf') 
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
    ax1.set_ylabel(r'Detection rate', fontsize = 12) 
    ax1.set_title('Rate vs Channel width', fontsize = 12) 
    fig1.savefig('Rate_vs_channel_width_beam_forming.pdf') 
    #plt.close() 
    
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.)
    ax1.set_xscale('log') 
    ax1.set_yscale('log') 
    ax1.scatter(my_channel_width, F0, s=5)  
    ax1.set_xlabel('Channel_width [MHz]', fontsize = 12)
    ax1.set_ylabel('Fluence Threshold [Jy ms]', fontsize = 12)
    ax1.set_title('F_0 vs channel width \n width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
    fig1.savefig('F0_vs_channel_width_beam_forming.pdf') 
    #plt.close() 
        
    fig1, ax1 = plt.subplots() 
    fig1.set_size_inches(8., 6.)
    ax1.set_xscale('log') 
    ax1.set_yscale('log') 
    ax1.scatter(my_channel_width, S2N, s=5)  
    ax1.set_xlabel('Channel width [MHz]', fontsize = 12)
    ax1.set_ylabel('S/N', fontsize = 12)
    ax1.set_title('S/N vs Channel width \n width = %.3f ms, DM=%.3f'%(my_w_int, my_DM), fontsize = 12) 
    fig1.savefig('S2N_vs_channel_width_beam_forming.pdf') 

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
    char.set_label('Rate', fontsize = 12) # colorbar label.
    ax1.set_xlabel(r'Channel width [MHz]', fontsize = 12) 
    ax1.set_ylabel(r'Time resolution [ms]', fontsize = 12) 
    ax1.set_title('Rate vs. (channel width and time resolution)', fontsize = 14)   
    fig1.savefig('Rate_grid_beam_forming.pdf')  
    
else: 
    print 'rate=', rate


print 'all time should be in ms.'


# test the effect of w_delta_DM 
test_channel_num = 1000
test_w_int = 0.5 
test_time_resolution = 1e-2 
DM_grid = Compute_DM_grid(test_channel_num, test_time_resolution, 0, 3e3) 
DM_real = np.linspace(0, 3e3,num=200) 
w_delta_DM = np.array([]) 
w_DM_and_t_samp = np.array([])
w_eff = np.array([]) 
for DM in DM_real:
    w_delta_DM = np.append(w_delta_DM, Compute_w_delta_DM(DM, test_channel_num, test_time_resolution, 0, 3e3)) 
    w_DM_and_t_samp = np.append(w_DM_and_t_samp, Compute_w_eff(my_w_int, DM, test_channel_num, test_time_resolution, 0, 3e3, include_w_delta_DM = False))
    w_eff = np.append(w_eff, Compute_w_eff(my_w_int, DM, test_channel_num, test_time_resolution, 0, 3e3, include_w_delta_DM = True))

w_DM = Compute_w_DM(DM_real, test_channel_num) # dispersion smearing without adding sampling time 

fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.) 
ax1.tick_params(labelsize=12) 
ax1.plot(DM_real, w_delta_DM, 'c-', label='w_delta_DM along') 
ax1.plot(DM_real, w_DM, 'b:', label='w_DM along')
ax1.plot(DM_real, w_DM_and_t_samp, 'k--', label='w_eff without w_delta_DM')
ax1.plot(DM_real, w_eff, 'r', label='w_eff total')
ax1.legend()
ax1.set_xlabel('DM')
ax1.set_ylabel('width [ms]') 
ax1.set_title(r'Dispersion smearing due to DM, $\Delta$DM and $t_{samp}$')
#fig1.savefig('w_delta_DM_test_with_t_samp.pdf') 

# test beam forming result
theta=np.linspace(-1*my_sky_angle, my_sky_angle, 3001) 
p = [Beam_forming_shape(item, my_sky_angle) for item in theta] 

fig3, ax3 = plt.subplots() 
fig3.set_size_inches(8., 6.) 
ax3.tick_params(labelsize=12) 
ax3.plot(theta, p)
ax3.set_xlabel('Angle from the Zenith (degree)')
ax3.set_ylabel('Power Response') 
ax3.set_title('Beam forming approximation') 
fig3.savefig('Beam_forming_approximation.pdf')


# test fluence cdf distribution 
F=np.linspace(0.5,100,num=500)
F_cdf = [Fluence_cdf(i) for i in F] 

fig3, ax3 = plt.subplots() 
fig3.set_size_inches(8., 6.) 
ax3.tick_params(labelsize=12) 
ax3.plot(F, F_cdf)
ax3.grid()
ax3.set_xlabel('Fluence [Jy ms]')
ax3.set_ylabel('cdf(F)')
ax3.set_title('Fluence Cumulative Distribution') 
fig3.savefig('F_cdf.pdf')
