#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:23:34 2019
@author: gechen

FRB_population_v1.py: this code fits for the DM, width and flux distributions of 77 FRBs.
FRB data from FRBCAT.org (Jul. 2, 2019) ('FRB_cat.txt') 
Fitting the cumulative histogram requires less parameters than fitting the histogram. 
"""

import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats 
import numpy as np 
from scipy.optimize import curve_fit 


def hist_cdf_array(data, name, bin_num, hist_range=None, weights=None, normalized=None): 
    '''
    this function returns data histogram and cdf arrays, and save the two plots. 
    it does not normalize the two arrays. 
    some arguments: 
        data: e.g. DM, width, flux 
        name: a string 
    '''
    hist_counts, bin_edges = np.histogram(data, bin_num, range=hist_range, weights=weights, density=normalized) 
    c_hist = np.cumsum(hist_counts) 
    x_mid = 0.5*(bin_edges[1:] + bin_edges[:-1]) 
    bin_width = 0.98*(bin_edges[1]-bin_edges[0])
    #x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size,res.cumcount.size) 
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    ax1.bar(x_mid, hist_counts, yerr=np.sqrt(hist_counts), width=bin_width) 
    ax1.set_title(name + ' Histogram')
    ax2.bar(x_mid, c_hist, yerr=np.sqrt(c_hist), width=bin_width) 
    ax2.set_title(name + ' Cumulative histogram')
    ax2.set_xlim([bin_edges.min(), bin_edges.max()])
    
    fig.savefig(name + '_raw.pdf')  
    #plt.close(fig)

    return x_mid, hist_counts, c_hist 

def normalize(hist, c_hist, x): 
    '''
    normalize each array with the total counts number 
    sum(hist) = cdf[-1]
    '''
    area = (x[1]-x[0])*sum(hist)
    hist_norm = 1.*hist/area 
    c_hist_norm = 1.*c_hist/c_hist[-1] 
    hist_err_norm = np.sqrt(hist)/area 
    c_hist_err_norm = np.sqrt(c_hist)/c_hist[-1]
    return hist_norm, c_hist_norm, hist_err_norm, c_hist_err_norm


def gaussian_cdf(x, mu, sigma):
    return stats.norm.cdf(x, mu, sigma)


def gaussian2_cdf(x, mu1, sigma1, mu2, sigma2):
    return stats.norm.cdf(x, mu1, sigma1) + stats.norm.cdf(x, mu2, sigma2)


def plot_data_and_fit(name, x_mid, hist_norm, c_hist_norm, hist_err_norm, c_hist_err_norm, popt, pdf=stats.norm.pdf, cdf=stats.norm.cdf): 
    '''
    Overplot the cdf fit result curves on top of the normalized histogram and cumulative histogram 
    Some arguments: 
        name: string 
        popt: curve_fit results array 
    '''
    bin_width = 0.98*(x_mid[1]-x_mid[0])
    
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    ax1.bar(x_mid, hist_norm, yerr=hist_err_norm, width=bin_width) 
    ax1.plot(x_mid, pdf(x_mid, popt[0], popt[1]), color='r')
    ax1.set_title(name + ' Histogram (normalized)')
    ax2.bar(x_mid, c_hist_norm, yerr=c_hist_err_norm, width=bin_width) 
    ax2.plot(x_mid, cdf(x_mid, popt[0], popt[1]), color='r')
    ax2.set_title(name + ' Cumulative histogram (normalized)')
    #ax2.set_xlim([bin_edges.min(), bin_edges.max()]) 
    
    fig.savefig(name + '_fit.pdf')  
    #plt.close(fig)

def Compute_w_int(w_eff, DM, frequency_central, channel_width):
    '''
    Compute dispersion smearing using the exact DM, i.e. infinite DM samples
    frequency_central: central frequency in MHz (1280-1530 MHz)
    channel_width: in MHz 
    returns dispersion broadening in ms.
    '''
    w_DM = 8.3 *1e6 * DM * channel_width * frequency_central ** (-3) # single channel 
    w_int = np.sqrt(w_eff ** 2 - w_DM ** 2)
    return w_int 


# read in FRB data 
my_csv = pd.read_csv('FRB_cat.txt')
print my_csv.columns 
width = my_csv.rmp_width
#DM: split into two columns by "&plusminus" and convert strings to floats 
DM = pd.to_numeric(my_csv.rmp_dm.str.split(pat='&plusmn',expand=True)[0]) 
DM_err = pd.to_numeric(my_csv.rmp_dm.str.split(pat='&plusmn',expand=True)[1]) # has some Nan
flux_full = pd.to_numeric(my_csv.rmp_flux, errors='coerce') # has some "null" 

fluence_full = width * flux_full 
# sort each series to decide on histogram range 
width.sort_values(ascending=True) # has 3 outliers much larger than the other 74 
DM.sort_values(ascending=True) # default range is okay 
flux_full.sort_values(ascending=True) 
fluence_full.sort_values(ascending=True) 

# width: no error bar, no null or Nan, but 3 outliers (too large to include for eaqually binned histogram) 
width_min = 0
width_high = 35 
# DM has error bar, but soma errorbars are Nan so I ignore them  
# flux has null values, and some are lower limits? 
# read Shannol 2018 paper: didn't see relavent contents? 
flux = flux_full[~np.isnan(flux_full)] # get rid of "null" 
fluence = fluence_full[~np.isnan(fluence_full)] 
fluence_min = 0
fluence_max = 410

# get the data histogram and c_hist, save the figures and arrays.
x_mid_width, hist_counts_width, cdf_width = hist_cdf_array(width,'width', 32, hist_range=(width_min, width_high)) 
x_mid_DM, hist_counts_DM, cdf_DM = hist_cdf_array(DM,'DM', 32) 
x_mid_flux, hist_counts_flux, cdf_flux = hist_cdf_array(flux,'flux', 32) 
x_mid_fluence, hist_counts_fluence, cdf_fluence = hist_cdf_array(fluence,'fluence', 32, hist_range=(fluence_min, fluence_max)) 
# normalize the histogram and cumulative histogram arrays for the fits
hist_norm_width, cdf_norm_width, hist_err_norm_width, c_hist_err_norm_width = normalize(hist_counts_width, cdf_width, x_mid_width)
hist_norm_DM, cdf_norm_DM, hist_err_norm_DM, c_hist_err_norm_DM = normalize(hist_counts_DM, cdf_DM, x_mid_DM)
hist_norm_flux, cdf_norm_flux, hist_err_norm_flux, c_hist_err_norm_flux = normalize(hist_counts_flux, cdf_flux, x_mid_flux)
hist_norm_fluence, cdf_norm_fluence, hist_err_norm_fluence, c_hist_err_norm_fluence = normalize(hist_counts_fluence, cdf_fluence, x_mid_fluence)

# best guess by eyes 
width_guess = [1.5, 2.5] # mean, sigma 
popt_width, pcov_width = curve_fit(gaussian_cdf, x_mid_width, cdf_norm_width, p0=width_guess) 
perr_width = np.sqrt(np.diag(pcov_width)) # Calculate the error bars from sqrt(covariance matrix) 
chisq_width, p_width = stats.chisquare(cdf_norm_width,gaussian_cdf(x_mid_width, popt_width[0], popt_width[1]), ddof=2)

DM_guess = [400, 500] # mean, sigma  
popt_DM, pcov_DM = curve_fit(gaussian_cdf, x_mid_DM, cdf_norm_DM, p0=DM_guess) 
perr_DM = np.sqrt(np.diag(pcov_DM)) # Calculate the error bars from sqrt(covariance matrix) 
chisq_DM, p_DM = stats.chisquare(cdf_norm_DM,gaussian_cdf(x_mid_DM, popt_DM[0], popt_DM[1]), ddof=2)

# fluence and flux are bimodal  
# but are they lower limits? 
fluence_guess = [2.5, 2, 60, 30]
popt_fluence, pcov_fluence = curve_fit(gaussian2_cdf, x_mid_fluence, cdf_norm_fluence, p0=fluence_guess) 

# plot the fit results and the normalized histograms 
plot_data_and_fit('DM', x_mid_DM, hist_norm_DM, cdf_norm_DM, hist_err_norm_DM, c_hist_err_norm_DM, popt_DM)
plot_data_and_fit('width', x_mid_width, hist_norm_width, cdf_norm_width, hist_err_norm_width, c_hist_err_norm_width, popt_width)

