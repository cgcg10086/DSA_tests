#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:20:31 2019

@author: gechen

FRB_rate_single_v2.py: adds the effect of finite DM sampling. 
"""
def Compute_dispersion_smearing_finite(DM, DM_low, DM_up, n, frequency_central=1405, channel_width=0.122):
    '''
    Using discrete DM values, i.e. finite number of DM samples.
        DM: the true DM
        frequency_central: central frequency in MHz (1280-1530 MHz)
        channel_width: in MHz (bandwidth 250 MHz/ 2048 channels)
    Ideally: w_DM = 8.3 *1e6 * DM * channel_width * frequency_central ** (-3) 
    '''
    DM_sample = np.linspace(DM_low, DM_high, num=n) 
    w_DM = 8.3 *1e6 * DM * channel_width * frequency_central ** (-3) # Instrumental broadening (in ms)
 
    return w_DM  
