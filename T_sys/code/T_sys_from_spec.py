#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:36:16 2019
T_sys_from_sky.py 
@author: gechen
"""
import numpy as np 
import pandas
import matplotlib.pyplot as plt
import sys 


T_h = float(eval(sys.argv[1]))
T_c = float(eval(sys.argv[2]))
col_h = sys.argv[3]
col_c = sys.argv[4]
my_file=sys.argv[5]


'''
T_h = 300
T_c = 5
col_h = 'C'
col_c = 'A'
my_file='ac1bd2.csv'
'''

print '5 input arguments: Th, Tc, col_h, col_c, file.'
print "E.g.: python T_sys_from_spec.py 300 10 'C' 'A' 'ac1bd2.csv'  " 

colnames = ['frequency', 'A', 'B', 'C', 'D']
spec = pandas.read_csv(my_file, names=colnames, skiprows=range(0, 32)) # skip the same number of rows for every file? 
logy_h = spec[col_h]
logy_c = spec[col_c]
frequency = spec['frequency'] 

#print 'Is y axis = log10(power/dBm)?'
y_h = 10 ** (logy_h / 10)
y_c = 10 ** (logy_c / 10)


Y = y_h / y_c # the Y factor 
T_sys = (T_h - Y * T_c) / (Y - 1) 

fig1, ax1 = plt.subplots() 
fig1.set_size_inches(8., 6.)
ax1.set_yscale('log') 
ax1.plot(frequency, y_h, 'r', label='Hot (absorber)')
ax1.plot(frequency, y_c, 'b', label='Cold (sky)')
ax1.set_xlabel(r'Frequency', fontsize = 12) 
ax1.set_ylabel(r'Power [dBm]', fontsize = 12) 
ax1.set_title('Power vs. frequency for '+my_file+col_h+col_c, fontsize = 12) 
ax1.legend()
fig1.savefig('Power_spectrum_'+my_file+col_h+col_c+'.pdf')

fig2, ax2 = plt.subplots() 
fig2.set_size_inches(8., 6.)
ax2.plot(frequency, T_sys)
ax2.set_ylim(0, 100)
ax2.set_xlabel(r'Frequency', fontsize = 12) 
ax2.set_ylabel(r'T_sys', fontsize = 12) 
ax2.set_title('T_sys vs. frequency for '+my_file+col_h+col_c, fontsize = 12) 
fig2.savefig('T_sys_'+my_file+col_h+col_c+'.pdf') 

