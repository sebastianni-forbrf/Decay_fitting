'''
Created by: Sebastian Nilsson
2020-11-06
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import cm
import scipy.special as sp
import scipy.optimize as optim
from scipy.interpolate import interp1d, LSQUnivariateSpline,UnivariateSpline
import scipy as sp
import lmfit as lm
import pandas as pd
import corner
import os
from tqdm import tqdm
import csv
from matplotlib import gridspec

mpl.rc('figure', max_open_warning = 0)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('axes', axisbelow=True)
plt.rcParams["savefig.dpi"] = 500

fit_dict = {
    ###############################################################################
    # Load data parameters
    ###############################################################################  
    'load_data_dir' : '/home/sebastian/OneDrive/Research/Decay_Signals/Calibration_data/Fitted_data/MFG_fit',
    'temp_data': True,
    ###############################################################################
    # Model fit parameters
    ############################################################################### 
    'log_data': True,
    'poly_order_1': 15,
    'poly_order_2': 15,
    ###############################################################################
    # Save Parameters
    ############################################################################### 
    'save_data_dir' : '/home/sebastian/OneDrive/Research/Decay_Signals/Calibration_data/Fitted_data/MFG_fit/Model',
    'img_format': 'jpg',
}

###############################################################################
# Loading Data
###############################################################################

if not os.path.exists(fit_dict['save_data_dir']):
    os.makedirs(fit_dict['save_data_dir'])

amp_data = pd.read_csv(fit_dict['load_data_dir'] + '/Amplitude_method_data.csv')
tau_data = pd.read_csv(fit_dict['load_data_dir'] + '/Tau_method_data.csv')


###############################################################################
# Displaying Data
###############################################################################

tau_plot_fig = plt.figure()
plt.grid(True,which='both')
plt.scatter(tau_data['Temperature'] + 273.15,tau_data['tau_final'],marker ='o',facecolors='none', edgecolors='k',label ='Tau Fit')
plt.scatter(amp_data['Temperature']+ 273.15,amp_data['tau_final'],marker ='v',facecolors='none', edgecolors='0.5',label = 'Amp Fit')
plt.yscale('log')
plt.ylim(min(tau_data['tau_final']) - 0.5*min(tau_data['tau_final']),max(tau_data['tau_final']) + 0.5*max(tau_data['tau_final']))
plt.title('Decay time [\u03C4]')
plt.xlabel('Temperature [K]')
plt.ylabel(' \u03C4 [s]')
plt.legend()

###############################################################################
# Doing modelfit
###############################################################################

if fit_dict['log_data'] == True:
    tau_decay_data = np.log10(tau_data['tau_final'].to_numpy())
    amp_decay_data = np.log10(amp_data['tau_final'].to_numpy())
    temp_data = tau_data['Temperature'].to_numpy()+273.15
elif fit_dict['log_data'] == False:
    tau_decay_data = tau_data['tau_final'].to_numpy()
    amp_decay_data = amp_data['tau_final'].to_numpy()
    temp_data = tau_data['Temperature'].to_numpy()+273.15


poly_fit_tau_temp = np.polyfit(temp_data,tau_decay_data,fit_dict['poly_order_1'],full = True,cov = True)
polyval_tau_temp = np.poly1d(poly_fit_tau_temp[0])

poly_fit_tau_decay = np.polyfit(polyval_tau_temp(temp_data),temp_data,fit_dict['poly_order_2'],full = True,cov = True)
polyval_tau_decay = np.poly1d(poly_fit_tau_decay[0])

poly_fit_amp_temp = np.polyfit(temp_data,amp_decay_data,fit_dict['poly_order_1'],full = True,cov = True)
polyval_amp_temp = np.poly1d(poly_fit_amp_temp[0])

poly_fit_amp_decay = np.polyfit(polyval_amp_temp(temp_data),temp_data,fit_dict['poly_order_2'],full = True,cov = True)
polyval_amp_decay = np.poly1d(poly_fit_amp_decay[0])

###############################################################################
# Displaying modelfit
###############################################################################

model_fig_temp = plt.figure()
plt.grid(True,which='both')
plt.scatter(temp_data,tau_decay_data,marker ='o',facecolor='0.75',s = 7,label = 'Temperature Data')
plt.plot(temp_data,polyval_tau_temp(temp_data),color = 'k',linestyle = '--',label = 'Fitted Temperature Model Tau')
plt.xlabel('Temperature [K]')
plt.ylabel(r'log($\tau$) [s]')
plt.title(r'Tau method data with fit $\tau$(T)')
plt.legend()
plt.savefig(fit_dict['save_data_dir'] + '/temperature_model_tau.' + fit_dict['img_format'],format = fit_dict['img_format'])

model_fig_decay = plt.figure()
plt.grid(True,which='both')
plt.scatter(tau_decay_data,temp_data,marker ='o',facecolor='0.75',s = 7,label = 'Temperature Data')
plt.plot(polyval_tau_temp(temp_data),polyval_tau_decay(polyval_tau_temp(temp_data)),color = 'k',linestyle = '--',label = 'Fitted Temperature Model Tau')
plt.ylabel('Temperature [K]')
plt.xlabel(r'log($\tau$) [s]')
plt.title(r'Tau method data with fit T($\tau$)')
plt.legend()
plt.savefig(fit_dict['save_data_dir'] + '/decay_model_tau.'+ fit_dict['img_format'],format = fit_dict['img_format'])


model_fig_temp = plt.figure()
plt.grid(True,which='both')
plt.scatter(temp_data,amp_decay_data,marker ='o',facecolor='0.75',s = 7,label = 'Temperature Data')
plt.plot(temp_data,polyval_amp_temp(temp_data),color = 'k',linestyle = '--',label = 'Fitted Temperature Model Amplitude')
plt.xlabel('Temperature [K]')
plt.ylabel(r'log($\tau$) [s]')
plt.title(r'Amp method data with fit $\tau$(T)')
plt.legend()
plt.savefig(fit_dict['save_data_dir'] + '/temperature_model_amp.' + fit_dict['img_format'],format = fit_dict['img_format'])

model_fig_decay = plt.figure()
plt.grid(True,which='both')
plt.scatter(amp_decay_data,temp_data,marker ='o',facecolor='0.75',s = 7,label = 'Temperature Data')
plt.plot(polyval_amp_temp(temp_data),polyval_amp_decay(polyval_amp_temp(temp_data)),color = 'k',linestyle = '--',label = 'Fitted Temperature Model Amplitude')
plt.ylabel('Temperature [K]')
plt.xlabel(r'log($\tau$) [s]')
plt.title(r'Amp method data with fit T($\tau$)')
plt.legend()
plt.savefig(fit_dict['save_data_dir'] + '/decay_model_amp.'+ fit_dict['img_format'],format = fit_dict['img_format'])

###############################################################################
# Saving fit data
###############################################################################

fit_data = pd.DataFrame()
fit_data['Temp_Coef_tau'] = poly_fit_tau_temp[0]
fit_data['Decay_Coef_tau'] = poly_fit_tau_decay[0]
fit_data['Temp_Coef_amp'] = poly_fit_amp_temp[0]
fit_data['Decay_Coef_amp'] = poly_fit_amp_decay[0]

fit_data.to_csv(fit_dict['save_data_dir'] + '/model_data.csv')

with open(fit_dict['save_data_dir'] + '/parameters.csv', 'w') as f:
    for key in fit_dict.keys():
        f.write("%s,%s\n"%(key,fit_dict[key]))

plt.show()