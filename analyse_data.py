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
from matplotlib import gridspec

fit_dict = {
    ###############################################################################
    # Load data parameters
    ###############################################################################  
    'load_model_data_dir' : 'c:\\Users\\Sebastian\\OneDrive - Lund University\\Research\\Decay_Signals\\Calibration_data\\Fitted_data\\MFG_fit\\Model\\model_data.csv',
    'load_data_dir' : 'c:\\Users\\Sebastian\\OneDrive - Lund University\\Research\\Decay_Signals\\Calibration_data\\Fitted_data\\MFG_fit',
    ###############################################################################
    # Model fit parameters
    ############################################################################### 
    'log_data': False,
    ###############################################################################
    # Save Parameters
    ############################################################################### 
    'save_data_dir' : 'c:\\Users\\Sebastian\\OneDrive - Lund University\\Research\\Decay_Signals\\Calibration_data\\Fitted_data\\MFG_fit\\Model',
    'img_format': 'jpg',
}


###############################################################################
# Loading data
############################################################################### 

exp_data = pd.read_csv(fit_dict['load_data_dir'] + '\\Tau_method_data.csv')
model_data = pd.read_csv(fit_dict['load_model_data_dir'])

decay_data = exp_data['tau_final'].to_numpy()

###############################################################################
# Fit to model
############################################################################### 

temp_model = np.poly1d(model_data['Decay_Coef_tau'].to_numpy())

if fit_dict['log_data'] == True:
    decay_data = np.log10(decay_data)
    fitted_data = temp_model(decay_data)
else:
    decay_data = decay_data
    fitted_data = temp_model(decay_data)

###############################################################################
# Plot fitted data
############################################################################### 

model_fig_temp = plt.figure()
plt.scatter(np.arange(len(fitted_data)),fitted_data,marker ='o',facecolor='0.75',s = 7)
# plt.plot(fitted_data)
plt.grid(True,which='both')
plt.ylabel('Temperature [K]')
plt.xlabel('Data point')
plt.title(r'Fitted data')
# plt.savefig(fit_dict['save_data_dir'] + '\\temperature_model_tau.' + fit_dict['img_format'],format = fit_dict['img_format'])

plt.show()