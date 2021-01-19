'''
Created by: Sebastian Nilsson
2020-09-10
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
import parmap as pmap
import multiprocessing as mp 
import logging
import sys
import warnings
import gc
from memory_profiler import profile
warnings.filterwarnings("ignore", category=RuntimeWarning) 

my_path = os.path.abspath(__file__)

mpl.rc('figure', max_open_warning = 0)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
plt.rc('axes', axisbelow=True)
plt.rcParams["savefig.dpi"] = 500

def decay_sig(fit_dict,gauss = True,shot = True):
    '''
    Simulates a monoexponential decay signal given the prameters show in the fit_dict.
    gauss == True: Enables gaussion noise according to the parameters in fit_dict.
    shot == True: Enables shot noise according to the parameters in fit_dict.
    '''
    t_before = np.linspace(0,fit_dict['tmax_sim']/2,int(fit_dict['N_sim']/2))
    t = np.linspace(0,fit_dict['tmax_sim'],int(fit_dict['N_sim']))

    a = 1/fit_dict['alpha_sim']**2
   
    s = a*np.exp(-t/fit_dict['tau_sim']) + fit_dict['bias_sim']
   
    s_out = np.concatenate((np.zeros(t_before.shape) + fit_dict['bias_sim'],s)) + fit_dict['bias_sim']
    t_out = np.concatenate((t_before,np.linspace(fit_dict['tmax_sim']/2,fit_dict['tmax_sim'],int(fit_dict['N_sim']))))

    if shot == 'True' and gauss  == 'True':
        s_out = gaus_noise(s_out,fit_dict['beta_sim'],fit_dict['mu_sim']) + shot_noise(s_out)
    elif shot == 'True':
        s_out = shot_noise(s_out)
    elif gauss == 'True':
        s_out = s_out + gaus_noise(s_out,fit_dict['beta_sim'],fit_dict['mu_sim'])
    else:
        s_out = s_out
    return s_out, t_out, a

def shot_noise(data):
    ss = np.random.poisson(data,data.shape)
    return ss

def gaus_noise(data,beta,mu):
    ss = np.random.normal(loc = mu,scale = beta,size = data.shape)
    return ss

def sig_uncert(sig,noise_sig,I_0,fit_dict):
    sigN = np.sqrt(fit_dict['alpha_sim']*fit_dict['alpha_sim']*I_0*sig + fit_dict['beta_sim']**2)
    residual = sig - noise_sig
    return sigN, residual

def prep_data(time_data,decay_data,fit_dict,C1,C2,cut_data):
    '''
    Cutting the data starting at the max peak value and shifting the time axis to this point

    Example Usage:
    Cut out only the decay curve:
    time_data, decay_curve, max_value, max_value_index = prep_data(y_data,x_data,fit_dict,c1 = 0,c2 = 0,cut_data= False)

    Cut the already existing decay curve for amplitude fit using the C1 and C2:

    time_data,decay_curve,time_start_index,time_end_index = prep_data(time_data,decay_data,fit_dict,C_1,C_2,cut_data=True)
    '''
    if cut_data == True:

        if fit_dict['norm_data'] == 1:
            time_data = time_data/time_data[-1]
            decay_data = decay_data/decay_data[0]
        elif fit_dict['norm_data'] == 2:
            time_data = time_data/time_data[-1]
            decay_data = decay_data/np.max(decay_data)

        window_size = int(len(decay_data)/fit_dict['avg_window'])
        moving_averages = moving_average(decay_data,n = window_size)

        c1_cut_idx = int(find_nearest(moving_averages,C1)) + window_size
        c2_cut_idx = int(find_nearest(moving_averages,C2))

        decay_data_out = decay_data[c1_cut_idx:c2_cut_idx]
        time_data_out = time_data[c1_cut_idx:c2_cut_idx]

        return time_data_out,decay_data_out,c1_cut_idx,c2_cut_idx
    else:
        window_size = int(len(decay_data)/fit_dict['avg_window'])
        moving_averages = moving_average(decay_data,n = window_size)

        max_idx = int(find_nearest(moving_averages,np.max(moving_averages))) + window_size
        max_val = decay_data[max_idx]

        decay_data_out = decay_data[max_idx:]
        time_data_out = time_data[:len(decay_data_out)]

        return time_data_out,decay_data_out,max_val,max_idx

def f(t,a,b):
    return a * np.exp(-t/b)

def exp_func(params,x,data):
    model = params['amp'] * np.exp(-x/ params['tau']) + params['c']
    return (model - data)

def exp_func_MC(x,amp,tau,c):
    model = amp * np.exp(-x/ tau) + c
    return model

def find_nearest(array, value):
    '''
    Finds the index of a value in an array that are the closest to this value.
    input: array or list,value
    output: positon index in the array or list
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def amplitude_fit(params,x,y,fit_param,fit_dict,data_cont,logger):
    '''
    Fits decay data using the Amplitude method where the y_data is cut of at a certain percentage C1 and C2.
    The relevant data shown in the preallocation of arrays are then saved the pandas DataFrame that is sent into the function(data_cont).

    """"""""""""
    OBS: This function should only be used if you wanna do a grid search to find the best C1 and C2. If you wanna do fitting with fixed C1 and C2 use paralell_eval_amp function.
    """"""""""""

    Function parameters:
    params: Handles the variation of C1 and C2 for amplitude cutting the y_data, see fit_exp function for functionality
    x: and list of arrays that contain time data for the decay curve
    y: a list of arrays that contains decay data for the decay curve (The decay curve must be cut out using prep_data before sending to this function)
    fit_param: Handles the paramters for fitting to an exponential function, see fit_exp function for functionality
    fit_dict: Dictionary containing all the relevant fitting data.
    data_cont: Pandas DataFrame to store all the data from this function.

    Example Usage based on fit_exp:

    exp_fit_params = lm.Parameters() # Creating Paramters Class using the lm-fit package that handles the updating and of exponential decay curve paramters, see more at https://lmfit.github.io/lmfit-py/
    exp_fit_params.add('tau',value = fit_dict['tau'], min = fit_dict['tau_min'], max = fit_dict['tau_max'], vary = fit_dict['tau_vary']) # Adding parameters to the Parameters Class
    exp_fit_params.add('amp',value = fit_dict['amp'], min = fit_dict['amp_min'], max = fit_dict['amp_max'], vary = fit_dict['amp_vary']) # Adding parameters to the Parameters Class
    exp_fit_params.add('c',value = fit_dict['c'], min = fit_dict['c_min'], max = fit_dict['c_max'], vary = fit_dict['c_vary']) # Adding parameters to the Parameters Class

    amp_params = lm.Parameters() Creating another parameters Class that handles the variation of C1 and C2 such that at grid search can be performed to find best C1 and C2 given a loss function(see fit_dict)
    amp_params.add('grid_c1_amp',value = fit_dict['grid_c1_amp'], min = fit_dict['grid_c1_amp_min'], max = fit_dict['grid_c1_amp_max'],brute_step = fit_dict['grid_c1_amp_step']) # Adding parameters to the Parameters Class
    amp_params.add('grid_c2_amp',value = fit_dict['grid_c2_amp'], min = fit_dict['grid_c2_amp_min'], max = fit_dict['grid_c2_amp_max'],brute_step = fit_dict['grid_c2_amp_step']) # Adding parameters to the Parameters Class

    amp_fit_c1c2 = lm.Minimizer(amplitude_fit, amp_params, fcn_args = (time_data,decay_data,exp_fit_params,fit_dict,amp_fit_data,),iter_cb=progress_update) # Creating a Minimizer class, see lmfit above.
    amp_best_c1c2 = amp_fit_c1c2.minimize(method = 'brute',workers = fit_dict['num_cores']) # Solving the brute search method by calling the minimize functon from the Minimizer class. 

    returns: The loss metric of choice, see fit_dict for possible choices. Also saves the data to an existing pandas DataFrame.
    '''
    start_idx = np.zeros(len(x))
    end_idx = np.zeros(len(x))
    time_start = np.zeros(len(x))
    time_stop = np.zeros(len(x))
    intensity_start = np.zeros(len(x))
    intensity_end = np.zeros(len(x))
    amplitude_guess_norm = np.zeros(len(x))
    tau_guess_norm = np.zeros(len(x))
    c_guess_norm = np.zeros(len(x))
    amplitude_final_norm = np.zeros(len(x))
    amplitude_final_norm_std = np.zeros(len(x))
    tau_final_norm = np.zeros(len(x))
    tau_final_norm_std = np.zeros(len(x))
    c_final_norm = np.zeros(len(x))
    c_final_norm_std = np.zeros(len(x))
    tau_final = np.zeros(len(x))
    tau_final_std = np.zeros(len(x))
    guess_residual = np.zeros(len(x))
    guess_iterations = np.zeros(len(x))
    final_residual = np.zeros(len(x))
    final_iterations = np.zeros(len(x))

    for i in range(len(y)):
        if fit_dict['load_data'] == True:
            time_data = x[i]
            decay_data = y[i]
        else:
            time_data = x[i]
            decay_data = y[i]

        if fit_dict['grid_amp_method'] == True:
            if params['grid_c1_amp'].value < params['grid_c2_amp'].value:
                logger.info('Amp Fit: C2 greater than C1: skipping this sim.')
                return 5
            C_1 = params['grid_c1_amp']
            C_2 = params['grid_c2_amp']
            time_norm_fit,decay_norm_fit,T_start,T_end = prep_data(time_data,decay_data,fit_dict,C_1,C_2,cut_data=True)
        else:
            time_norm_fit,decay_norm_fit,T_start,T_end = prep_data(time_data,decay_data,fit_dict,fit_dict['c1_amp'],fit_dict['c2_amp'],cut_data=True)

        estTau = super_rough_guess(time_norm_fit,decay_norm_fit,fit_dict)

        fit_param['tau'].value = estTau
        fit_param['amp'].value = decay_norm_fit[0]

        guess_result = lm.minimize(exp_func,fit_param,args=(time_norm_fit,decay_norm_fit,),method= fit_dict['rough_guess_method'],nan_policy=fit_dict['rough_guess_nan_policy'],reduce_fcn= fit_dict['rough_guess_reduce_func'])

        if guess_result.success != True:
            logger.warning('Amp Fit: Guess method fit did not converge:\n %s,\n %s, \nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(guess_result),guess_result.params,str(i),params['grid_c1_amp'],params['grid_c2_amp'])

        fit_param['amp'].value = guess_result.params['amp'].value
        fit_param['tau'].value = guess_result.params['tau'].value

        result = lm.minimize(exp_func,fit_param,args=(time_norm_fit,decay_norm_fit,),method = fit_dict['final_guess_method'],nan_policy = fit_dict['final_guess_nan_policy'],reduce_fcn = fit_dict['final_guess_reduce_func'])

        if result.success != True:
            logger.warning('Amp Fit: Final method fit did not converge:\n %s,\n %s,  \nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(result),result.params,str(i),params['grid_c1_amp'],params['grid_c2_amp'])
        
        
        start_idx[i] = T_start
        end_idx[i] = T_end
        time_start[i] = time_data[T_start]
        time_stop[i] = time_data[T_end]
        intensity_start[i] = decay_data[T_start]
        intensity_end[i] = decay_data[T_end]
        amplitude_guess_norm[i] = guess_result.params['amp'].value
        tau_guess_norm[i] = guess_result.params['tau'].value
        c_guess_norm[i] = guess_result.params['c'].value
        amplitude_final_norm[i] = result.params['amp'].value
        try:
            amplitude_final_norm_std[i] = result.params['amp'].stderr
        except (NameError ,RuntimeError,TypeError):
            logger.warning('Amp Fit: Amplitude standard deviation could not be computed:\n %s,\n %s,  \nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(result),result.params,str(i),params['grid_c1_amp'],params['grid_c2_amp'])
            amplitude_final_norm_std = 0
        tau_final_norm[i] = result.params['tau'].value
        try:
            tau_final_norm_std[i] = result.params['tau'].stderr
            tau_final_std[i] = result.params['tau'].stderr*time_data[-1]
        except (NameError ,RuntimeError,TypeError):
            logger.warning('Amp Fit: Tau standard deviation could not be computed:\n %s,\n %s,  \nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(result),result.params,str(i),params['grid_c1_amp'],params['grid_c2_amp'])
            tau_final_norm_std[i] = 0
            tau_final_std[i] = 0
        c_final_norm[i] = result.params['c'].value

        try:
            c_final_norm_std = result.params['c'].stderr
        except (NameError ,RuntimeError,TypeError):
            logger.warning('Amp Fit: C standard deviation could not be computed:\n %s,\n %s,  \nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(result),result.params,str(i),params['grid_c1_amp'],params['grid_c2_amp'])
            c_final_norm_std[i] = 0
        tau_final[i] = result.params['tau'].value*time_data[-1]
        guess_residual[i] = guess_result.redchi
        guess_iterations[i] = guess_result.nfev
        final_residual[i] = result.redchi
        final_iterations[i] = result.nfev

    data_cont['start_idx'] = start_idx
    data_cont['end_idx'] = end_idx
    data_cont['time_start'] = time_start
    data_cont['time_stop'] = time_stop
    data_cont['intensity_start'] = intensity_start
    data_cont['intensity_end'] = intensity_end
    data_cont['amplitude_guess_norm'] = amplitude_guess_norm
    data_cont['tau_guess_norm'] = tau_guess_norm
    data_cont['c_guess_norm'] = c_guess_norm
    data_cont['amplitude_final_norm'] = amplitude_final_norm
    data_cont['amplitude_final_norm_std'] = amplitude_final_norm_std
    data_cont['tau_final_norm'] = tau_final_norm
    data_cont['tau_final_norm_std'] = tau_final_norm_std
    data_cont['c_final_norm'] = c_final_norm
    data_cont['c_final_norm_std'] = c_final_norm_std
    data_cont['tau_final'] = tau_final
    data_cont['tau_final_std'] = tau_final_std
    data_cont['guess_residual'] = guess_residual
    data_cont['guess_iterations'] = guess_iterations
    data_cont['final_residual'] = final_residual
    data_cont['final_iterations'] = final_iterations

    if fit_dict['loss_metric'] == 1:
        return np.mean(tau_final_norm_std)
    elif fit_dict['loss_metric'] == 2:
        return np.std(tau_final_norm_std)
    elif fit_dict['loss_metric'] == 3:
        return np.std(tau_final_norm_std)/np.mean(tau_final_norm_std)
    elif fit_dict['loss_metric'] == 4:
        return np.mean(tau_final_norm)
    elif fit_dict['loss_metric'] == 5:
        return np.std(tau_final_norm)
    else:
        return np.std(tau_final_norm)/np.mean(tau_final_norm)

def tau_fit(params,x,y,fit_param,fit_dict,data_cont,logger):
    '''
    Fits decay data using the Amplitude method where the y_data is cut of at a certain percentage C1 and C2.
    The relevant data shown in the preallocation of arrays are then saved the pandas DataFrame that is sent into the function(data_cont).

    """"""""""""
    OBS: This function should only be used if you wanna do a grid search to find the best C1 and C2. If you wanna do fitting with fixed C1 and C2 use paralell_eval_amp function.
    """"""""""""

    Function parameters:
    params: Handles the variation of C1 and C2 for amplitude cutting the y_data, see fit_exp function for functionality
    x: and list of arrays that contain time data for the decay curve
    y: a list of arrays that contains decay data for the decay curve (The decay curve must be cut out using prep_data before sending to this function)
    fit_param: Handles the paramters for fitting to an exponential function, see fit_exp function for functionality
    fit_dict: Dictionary containing all the relevant fitting data.
    data_cont: Pandas DataFrame to store all the data from this function.

    Example Usage based on fit_exp:

    exp_fit_params = lm.Parameters() # Creating Paramters Class using the lm-fit package that handles the updating and of exponential decay curve paramters, see more at https://lmfit.github.io/lmfit-py/
    exp_fit_params.add('tau',value = fit_dict['tau'], min = fit_dict['tau_min'], max = fit_dict['tau_max'], vary = fit_dict['tau_vary']) # Adding parameters to the Parameters Class
    exp_fit_params.add('amp',value = fit_dict['amp'], min = fit_dict['amp_min'], max = fit_dict['amp_max'], vary = fit_dict['amp_vary']) # Adding parameters to the Parameters Class
    exp_fit_params.add('c',value = fit_dict['c'], min = fit_dict['c_min'], max = fit_dict['c_max'], vary = fit_dict['c_vary']) # Adding parameters to the Parameters Class

    amp_params = lm.Parameters() Creating another parameters Class that handles the variation of C1 and C2 such that at grid search can be performed to find best C1 and C2 given a loss function(see fit_dict)
    amp_params.add('grid_c1_tau',value = fit_dict['grid_c1_tau'], min = fit_dict['grid_c1_tau_min'], max = fit_dict['grid_c1_tau_max'],brute_step = fit_dict['grid_c1_tau_step']) # Adding parameters to the Parameters Class
    amp_params.add('grid_c2_tau',value = fit_dict['grid_c2_tau'], min = fit_dict['grid_c2_tau_min'], max = fit_dict['grid_c2_tau_max'],brute_step = fit_dict['grid_c2_tau_step']) # Adding parameters to the Parameters Class

    amp_fit_c1c2 = lm.Minimizer(tau_fit, amp_params, fcn_args = (time_data,decay_data,exp_fit_params,fit_dict,tau_fit_data,),iter_cb=progress_update) # Creating a Minimizer class, see lmfit above.
    amp_best_c1c2 = amp_fit_c1c2.minimize(method = 'brute',workers = fit_dict['num_cores']) # Solving the brute search method by calling the minimize functon from the Minimizer class. 

    returns: The loss metric of choice, see fit_dict for possible choices. Also saves the data to an existing pandas DataFrame.
    '''
    start_idx = np.zeros(len(x))
    end_idx = np.zeros(len(x))
    time_start = np.zeros(len(x))
    time_stop = np.zeros(len(x))
    intensity_start = np.zeros(len(x))
    intensity_end = np.zeros(len(x))
    amplitude_guess_norm = np.zeros(len(x))
    tau_guess_norm = np.zeros(len(x))
    c_guess_norm = np.zeros(len(x))
    amplitude_final_norm = np.zeros(len(x))
    amplitude_final_norm_std = np.zeros(len(x))
    tau_final_norm = np.zeros(len(x))
    tau_final_norm_std = np.zeros(len(x))
    c_final_norm = np.zeros(len(x))
    c_final_norm_std = np.zeros(len(x))
    tau_final = np.zeros(len(x))
    tau_final_std = np.zeros(len(x))
    guess_residual = np.zeros(len(x))
    guess_iterations = np.zeros(len(x))
    final_residual = np.zeros(len(x))
    final_iterations = np.zeros(len(x))
    while_loops = np.zeros(len(x))

    for i in range(len(y)):
        if fit_dict['grid_tau_method'] == True and fit_dict['load_data'] == True or fit_dict['simulate_data'] == True :
            if params['grid_c1_tau'].value > params['grid_c2_tau'].value:
                logger.info('C1 greater than C2: skipping this sim.')
                return 5
            time_data = x[i]
            decay_data = y[i]
        else:
            time_data = x[i]
            decay_data = y[i]

        if fit_dict['norm_data'] == 1:
            time_data_cut_norm = time_data/time_data[-1]
            decay_data_cut_norm = decay_data/decay_data[0]
        elif fit_dict['norm_data'] == 2:
            time_data_cut_norm = time_data/time_data[-1]
            decay_data_cut_norm = decay_data/np.max(decay_data)

        giga_rough_guess = super_rough_guess(time_data_cut_norm,decay_data_cut_norm,fit_dict)
        new_tau = giga_rough_guess
        
        k = 0
        fit_param['amp'].value = decay_data_cut_norm[0]
        fit_param['tau'].value = new_tau
        
        guess_result = lm.minimize(exp_func,fit_param,args=(time_data_cut_norm,decay_data_cut_norm,),method= fit_dict['rough_guess_method'],nan_policy=fit_dict['rough_guess_nan_policy'],reduce_fcn= fit_dict['rough_guess_reduce_func'])

        if guess_result.success != True:
            logger.warning('Tau Fit: Guess method fit did not converge:\n %s,\n %s, \nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(guess_result),guess_result.params,str(i),params['grid_c1_tau'],params['grid_c2_tau'])
        
        fit_param['amp'].value = guess_result.params['amp'].value
        fit_param['tau'].value = guess_result.params['tau'].value
        fit_param['c'].value = guess_result.params['c'].value
        new_tau = guess_result.params['tau'].value

        old_tau = 100
        while np.abs(old_tau - new_tau) > fit_dict['tau_diff']:
            
            if fit_dict['grid_tau_method'] == True:
                T_start = find_nearest(time_data_cut_norm,params['grid_c1_tau']*new_tau)
                T_end = find_nearest(time_data_cut_norm,params['grid_c2_tau']*new_tau)
            else:
                T_start = find_nearest(time_data_cut_norm,fit_dict['c1_tau']*new_tau)
                T_end = find_nearest(time_data_cut_norm,fit_dict['c2_tau']*new_tau)

            old_tau = new_tau
            time_norm_fit = time_data_cut_norm[T_start:T_end]
            decay_norm_fit = decay_data_cut_norm[T_start:T_end]

            if (T_end-T_start) < fit_dict['short_fit_vec_lim']:
                # logger.warning('Short fit vector, doing amplitude fit: Length %s,\n %s,\n %s,\nWaveform',str(T_end-T_start),lm.fit_report(guess_result),guess_result.params,str(i))
                guess_result = lm.minimize(exp_func,fit_param,args=(time_data_cut_norm,decay_data_cut_norm,),method= fit_dict['rough_guess_method'],nan_policy=fit_dict['rough_guess_nan_policy'],reduce_fcn= fit_dict['rough_guess_reduce_func'])
                fit_param['amp'].value = guess_result.params['amp'].value
                fit_param['tau'].value = guess_result.params['tau'].value
                fit_param['c'].value = guess_result.params['c'].value
                new_tau = guess_result.params['tau'].value
                T_start = find_nearest(time_data_cut_norm,params['grid_c1_tau']*new_tau)
                T_end = find_nearest(time_data_cut_norm,params['grid_c2_tau']*new_tau)
                time_norm_fit = time_data_cut_norm[T_start:T_end]
                decay_norm_fit = decay_data_cut_norm[T_start:T_end]
                result = lm.minimize(exp_func,fit_param,args=(time_norm_fit,decay_norm_fit,),method = fit_dict['final_guess_method'],nan_policy = fit_dict['final_guess_nan_policy'],reduce_fcn = fit_dict['final_guess_reduce_func'])
                break

            result = lm.minimize(exp_func,fit_param,args=(time_norm_fit,decay_norm_fit,),method = fit_dict['final_guess_method'],nan_policy = fit_dict['final_guess_nan_policy'],reduce_fcn = fit_dict['final_guess_reduce_func'])
            
            new_tau = result.params['tau'].value
            k+= 1
            if k > fit_dict['while_loops']:
                break
        
        if result.success != True:
            logger.warning('Tau Fit: Final method fit did not converge:\n %s,\n %s,\nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(result),result.params,str(i),params['grid_c1_tau'],params['grid_c2_tau'])
        
        start_idx[i] = T_start
        end_idx[i] = T_end
        time_start[i] = time_data[T_start]
        time_stop[i] = time_data[T_end]
        intensity_start[i] = decay_data[T_start]
        intensity_end[i] = decay_data[T_end]
        amplitude_guess_norm[i] = guess_result.params['amp'].value
        tau_guess_norm[i] = guess_result.params['tau'].value
        c_guess_norm[i] = guess_result.params['c'].value
        amplitude_final_norm[i] = result.params['amp'].value
        try:
            amplitude_final_norm_std[i] = result.params['amp'].stderr
        except(NameError,RuntimeError,TypeError):
            logger.warning('Tau Fit: Amplitude standard deviation could not be computed:\n %s,\n %s,\nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(result),result.params,str(i),params['grid_c1_tau'],params['grid_c2_tau'])
            amplitude_final_norm_std = 0
        
        tau_final_norm[i] = result.params['tau'].value
        
        try:
            tau_final_norm_std[i] = result.params['tau'].stderr
            tau_final_std[i] = result.params['tau'].stderr*time_data[-1]
        except(NameError,RuntimeError,TypeError):
            logger.warning('Tau Fit: Tau standard deviation could not be computed:\n %s,\n %s,\nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(result),result.params,str(i),params['grid_c1_tau'],params['grid_c2_tau'])
            tau_final_norm_std[i] = 0
            tau_final_std[i] = 0
        c_final_norm[i] = result.params['c'].value

        try:
            c_final_norm_std = result.params['c'].stderr
        except(NameError ,RuntimeError,TypeError):
            logger.warning('Tau Fit: C standard deviation could not be computed:\n %s,\n %s,\nWaveform: %s, \nC1: %s C2: %s',lm.fit_report(result),result.params,str(i),params['grid_c1_tau'],params['grid_c2_tau'])
            c_final_norm_std[i] = 0
        tau_final[i] = result.params['tau'].value*time_data[-1]
        guess_residual[i] = guess_result.redchi
        guess_iterations[i] = guess_result.nfev
        final_residual[i] = result.redchi
        final_iterations[i] = result.nfev
        while_loops[i] = k

    data_cont['start_idx'] = start_idx
    data_cont['end_idx'] = end_idx
    data_cont['time_start'] = time_start
    data_cont['time_stop'] = time_stop
    data_cont['intensity_start'] = intensity_start
    data_cont['intensity_end'] = intensity_end
    data_cont['amplitude_guess_norm'] = amplitude_guess_norm
    data_cont['tau_guess_norm'] = tau_guess_norm
    data_cont['c_guess_norm'] = c_guess_norm
    data_cont['amplitude_final_norm'] = amplitude_final_norm
    data_cont['amplitude_final_norm_std'] = amplitude_final_norm_std
    data_cont['tau_final_norm'] = tau_final_norm
    data_cont['tau_final_norm_std'] = tau_final_norm_std
    data_cont['c_final_norm'] = c_final_norm
    data_cont['c_final_norm_std'] = c_final_norm_std
    data_cont['tau_final'] = tau_final
    data_cont['tau_final_std'] = tau_final_std
    data_cont['guess_residual'] = guess_residual
    data_cont['guess_iterations'] = guess_iterations
    data_cont['final_residual'] = final_residual
    data_cont['final_iterations'] = final_iterations
    data_cont['while_loops'] = while_loops

    if fit_dict['loss_metric'] == 1:
        return np.mean(tau_final_norm_std)
    elif fit_dict['loss_metric'] == 2:
        return np.std(tau_final_norm_std)
    elif fit_dict['loss_metric'] == 3:
        return np.std(tau_final_norm_std)/np.mean(tau_final_norm_std)
    elif fit_dict['loss_metric'] == 4:
        return np.mean(tau_final_norm)
    elif fit_dict['loss_metric'] == 5:
        return np.std(tau_final_norm)
    else:
        return np.std(tau_final_norm)/np.mean(tau_final_norm)

def amplitude_MCMC_sampler(time_data,decay_data,function,fit_dict):
    '''
    Affineinvariant ensemble sampler for Markov chain Monte Carlo
    emcee: The MCMC Hammer: https://arxiv.org/pdf/1202.3665.pdf 
    Taken from the EMCEE example library in https://lmfit.github.io/lmfit-py/  
    '''
    # Set up a function and create a Model 
    model = lm.Model(function)

    ###############################################################################
    # Extract relevant data 
    if fit_dict['norm_data'] == 1:
        anal_time_data_norm = time_data/time_data[-1]
        anal_decay_data_norm = decay_data/decay_data[0]
    elif fit_dict['norm_data'] == 2:
        anal_time_data_norm = time_data/time_data[-1]
        anal_decay_data_norm = decay_data/np.max(decay_data)
    
    anal_decay_data_norm = anal_decay_data_norm[int(amp_fit_data['start_idx'][anal_idx]):int(amp_fit_data['end_idx'][anal_idx])]
    anal_time_data_norm = anal_time_data_norm[int(amp_fit_data['start_idx'][anal_idx]):int(amp_fit_data['end_idx'][anal_idx])]

    ###############################################################################
    # Create model parameters and give them initial values
    p = model.make_params(amp = amp_fit_data['amplitude_final_norm'][anal_idx], tau=amp_fit_data['tau_final_norm'][anal_idx], c=amp_fit_data['c_final_norm'][anal_idx])

    ###############################################################################
    # Fit the model using a traditional minimizer, and show the output:
    result = model.fit(data=anal_decay_data_norm, params=p, x=anal_time_data_norm, method = fit_dict['final_guess_method'],nan_policy = fit_dict['final_guess_nan_policy'])
    # resi_plot = plt.figure()
    # ax_resi_plot = resi_plot.add_subplot(111)
    # lm.report_fit(result)
    fig = plt.figure()
    plt.title('Amplitude Method Fit Analysis')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
    ax0 = plt.subplot(gs[0])
    ax0.scatter(anal_time_data_norm,anal_decay_data_norm-result.best_fit,marker ='.', facecolors='none',c= '0.5',label = 'Residuals')
    plt.title('Amplitude Method Fit Analysis')
    plt.legend()
    ax1 = plt.subplot(gs[1])
    ax1.scatter(anal_time_data_norm, anal_decay_data_norm,marker = 'o',c = '0.75',label = 'Data')
    dely = result.eval_uncertainty(sigma=5)
    ax1.plot(anal_time_data_norm,result.best_fit,c = 'k',linestyle = '-',label = 'Best Fit')
    ax1.fill_between(anal_time_data_norm, result.best_fit-dely,
                 result.best_fit+dely, color='0.5',label='5-$\sigma$ uncertainty band')
    plt.grid(True,which='both')
    plt.legend()
    plt.xlabel('Normalised Time [a.u]')
    plt.ylabel(' Normalised Intensity [a.u]')
    
    plt.savefig(fit_dict['save_data_dir'] + '/amplitude_residual_plot_with_final_guess.'+ fit_dict['img_format'],format = fit_dict['img_format'])
    ###############################################################################
    # Calculate parameter covariance using emcee:
    #  - start the walkers out at the best-fit values
    #  - set is_weighted to False to estimate the noise weights
    #  - set some sensible priors on the uncertainty to keep the MCMC in check
    #
    emcee_kws = dict(steps=fit_dict['MC_steps'], burn=300, thin=20, is_weighted=False,
                    progress=True)
    emcee_params = result.params.copy()
    emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))

    ###############################################################################
    # run the MCMC algorithm and show the results:
    result_emcee = model.fit(data=anal_decay_data_norm, x=anal_time_data_norm, params=emcee_params, method='emcee',
                            nan_policy='omit', fit_kws=emcee_kws)

    lm.report_fit(result_emcee)
    emcee_result = plt.figure()
    plt.scatter(anal_time_data_norm,anal_decay_data_norm,marker ='.', facecolors='none',c= '0.5',label = 'Data')
    plt.plot(anal_time_data_norm, model.eval(params=result.params, x=anal_time_data_norm),linestyle ='--', c = 'k', label='Solver')
    plt.plot(anal_time_data_norm,result_emcee.best_fit,linestyle ='-', color = '0.25', label='MC Best-Fit')
    plt.grid(True,which='both')
    plt.legend()
    plt.xlabel('Normalised Time [a.u]')
    plt.ylabel(' Normalised Intensity [a.u]')
    plt.title('Amplitude Monte Carlo Best Fit')
    plt.savefig(fit_dict['save_data_dir'] + '/amplitude_MCMC_residual_with_fit.'+ fit_dict['img_format'],format = fit_dict['img_format'])


    ###############################################################################
    # check the acceptance fraction to see whether emcee performed well
    walkerplot = plt.figure()
    plt.scatter(np.arange(len(result_emcee.acceptance_fraction)),result_emcee.acceptance_fraction,marker ='.',c = 'k')
    plt.xlabel('Walker')
    plt.ylabel('Acceptance fraction')
    plt.grid(True,which='both')
    plt.savefig(fit_dict['save_data_dir'] + '/amplitude_walker_plot.'+ fit_dict['img_format'],format = fit_dict['img_format'])


    ###############################################################################
    # try to compute the autocorrelation time
    # if hasattr(result_emcee, "acor"):
    #     print("Autocorrelation time for the parameters:")
    #     print("----------------------------------------")
    #     for i, p in enumerate(result.params):
    #         print(p, result.acor[i])


    ###############################################################################
    # Plot the parameter covariances returned by emcee using corner
    # corner_fig = plt.figure()
    emcee_corner = corner.corner(result_emcee.flatchain, labels=[r"$amp$", r"$\tau$",r"$c$", r"$\log \sigma$",r'$\Gamma \, [\mathrm{parsec}]$'],
                                truths=list(result_emcee.params.valuesdict().values()),quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 12},show_titles=True,
                                truth_color = '0.25',)
    plt.savefig(fit_dict['save_data_dir'] + '/amplitude_corner_plot.'+ fit_dict['img_format'],format = fit_dict['img_format'])
    ###############################################################################
    #
    print("\nMedian of posterior probability distribution")
    print('--------------------------------------------')
    lm.report_fit(result_emcee.params)

    # find the maximum likelihood solution
    highest_prob = np.argmax(result_emcee.lnprob)
    hp_loc = np.unravel_index(highest_prob, result_emcee.lnprob.shape)
    mle_soln = result_emcee.chain[hp_loc]
    print("\nMaximum likelihood Estimation")
    print('-----------------------------')
    for ix, param in enumerate(emcee_params):
        print(param + ': ' + str(mle_soln[ix]))

    quantiles = np.percentile(result_emcee.flatchain['tau'], [2.28, 15.9, 50, 84.2, 97.7])
    print("\nTau spread:")
    print('--------------------------------------------')
    print("1 sigma spread", 0.5 * (quantiles[3] - quantiles[1]))
    print("2 sigma spread", 0.5 * (quantiles[4] - quantiles[0]))

def tau_MCMC_sampler(time_data,decay_data,function,fit_dict):
    '''
    Affineinvariant ensemble sampler for Markov chain Monte Carlo
    emcee: The MCMC Hammer: https://arxiv.org/pdf/1202.3665.pdf 
    Taken from the EMCEE example library in https://lmfit.github.io/lmfit-py/  
    '''
    # Set up a function and create a Model 
    model = lm.Model(function)

    ###############################################################################
    # Extract relevant data 

    if fit_dict['norm_data'] == 1:
        anal_time_data_norm = time_data/time_data[-1]
        anal_decay_data_norm = decay_data/decay_data[0]
    elif fit_dict['norm_data'] == 2:
        anal_time_data_norm = time_data/time_data[-1]
        anal_decay_data_norm = decay_data/np.max(decay_data)
    
    anal_decay_data_norm = anal_decay_data_norm[int(tau_fit_data['start_idx'][anal_idx]):int(tau_fit_data['end_idx'][anal_idx])]
    anal_time_data_norm = anal_time_data_norm[int(tau_fit_data['start_idx'][anal_idx]):int(tau_fit_data['end_idx'][anal_idx])]

    ###############################################################################
    # Create model parameters and give them initial values
    p = model.make_params(amp = tau_fit_data['amplitude_final_norm'][anal_idx], tau=tau_fit_data['tau_final_norm'][anal_idx], c=tau_fit_data['c_final_norm'][anal_idx])

    ###############################################################################
    # Fit the model using a traditional minimizer, and show the output:
    result = model.fit(data=anal_decay_data_norm, params=p, x=anal_time_data_norm, method = fit_dict['final_guess_method'],nan_policy = fit_dict['final_guess_nan_policy'])
    # resi_plot = plt.figure()
    # ax_resi_plot = resi_plot.add_subplot(111)
    # lm.report_fit(result)
    fig = plt.figure()
    plt.title('Tau Method Fit Analysis')
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3]) 
    ax0 = plt.subplot(gs[0])
    ax0.scatter(anal_time_data_norm,anal_decay_data_norm-result.best_fit,marker ='.', facecolors='none',c= '0.5',label = 'Residuals')
    plt.title('Tau Method Fit Analysis')
    plt.legend()
    ax1 = plt.subplot(gs[1])
    ax1.scatter(anal_time_data_norm, anal_decay_data_norm,marker = 'o',color = '0.75',label = 'Data')
    dely = result.eval_uncertainty(sigma=5)
    ax1.plot(anal_time_data_norm,result.best_fit,c = 'k',linestyle = '-',label = 'Best Fit')
    ax1.fill_between(anal_time_data_norm, result.best_fit-dely,
                 result.best_fit+dely, color='0.5',label='5-$\sigma$ uncertainty band')
    plt.grid(True,which='both')
    plt.legend()
    plt.xlabel('Normalised Time [a.u]')
    plt.ylabel(' Normalised Intensity [a.u]')
    
    plt.savefig(fit_dict['save_data_dir'] + '/tau_residual_plot_with_final_guess.'+ fit_dict['img_format'],format = fit_dict['img_format'])
    ###############################################################################
    # Calculate parameter covariance using emcee:
    #  - start the walkers out at the best-fit values
    #  - set is_weighted to False to estimate the noise weights
    #  - set some sensible priors on the uncertainty to keep the MCMC in check
    #
    emcee_kws = dict(steps=fit_dict['MC_steps'], burn=300, thin=20, is_weighted=False,
                    progress=True)
    emcee_params = result.params.copy()
    emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))

    ###############################################################################
    # run the MCMC algorithm and show the results:
    result_emcee = model.fit(data=anal_decay_data_norm, x=anal_time_data_norm, params=emcee_params, method='emcee',
                            nan_policy='omit', fit_kws=emcee_kws)

    lm.report_fit(result_emcee)
    emcee_result = plt.figure()
    plt.scatter(anal_time_data_norm,anal_decay_data_norm,marker ='.', facecolors='none',c= '0.5',label = 'Data')
    plt.plot(anal_time_data_norm, model.eval(params=result.params, x=anal_time_data_norm),linestyle ='--', c = 'k', label='Solver')
    plt.plot(anal_time_data_norm,result_emcee.best_fit,linestyle ='-', color = '0.25', label='MC Best-Fit')
    plt.grid(True,which='both')
    plt.legend()
    plt.xlabel('Normalised Time [a.u]')
    plt.ylabel(' Normalised Intensity [a.u]')
    plt.title('Monte Carlo Best Fit')
    plt.savefig(fit_dict['save_data_dir'] + '/tau_MCMC_residual_with_fit.'+ fit_dict['img_format'],format = fit_dict['img_format'])


    ###############################################################################
    # check the acceptance fraction to see whether emcee performed well
    walkerplot = plt.figure()
    plt.scatter(np.arange(len(result_emcee.acceptance_fraction)),result_emcee.acceptance_fraction,marker ='.',c = 'k')
    plt.xlabel('walker')
    plt.ylabel('acceptance fraction')
    plt.grid(True,which='both')
    plt.savefig(fit_dict['save_data_dir'] + '/tau_walker_plot.'+ fit_dict['img_format'],format = fit_dict['img_format'])


    ###############################################################################
    # try to compute the autocorrelation time
    # if hasattr(result_emcee, "acor"):
    #     print("Autocorrelation time for the parameters:")
    #     print("----------------------------------------")
    #     for i, p in enumerate(result.params):
    #         print(p, result.acor[i])


    ###############################################################################
    # Plot the parameter covariances returned by emcee using corner
    # corner_fig = plt.figure()
    emcee_corner = corner.corner(result_emcee.flatchain, labels=[r"$amp$", r"$\tau$",r"$c$", r"$\log \sigma$",r'$\Gamma \, [\mathrm{parsec}]$'],
                                truths=list(result_emcee.params.valuesdict().values()),quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 12},show_titles=True,
                                truth_color = '0.25')
    plt.savefig(fit_dict['save_data_dir'] + '/tau_corner_plot.'+ fit_dict['img_format'],format = fit_dict['img_format'])
    ###############################################################################
    #
    print("\nMedian of posterior probability distribution")
    print('--------------------------------------------')
    lm.report_fit(result_emcee.params)

    # find the maximum likelihood solution
    highest_prob = np.argmax(result_emcee.lnprob)
    hp_loc = np.unravel_index(highest_prob, result_emcee.lnprob.shape)
    mle_soln = result_emcee.chain[hp_loc]
    print("\nMaximum likelihood Estimation")
    print('-----------------------------')
    for ix, param in enumerate(emcee_params):
        print(param + ': ' + str(mle_soln[ix]))

    quantiles = np.percentile(result_emcee.flatchain['tau'], [2.28, 15.9, 50, 84.2, 97.7])
    print("\nTau spread:")
    print('--------------------------------------------')
    print("1 sigma spread", 0.5 * (quantiles[3] - quantiles[1]))
    print("2 sigma spread", 0.5 * (quantiles[4] - quantiles[0]))

def super_rough_guess(x,y,fit_dict):
    '''
    Initially was very complex but it turned out using a simple guess is more robust, see fit_dict.
    '''
    # estTau = (x[0]-x[-1])/np.log(np.abs(y[-1])/y[0])
    estTau = fit_dict['c0']
    return estTau

def progress_update(params, iter, resid, *args, **kws):
    '''
    Post status updates every time the brute algorithm is done with one C1 C2 combination.
    '''
    print('Calculating...')

    # print(resid)

def moving_average(a, n=3) :
    '''
    Moving average filter.
    a: an array
    n: window length
    returns: moving averaged array
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def fit_data(fit_dict): 
    '''
    Main functon the does all the fitting based on a dictionary, see below.
        fit_dict = {
            ###############################################################################
            # Simulation paramters and load data paramters
            ###############################################################################  
            'simulate_data' : False, # Simulate some test data
            'N_sim': 2000,          # Number of data points in simulated decay
            'N_sig' : 200,          # Number of signals to simulate        
            'I_0' : 25,             # For uncertainty anlysis
            'amp_sim': 10,          # Amplitude in simulated decay
            'bias_sim': 2,        # Bias in simulated decay
            'tmax_sim' : 10,         # Arbitrairy sim time in simulated decay
            'photon_noise': 'True',   # If to include photon noise
            'alpha_sim':0.2,        # Photon noise in simulated decay, 0 <alpha <= 1, close to zero strong signal, close to one low signal
            'gaussian_noise': 'True', # If to include gaussian noise
            'beta_sim': 0.5,          # Gaussian noise standard deviation in simulated decay
            'mu_sim': 2,            # Gaussian bias in simulated decay
            'tau_sim': 1,            # Arbitrary tau in simulated decay
            ###############################################################################
            # Load data parameters
            ###############################################################################  
            'load_data' : True,    # Load some data 
            'hdf': True,
            'min_length': 250,      # Minimum data length to analyse
            'interpolate': False,    # Shorten or lengthen the data
            'inter_points': 1000,    # Number of interpolated data points
            'temp_data' : False,     # Temperature data
            'wind_data': False,      # Osciliscope window data
            'check_data': True,     # Simple data check
            'data_length_cut':200,  # Remove data sets shorter then this number
            'background_data': False,   #Background data
            'background_data_dir': '/home/sebastian/OneDrive/Research/Decay_Signals/Calibration_data/Data/LED_Study/signal_of_10_mV/Background_data',
            'load_data_dir' : '/home/sebastian/OneDrive/Research/Decay_Signals/Calibration_data/Data/LED_Study',
            ###############################################################################
            # GRID Window fitting paramters, both for tau method and amplitude method GRID
            ###############################################################################  
            'grid_amp_method': False,     # True: Use amplitude cutting
            'grid_tau_method': False,     # True: Use tau fitting
            'c0': 1/10,                 # Guess tau
            'grid_c1_amp': 0.75,             # First amplitude cut baseline
            'grid_c1_amp_max': 0.9,         # First amplitude cut max val
            'grid_c1_amp_min': 0.65,         # First amplitude cut min val
            'grid_c1_amp_step': 4,        # First amplitude cut step for brute algorithm
            'grid_c2_amp': 0.20,              # Last amplitude cut baseline
            'grid_c2_amp_max': 0.3,         # Last amplitude cut max val
            'grid_c2_amp_min': 0.15,         # Last amplitude cut min val
            'grid_c2_amp_step': 4,        # Last amplitude cut step for brute alogrithm 
            'avg_window': 40 ,       # Moving average window: len(decay_data)/'avg_window'  
            'tau_diff' : 1e-3,      #difference in tau to judge it to be stable.
            'short_fit_vec_lim': 30, # If tau fit vector is to short, it will use amplitude fitting
            'while_loops': 20,       # How many while loops in tau method before terminate if no stable tau can be found.
            'grid_c1_tau': 0.6,         # First tau cut baseline
            'grid_c1_tau_max': 0.2,     # First tau cut max val
            'grid_c1_tau_min': 1.2,     # First tau cut min val
            'grid_c1_tau_step': 4,    # First tau cut step for brute algorithm
            'grid_c2_tau': 2.5,          # Last tau cut baseline
            'grid_c2_tau_max': 0.7,     # Last tau cut max val
            'grid_c2_tau_min': 3,     # Last tau cut min val
            'grid_c2_tau_step': 4,    # Last tau cut step for brute alogrithm
            ###############################################################################
            # Fixed window fitting paramters, both for tau method and amplitude method
            ###############################################################################    
            'amp_method': True,     # True: Use amplitude method
            'tau_method': True,     # True: Use tau fitting
            'c1_amp': 0.75,         # Amplitude cut first cut
            'c2_amp': 0.05,         # Amplitude cut last cut
            'c1_tau': 0.8,          # Tau first cut
            'c2_tau': 3.5,            # Tau last cut
            ###############################################################################
            # Fitting paramters for tau and amplitude
            ###############################################################################     
            'tau': 0.5,         # Tau baseline
            'tau_max': 2,       # Tau max value
            'tau_min': 0.001,    # Tau min value 
            'tau_vary': True,   # True: allows tau to vary, False: Fix tau
            'amp': 0.7,     # Amplitude baseline
            'amp_max': 5,   # Amplitude max allowed value
            'amp_min': 0.01,   # Amplitude min allowed value
            'amp_vary': True,   # True: allows amplitude to vary, False: Fix amplitude
            'c': 0.0,           # bias level on the decay curve
            'c_max':1,        # maximim value on bias level
            'c_min': -1,      # maximim value on bias level
            'c_vary': True,     # True allows C to vary. False: Fix the amplitude
            'loss_metric': 1, # Minimises; 1: mean of tau,2: Std of tau, 3: Coeficcent of variation of tau, 4: mean of residual, 5: std of residual, 6: Coeficcent of variation of residual
            'norm_data' : 2, # 1:First value in decay data, 2: Max value in decay data
            ###############################################################################
            # Solver configurations
            ###############################################################################     
            'rough_guess_method': 'nelder',     #Robust solver to get initial guess.
            'rough_guess_nan_policy': 'omit',   # Options: 1:’raise’ : a ValueError is raised 2:’propagate’ : the values returned from userfcn are un-altered 3:’omit’ : non-finite values are filtered
            'rough_guess_reduce_func': None,    # Function to convert a residual array: None = sum of squares of residual, ’negentropy’ : neg entropy, using normal distribution, ’neglogcauchy’: neg log likelihood, using Cauchy distribution
            'final_guess_method': 'least_squares',    # Options: leastsq. ’leastsq’: Levenberg-Marquardt, ’least_squares’: Trust Region Reflective
            'final_guess_nan_policy': 'omit',   # Same as rough guess
            'final_guess_reduce_func': None,    # Same as rough guess
            'num_cores': -1, # Options: -1 will use all availble cores: 1 will use one core, 2 will use 2 etc.. (if more then one core is used data from grid_c1_amp grid_c2_amp fitting will not be availble, only the best paramters are saved)
            ###############################################################################
            # Post process configuration
            ###############################################################################  
            'save_data_dir':'/home/sebastian/OneDrive/Research/Decay_Signals/Calibration_data/Fitted_data/LED_study_fitted_data', # Data DIR
            'tau_data': '/Tau_method_data.csv', # Save the best tau data in a CSV file
            'amp_data': '/Amplitude_method_data.csv', # Save the best amp data in a CSV file
            'post_fit_anal': False, # True: reads a .csv file generated from the script to display the data obtained from previous fitting.
            'monte_carlo': False, # Calculating the posterior probability distribution of parameters, using mote carlo methods
            'plot_grid_data': True, # Plot grid_c1_amp grid_c2_amp grid output, with best fit for normalised taus normalised taus
            'diagnostic_on': True, # Generates plots of the data obtained from fitting
            'analyse_random': True, # True: will select random waveform to perform analysis on. N: will perform analysis on N:th waveform     
            'MC_steps': 500, # Monte Carlo steps in the monte_carlo option
            'num_data_fig': 40, #Amount of data figures to create
            'ax_comp_log': True, # Log data axises for the waveform data figures
            'interactive_plots': False, # Generates an interactive plot that allows you to pan through data from fitting
            'current_pos': 0, # Position to start the panning
            'img_format': 'jpg' # Image format to save the figures in.
        }
    '''
    
    global curr_pos
    if not os.path.exists(fit_dict['save_data_dir']):
        os.makedirs(fit_dict['save_data_dir'])

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%Y/%m/%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    output_file_handler = logging.FileHandler(fit_dict['save_data_dir']+'/log.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    output_file_handler.setFormatter(formatter)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
 
    logger.info('Saving data to: %s',fit_dict['save_data_dir'])

    ###############################################################################
    # Simulating and displaying test data
    ###############################################################################  

    if fit_dict['simulate_data'] == True:
        logger.info('Simulate data: True, simulating data...')
        decay_data_temp = np.zeros((fit_dict['N_sig'],fit_dict['N_sim'] + int(fit_dict['N_sim']/2)))
        time_data_temp = np.zeros((fit_dict['N_sig'],fit_dict['N_sim'] + int(fit_dict['N_sim']/2)))
        for i in range(fit_dict['N_sig']):
            yo,tp,std_a = decay_sig(fit_dict,fit_dict['gaussian_noise'],fit_dict['photon_noise'])
            decay_data_temp[i,:] = yo
            time_data_temp[i,:] = tp
        org_sig, t_org_sig, I_0_alpha = decay_sig(fit_dict,gauss = False, shot = False)
        sigmaN, resi = sig_uncert(org_sig,decay_data_temp[0,:],I_0_alpha,fit_dict)


        #Display a decay curve from simulated data
        figDecay = plt.figure()
        axDecay = figDecay.add_subplot(111)
        axDecay.plot(time_data_temp[0], decay_data_temp[0], color='k',label = '$r_n$')
        axDecay.plot(t_org_sig,org_sig,color='0.50', label = '$s_n$')
        figDecay.text(0.65, 0.78,'\u03C4 = ' + str(fit_dict['tau_sim']) + '\na = ' + str(np.round(I_0_alpha)) +
                '\n\u03B1 = '+ str(fit_dict['alpha_sim']) +'\n\u03B2 = '+ str(fit_dict['beta_sim']),ha = 'center',va = 'center', size=14)
        axDecay.legend()
        axDecay.set_xlabel('t [a.u]')
        axDecay.set_ylabel('Intensity [a.u]')
        plt.title('Noisy and original decay curve')
        plt.savefig(fit_dict['save_data_dir'] + '/sim_data.'+ fit_dict['img_format'],format = fit_dict['img_format'])
        
        # Displaying signal uncertainties
        # Plotting standard deviation of the curve
        
        figStd = plt.figure()
        axStd = figStd.add_subplot(111)
        axStd.plot(t_org_sig,resi,color='k', label = '$s_n - r_n$')
        axStd.plot(t_org_sig,sigmaN,color='0.5',label = '$\sigma_n$')
        axStd.set_xlabel('t [a.u]')
        axStd.set_ylabel('s_n - r_n [a.u]')
        axStd.legend()
        plt.title('Residual with $\sigma_n$')
        plt.savefig(fit_dict['save_data_dir'] + '/sigma_sim_data.'+ fit_dict['img_format'],format = fit_dict['img_format'])
        logger.info('Done!')

        logger.info('Cutting out data...')
        time_data = []
        decay_data = []
        temp_max_val_arr = []
        temp_max_idx_arr = []
        for i in tqdm(range(len(decay_data_temp))):
            try:
                temp_time_data, temp_decay_data, temp_max_val, temp_max_idx = prep_data(time_data_temp[i],decay_data_temp[i],fit_dict,C1 = 0,C2 = 0,cut_data= False)
                if len(temp_decay_data) < 1:
                    time_data.append(temp_time_data)
                    decay_data.append(temp_decay_data)
                    temp_max_val_arr.append(temp_max_val)
                    temp_max_idx_arr.append(temp_max_idx)
            except IndexError:
                logger.warning('Invalid data, check for validity:' + fit_dict['load_data_dir'])
                logger.removeHandler(stdout_handler)
                logger.removeHandler(output_file_handler)   
                return

    ###############################################################################
    #Creating pandas dataframe to store data
    ###############################################################################  

    tau_fit_data = pd.DataFrame()
    amp_fit_data = pd.DataFrame()

    ###############################################################################
    #Load some data
    ###############################################################################   

    if fit_dict['load_data'] == True and fit_dict['simulate_data'] != True:
        logger.info('Load data: True, lodaing some data...')
        if fit_dict['hdf'] == True:
            csv_decay_data = pd.read_hdf(fit_dict['load_data_dir'] + '/intensity.h5',key = 'intensity')
            csv_dt_data = pd.read_hdf(fit_dict['load_data_dir'] + '/dt.h5',key = 'deltat')
        else:  
            csv_decay_data = pd.read_csv(fit_dict['load_data_dir'] + '/intensity.csv')
            csv_dt_data = pd.read_csv(fit_dict['load_data_dir'] + '/dt.csv')
        try:
            csv_decay_data = csv_decay_data.drop(['Unnamed: 0'],axis = 1)
        except KeyError:
            print('No Unnamed colums')
        temp_decay_data_arr = []
        temp_time_data_arr = []
        rows, col = csv_decay_data.shape
        for i in tqdm(range(col-1)):
            if fit_dict['hdf'] == True:
                temp_decay_data_nan = csv_decay_data[i].to_numpy()
            else:
                temp_decay_data_nan = csv_decay_data[str(i)].to_numpy()
            nan_array = np.isnan(temp_decay_data_nan)
            not_nan_array = ~ nan_array
            temp_decay_data = temp_decay_data_nan[not_nan_array]
            if len(temp_decay_data) < fit_dict['min_length']:
                break
            temp_decay_data_arr.append(temp_decay_data)
            temp_time_data_arr.append(np.arange(len(temp_decay_data))*csv_dt_data['dt'][i])
        del csv_decay_data
        del csv_dt_data
        #Cut out the intresting part
        if fit_dict['background_data'] == True:
            background_data = 0
            if fit_dict['hdf'] == True:
                csv_background_data = pd.read_hdf(fit_dict['background_data_dir'] + '/intensity.h5')
            else:
                csv_background_data = pd.read_csv(fit_dict['background_data_dir'] + '/intensity.csv')
            try:
                csv_background_data = csv_background_data.drop(['Unnamed: 0'],axis = 1)
            except KeyError:
                print('No Unnamed colums')
            rows, col = csv_background_data.shape
            for i in range(col-1):
                background_data += csv_background_data[str(i)].to_numpy()
            background_data = background_data/(col-1)
            for i in range(col-1):
                temp_decay_data_arr[i] =  temp_decay_data_arr[i] - background_data
            del csv_background_data
        #Cut out the intresting part
        logger.info('Cutting out data...')
        time_data = []
        decay_data = []
        temp_max_val_arr = []
        temp_max_idx_arr = []
        for i in tqdm(range(len(temp_decay_data_arr))):
            if fit_dict['check_data'] == True:
                if i < fit_dict['data_length_cut'] and len(temp_decay_data_arr) >50:
                    continue
                if temp_decay_data_arr[i].min() < 0 :
                    temp_decay_data_arr[i] = temp_decay_data_arr[i] + abs(temp_decay_data_arr[i].min())

            temp_time_data, temp_decay_data, temp_max_val, temp_max_idx = prep_data(temp_time_data_arr[i],temp_decay_data_arr[i],fit_dict,C1 = 0,C2 = 0,cut_data= False)
            time_data.append(temp_time_data)
            decay_data.append(temp_decay_data)
            temp_max_val_arr.append(temp_max_val)
            temp_max_idx_arr.append(temp_max_idx)

        if fit_dict['interpolate'] == True:
            logger.info('Interpolate: True, Interpolating...')
            inter_decay_time_arr = []
            inter_decay_data_arr = []
            for i in tqdm(range(len(decay_data))):
                temp_inter_time_data = time_data[i]
                inter_time_data = np.linspace(temp_inter_time_data[0],temp_inter_time_data[-1],num = fit_dict['inter_points'],endpoint= True)
                s = UnivariateSpline(time_data[i], time_data[i], s=0 ,ext = 3)
                t = s.get_knots()
                f_inter = LSQUnivariateSpline(time_data[i],decay_data[i],t = t[1:-1],ext = 3)
                inter_decay_time_arr.append(inter_time_data)
                inter_decay_data_arr.append(f_inter(inter_time_data))

            decay_data = inter_decay_data_arr
            time_data = inter_decay_time_arr

        if fit_dict['temp_data'] == True:
            if fit_dict['hdf'] == True:
                csv_temp_data = pd.read_hdf(fit_dict['load_data_dir'] + '/temperature_data.h5')
            else:
                csv_temp_data = pd.read_csv(fit_dict['load_data_dir'] + '/temperature_data.csv')
            temp_true = csv_temp_data.iloc[0:int(len(decay_data)),2:5].sum(axis = 1)
            temp_true = temp_true.to_numpy()/3

            amp_fit_data['Temperature'] = temp_true
            tau_fit_data['Temperature'] = temp_true   

            del csv_temp_data

        if fit_dict['wind_data'] == True:
            if fit_dict['hdf'] == True:
                csv_window_data = pd.read_hdf(fit_dict['load_data_dir'] + '/window_settings.h5')
            else:
                csv_window_data = pd.read_csv(fit_dict['load_data_dir'] + '/window_settings.csv')
            tau_fit_data = pd.concat([tau_fit_data,csv_window_data.iloc[0:int(len(decay_data)),:]],axis = 1)
            amp_fit_data = pd.concat([amp_fit_data,csv_window_data.iloc[0:int(len(decay_data)),:]],axis = 1)
            tau_fit_data = pd.concat([tau_fit_data,csv_dt_data.iloc[0:int(len(decay_data)),:]],axis = 1)
            amp_fit_data = pd.concat([amp_fit_data,csv_dt_data.iloc[0:int(len(decay_data)),:]],axis = 1)
            
        amp_fit_data['max_idx'] = temp_max_idx_arr
        amp_fit_data['intensity_max'] = temp_max_val_arr
        tau_fit_data['max_idx'] = temp_max_idx_arr
        tau_fit_data['intensity_max'] = temp_max_val_arr

        try:
            amp_fit_data = amp_fit_data.drop(['Unnamed: 0'],axis = 1)
            tau_fit_data= tau_fit_data.drop(['Unnamed: 0'],axis = 1)
        except KeyError:
            print('No Unnamed colums')

        
        logger.info('Done!')


    ###############################################################################
    # Fitting function paramters
    ###############################################################################   
    exp_fit_params = lm.Parameters()
    exp_fit_params.add('tau',value = fit_dict['tau'], min = fit_dict['tau_min'], max = fit_dict['tau_max'], vary = fit_dict['tau_vary'])
    exp_fit_params.add('amp',value = fit_dict['amp'], min = fit_dict['amp_min'], max = fit_dict['amp_max'], vary = fit_dict['amp_vary'])
    exp_fit_params.add('c',value = fit_dict['c'], min = fit_dict['c_min'], max = fit_dict['c_max'], vary = fit_dict['c_vary'])

    ###############################################################################
    # First, we initialize a Minimizer and perform the grid search for grid_c1_amp and grid_c2_amp:
    # Here we do solutions for both amplitude method an tau method
    ###############################################################################  
    if fit_dict['grid_amp_method'] == True:
        logger.info('Amplitude method: True. Initilising paramters..')

        c1_amp_step = round((fit_dict['grid_c1_amp_max']-fit_dict['grid_c1_amp_min'])/fit_dict['grid_c1_amp_step'],3)
        c2_amp_step = round((fit_dict['grid_c2_amp_max']-fit_dict['grid_c2_amp_min'])/fit_dict['grid_c2_amp_step'],3)
        fit_dict['grid_c1_amp_step'] = c1_amp_step
        fit_dict['grid_c2_amp_step'] = c2_amp_step

        amp_params = lm.Parameters()
        amp_params.add('grid_c1_amp',value = fit_dict['grid_c1_amp'], min = fit_dict['grid_c1_amp_min'], max = fit_dict['grid_c1_amp_max'],brute_step = fit_dict['grid_c1_amp_step'])
        amp_params.add('grid_c2_amp',value = fit_dict['grid_c2_amp'], min = fit_dict['grid_c2_amp_min'], max = fit_dict['grid_c2_amp_max'],brute_step = fit_dict['grid_c2_amp_step'])

        logger.info('Starting simulation, hold on...')

        amp_fit_c1c2 = lm.Minimizer(amplitude_fit, amp_params, fcn_args = (time_data,decay_data,exp_fit_params,fit_dict,amp_fit_data,logger,),iter_cb=progress_update)
        amp_best_c1c2 = amp_fit_c1c2.minimize(method = 'brute',workers = fit_dict['num_cores'])
        logger.info('Done!')

    if fit_dict['grid_tau_method'] == True:
        logger.info('Grid Tau Method: True. Initilising paramters..')

        grid_c1_tau_step = round((abs(fit_dict['grid_c1_tau_max']-fit_dict['grid_c1_tau_min']))/fit_dict['grid_c1_tau_step'],3)
        grid_c2_tau_step = round((abs(fit_dict['grid_c2_tau_max']-fit_dict['grid_c2_tau_min']))/fit_dict['grid_c2_tau_step'],3)
        fit_dict['grid_c1_tau_step'] = grid_c1_tau_step
        fit_dict['grid_c2_tau_step'] = grid_c2_tau_step

        tau_params = lm.Parameters()
        tau_params.add('grid_c1_tau',value = fit_dict['grid_c1_tau'], min = fit_dict['grid_c1_tau_min'], max = fit_dict['grid_c1_tau_max'],brute_step =  fit_dict['grid_c1_tau_step'] )
        tau_params.add('grid_c2_tau',value = fit_dict['grid_c2_tau'], min = fit_dict['grid_c2_tau_min'], max = fit_dict['grid_c2_tau_max'],brute_step = fit_dict['grid_c2_tau_step'])
        
        logger.info('Starting simulation, hold on...')

        tau_fit_c1c2 = lm.Minimizer(tau_fit, tau_params, fcn_args = (time_data,decay_data,exp_fit_params,fit_dict,tau_fit_data,logger,),iter_cb=progress_update)
        tau_best_c1c2 = tau_fit_c1c2.minimize(method = 'brute',workers = fit_dict['num_cores'])

        logger.info('Done!')

    if fit_dict['amp_method'] == True and fit_dict['grid_amp_method'] != True:
        try:
            logger.info('Amplitude Method: True. Initilising paramters..')
            exp_fit_params = lm.Parameters()
            exp_fit_params.add('tau',value = fit_dict['tau'], min = fit_dict['tau_min'], max = fit_dict['tau_max'], vary = fit_dict['tau_vary'])
            exp_fit_params.add('amp',value = fit_dict['amp'], min = fit_dict['amp_min'], max = fit_dict['amp_max'], vary = fit_dict['amp_vary'])
            exp_fit_params.add('c',value = fit_dict['c'], min = fit_dict['c_min'], max = fit_dict['c_max'], vary = fit_dict['c_vary'])
            amp_fit_list = pmap.starmap(paralell_eval_amp,list(zip(time_data,decay_data)),exp_fit_params,fit_dict,logger,pm_pbar=True)
            start_idx = []
            end_idx = []
            time_start = []
            time_stop = []
            intensity_start = []
            intensity_end = []
            amplitude_guess_norm = []
            tau_guess_norm = []
            c_guess_norm = []
            amplitude_final_norm = []
            amplitude_final_norm_std = []
            tau_final_norm = []
            tau_final_norm_std = []
            c_final_norm = []
            c_final_norm_std = []
            tau_final = []
            tau_final_std = []
            guess_residual = []
            guess_iterations = []
            final_residual = []
            final_iterations = []
            for i in range(len(amp_fit_list)):
                start_idx.append(amp_fit_list[i][0])
                end_idx.append(amp_fit_list[i][1])
                time_start.append(amp_fit_list[i][2])
                time_stop.append(amp_fit_list[i][3])
                intensity_start.append(amp_fit_list[i][4])
                intensity_end.append(amp_fit_list[i][5])
                amplitude_guess_norm.append(amp_fit_list[i][6])
                tau_guess_norm.append(amp_fit_list[i][7])
                c_guess_norm.append(amp_fit_list[i][8])
                amplitude_final_norm.append(amp_fit_list[i][9])
                amplitude_final_norm_std.append(amp_fit_list[i][10])
                tau_final_norm.append(amp_fit_list[i][11])
                tau_final_norm_std.append(amp_fit_list[i][12])
                c_final_norm.append(amp_fit_list[i][13])
                c_final_norm_std.append(amp_fit_list[i][14])
                tau_final.append(amp_fit_list[i][15])
                tau_final_std.append(amp_fit_list[i][16])
                guess_residual.append(amp_fit_list[i][17])
                guess_iterations.append(amp_fit_list[i][18])
                final_residual.append(amp_fit_list[i][19])
                final_iterations.append(amp_fit_list[i][20])

            amp_fit_data['start_idx'] = start_idx
            amp_fit_data['end_idx'] = end_idx
            amp_fit_data['time_start'] = time_start
            amp_fit_data['time_stop'] = time_stop
            amp_fit_data['intensity_start'] = intensity_start
            amp_fit_data['intensity_end'] = intensity_end
            amp_fit_data['amplitude_guess_norm'] = amplitude_guess_norm
            amp_fit_data['tau_guess_norm'] = tau_guess_norm
            amp_fit_data['c_guess_norm'] = c_guess_norm
            amp_fit_data['amplitude_final_norm'] = amplitude_final_norm
            amp_fit_data['amplitude_final_norm_std'] = amplitude_final_norm_std
            amp_fit_data['tau_final_norm'] = tau_final_norm
            amp_fit_data['tau_final_norm_std'] = tau_final_norm_std
            amp_fit_data['c_final_norm'] = c_final_norm
            amp_fit_data['c_final_norm_std'] = c_final_norm_std
            amp_fit_data['tau_final'] = tau_final
            amp_fit_data['tau_final_std'] = tau_final_std
            amp_fit_data['guess_residual'] = guess_residual
            amp_fit_data['guess_iterations'] = guess_iterations
            amp_fit_data['final_residual'] = final_residual
            amp_fit_data['final_iterations'] = final_iterations
            gc.collect()
        except IndexError:
            logger.warning('Unstable soulutions found for amp fit.. Check input data.'+ fit_dict['load_data_dir'])
            logger.removeHandler(stdout_handler)
            logger.removeHandler(output_file_handler)   
            return

    if fit_dict['tau_method'] == True and fit_dict['grid_tau_method'] != True:
        try:
            logger.info('Tau Method: True. Initilising paramters..')
            exp_fit_params = lm.Parameters()
            exp_fit_params.add('tau',value = fit_dict['tau'], min = fit_dict['tau_min'], max = fit_dict['tau_max'], vary = fit_dict['tau_vary'])
            exp_fit_params.add('amp',value = fit_dict['amp'], min = fit_dict['amp_min'], max = fit_dict['amp_max'], vary = fit_dict['amp_vary'])
            exp_fit_params.add('c',value = fit_dict['c'], min = fit_dict['c_min'], max = fit_dict['c_max'], vary = fit_dict['c_vary'])
            tau_fit_list = pmap.starmap(paralell_eval_tau,list(zip(time_data,decay_data)),exp_fit_params,fit_dict,logger,pm_pbar=True)

            start_idx = []
            end_idx = []
            time_start = []
            time_stop = []
            intensity_start = []
            intensity_end = []
            amplitude_guess_norm = []
            tau_guess_norm = []
            c_guess_norm = []
            amplitude_final_norm = []
            amplitude_final_norm_std = []
            tau_final_norm = []
            tau_final_norm_std = []
            c_final_norm = []
            c_final_norm_std = []
            tau_final = []
            tau_final_std = []
            guess_residual = []
            guess_iterations = []
            final_residual = []
            final_iterations = []
            while_loops = []
            for i in range(len(tau_fit_list)):
                start_idx.append(tau_fit_list[i][0])
                end_idx.append(tau_fit_list[i][1])
                time_start.append(tau_fit_list[i][2])
                time_stop.append(tau_fit_list[i][3])
                intensity_start.append(tau_fit_list[i][4])
                intensity_end.append(tau_fit_list[i][5])
                amplitude_guess_norm.append(tau_fit_list[i][6])
                tau_guess_norm.append(tau_fit_list[i][7])
                c_guess_norm.append(tau_fit_list[i][8])
                amplitude_final_norm.append(tau_fit_list[i][9])
                amplitude_final_norm_std.append(tau_fit_list[i][10])
                tau_final_norm.append(tau_fit_list[i][11])
                tau_final_norm_std.append(tau_fit_list[i][12])
                c_final_norm.append(tau_fit_list[i][13])
                c_final_norm_std.append(tau_fit_list[i][14])
                tau_final.append(tau_fit_list[i][15])
                tau_final_std.append(tau_fit_list[i][16])
                guess_residual.append(tau_fit_list[i][17])
                guess_iterations.append(tau_fit_list[i][18])
                final_residual.append(tau_fit_list[i][19])
                final_iterations.append(tau_fit_list[i][20])
                while_loops.append(tau_fit_list[i][21])

            tau_fit_data['start_idx'] = start_idx
            tau_fit_data['end_idx'] = end_idx
            tau_fit_data['time_start'] = time_start
            tau_fit_data['time_stop'] = time_stop
            tau_fit_data['intensity_start'] = intensity_start
            tau_fit_data['intensity_end'] = intensity_end
            tau_fit_data['amplitude_guess_norm'] = amplitude_guess_norm
            tau_fit_data['tau_guess_norm'] = tau_guess_norm
            tau_fit_data['c_guess_norm'] = c_guess_norm
            tau_fit_data['amplitude_final_norm'] = amplitude_final_norm
            tau_fit_data['amplitude_final_norm_std'] = amplitude_final_norm_std
            tau_fit_data['tau_final_norm'] = tau_final_norm
            tau_fit_data['tau_final_std'] = tau_final_norm_std
            tau_fit_data['c_final_norm'] = c_final_norm
            tau_fit_data['c_final_norm_std'] = c_final_norm_std
            tau_fit_data['tau_final'] = tau_final
            tau_fit_data['tau_final_std'] = tau_final_std
            tau_fit_data['guess_residual'] = guess_residual
            tau_fit_data['guess_iterations'] = guess_iterations
            tau_fit_data['final_residual'] = final_residual
            tau_fit_data['final_iterations'] = final_iterations
            tau_fit_data['while_loops'] = while_loops
            gc.collect()
        except IndexError:
            logger.warning('Unstable soulutions found for tau fit.. Check input data in.' + fit_dict['load_data_dir'])
            logger.removeHandler(stdout_handler)
            logger.removeHandler(output_file_handler)  
            return


    ###############################################################################
    # Plotting the best results from the grid search and saving data
    ###############################################################################  

    if fit_dict['post_fit_anal'] == True and fit_dict['tau_method'] != True and fit_dict['grid_tau_method'] != True and fit_dict['amp_method'] != True and fit_dict['grid_amp_method'] != True:
        logger.info('Loading fited data...')
        tau_fit_data = pd.read_csv(fit_dict['save_data_dir']+fit_dict['tau_data'])
        amp_fit_data = pd.read_csv(fit_dict['save_data_dir']+fit_dict['amp_data'])
    else: 
        logger.info('Saving data to CSV...')
        tau_fit_data.to_csv(fit_dict['save_data_dir']+fit_dict['tau_data'])
        amp_fit_data.to_csv(fit_dict['save_data_dir']+fit_dict['amp_data'])


    logger.info('Done!')

    if fit_dict['plot_grid_data'] == True:

        logger.info('Plotting results from grid searches..')

        if fit_dict['grid_amp_method'] == True:
            amp_aspect = fit_dict['grid_c2_amp_step']/fit_dict['grid_c1_amp_step']
            amp_grid_fig = plt.figure()
            amp_ax_grid_fig = amp_grid_fig.add_subplot(111)
            amp_grid_im = amp_ax_grid_fig.imshow(amp_best_c1c2.brute_Jout,extent=[fit_dict['grid_c2_amp_min'],fit_dict['grid_c2_amp_max'],fit_dict['grid_c1_amp_min'],fit_dict['grid_c1_amp_max']],origin='lower',cmap='gray_r',interpolation='nearest',aspect=amp_aspect)
            cbar = amp_grid_fig.colorbar(amp_grid_im)
            cbar.set_label('Normalised error')
            plt.title('Amplitude method of C1 = '+ str(round(amp_best_c1c2.brute_x0[0],3)) + ' and C2 = ' + str(round(amp_best_c1c2.brute_x0[1],3)))
            plt.xlabel('C2')
            plt.ylabel('C1')
            plt.savefig(fit_dict['save_data_dir'] + '/amplitude_method_for_combinations_of_C1_and_C2.'+ fit_dict['img_format'],format = fit_dict['img_format'])

        if fit_dict['grid_tau_method'] == True:
            tau_aspect = fit_dict['grid_c2_tau_step']/fit_dict['grid_c1_tau_step']
            tau_grid_fig = plt.figure()
            tau_ax_grid_fig = tau_grid_fig.add_subplot(111)
            tau_grid_im = tau_ax_grid_fig.imshow(np.rot90(tau_best_c1c2.brute_Jout,k=2),extent=[fit_dict['grid_c2_tau_min'],fit_dict['grid_c2_tau_max'],fit_dict['grid_c1_tau_min'],fit_dict['grid_c1_tau_max']],origin='lower',cmap='gray_r',interpolation='nearest',aspect=tau_aspect)
            cbar = tau_grid_fig.colorbar(tau_grid_im)
            cbar.set_label('Normalised error')
            plt.title('Tau method of C1 = '+str(round(tau_best_c1c2.brute_x0[0],3)) + ' and C2 = ' + str(round(tau_best_c1c2.brute_x0[1],3)))
            plt.xlabel('C2')
            plt.ylabel('C1')
            plt.savefig(fit_dict['save_data_dir'] + '/tau_method_for_combinations_of_C1_and_C2.'+ fit_dict['img_format'],format = fit_dict['img_format'])

    amp_best_taus = amp_fit_data['tau_final_norm'].to_numpy()
    amp_best_fit_fig = plt.figure()
    amp_ax_best_fit_fig = amp_best_fit_fig.add_subplot(111)
    amp_ax_best_fit_fig.grid(True, which = 'both')
    amp_ax_best_fit_fig.scatter(np.arange(len(amp_best_taus)),amp_best_taus,alpha = 0.5, s = 5,marker ='.',c='k')
    plt.title('Normalised \u03C4(Amp method)')
    amp_ax_best_fit_fig.set_xlabel('Waveform')
    amp_ax_best_fit_fig.set_ylabel('Normalised \u03C4 [a.u]')
    amp_ax_best_fit_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
    amp_ax_best_fit_fig.minorticks_on()
    amp_ax_best_fit_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(fit_dict['save_data_dir'] + '/estimated_tau_amplitude_method.'+ fit_dict['img_format'],format = fit_dict['img_format'])

    tau_best_taus = tau_fit_data['tau_final_norm'].to_numpy()
    tau_best_fit_fig = plt.figure()
    tau_ax_best_fit_fig = tau_best_fit_fig.add_subplot(111,sharey = amp_ax_best_fit_fig)
    tau_ax_best_fit_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
    tau_ax_best_fit_fig.minorticks_on()
    tau_ax_best_fit_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    tau_ax_best_fit_fig.scatter(np.arange(len(tau_best_taus)),tau_best_taus,alpha = 0.5, s = 5,marker ='.',c='k')
    plt.title('Normalised \u03C4(Tau method)')    
    tau_ax_best_fit_fig.set_xlabel('Waveform')
    tau_ax_best_fit_fig.set_ylabel('Normalised \u03C4 [a.u]')
    plt.savefig(fit_dict['save_data_dir'] + '/estimated_tau_for_tau_method.'+ fit_dict['img_format'],format = fit_dict['img_format'])

    logger.info('Done!')

    ###############################################################################
    # Post fit analysis
    ###############################################################################  

    if fit_dict['diagnostic_on'] == True:

        logger.info('Diagnostic_on: True. Generating intresting plots..')

        logger.info('Plotting results from fitting...')

        if fit_dict['temp_data'] == True and fit_dict['load_data'] == True and fit_dict['simulate_data'] != True:
            # Plotting all the real taus
            tau_plot_fig = plt.figure()
            plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.scatter(tau_fit_data['Temperature'] + 273.15,tau_fit_data['tau_final'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
            plt.scatter(amp_fit_data['Temperature']+ 273.15,amp_fit_data['tau_final'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
            plt.ylim(min(tau_fit_data['tau_final']) - 0.5*min(tau_fit_data['tau_final']),max(tau_fit_data['tau_final']) + 0.5*max(tau_fit_data['tau_final']))
            plt.yscale('log')
            plt.title('Decay time [\u03C4]')
            plt.xlabel('Temperature [K]')
            plt.ylabel(' \u03C4 [s]')
            plt.legend()
            plt.savefig(fit_dict['save_data_dir'] + '/Decay_times.'+ fit_dict['img_format'],format = fit_dict['img_format'])
            
            tau_dif_plot_fig = plt.figure()
            plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.scatter(tau_fit_data['Temperature'] + 273.15,abs(tau_fit_data['tau_final']-amp_fit_data['tau_final']),alpha = 0.5, s = 5,marker ='.',c = 'k')
            plt.yscale('log')
            plt.title('Decay time difference Tau vs Amp [\u03C4]')
            plt.xlabel('Temperature [K]')
            plt.ylabel('$\Delta$ \u03C4 [s]')
            plt.ylim(min(abs(tau_fit_data['tau_final']-amp_fit_data['tau_final'])) - 0.5*min(abs(tau_fit_data['tau_final']-amp_fit_data['tau_final'])),max(abs(tau_fit_data['tau_final']-amp_fit_data['tau_final'])) + 0.5*max(abs(tau_fit_data['tau_final']-amp_fit_data['tau_final'])))
            plt.legend()
            plt.savefig(fit_dict['save_data_dir'] + '/delta_decay_times.'+ fit_dict['img_format'],format = fit_dict['img_format'])
        else:
            # Plotting all the real taus
            tau_plot_fig = plt.figure()
            plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['tau_final'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
            plt.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['tau_final'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
            plt.title('Decay time [\u03C4]')
            plt.xlabel('Waveform')
            plt.yscale('log')
            plt.ylim(min(tau_fit_data['tau_final']) - 0.5*min(tau_fit_data['tau_final']),max(tau_fit_data['tau_final']) + 0.5*max(tau_fit_data['tau_final']))
            plt.ylabel('\u03C4 [s]')
            plt.legend()
            plt.savefig(fit_dict['save_data_dir'] + '/Decay_times.'+ fit_dict['img_format'],format = fit_dict['img_format'])

            tau_dif_plot_fig = plt.figure()
            plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            plt.scatter(np.arange(len(tau_fit_data['tau_final'])),abs(tau_fit_data['tau_final']-amp_fit_data['tau_final']),alpha = 0.5, s = 5,marker ='.',c = 'k')
            plt.yscale('log')
            plt.title('Decay time difference Tau vs Amp [\u03C4]')
            plt.xlabel('Waveform')
            plt.ylabel('$\Delta$ \u03C4 [s]')
            plt.ylim(min(abs(tau_fit_data['tau_final']-amp_fit_data['tau_final'])) - 0.5*min(abs(tau_fit_data['tau_final']-amp_fit_data['tau_final'])),max(abs(tau_fit_data['tau_final']-amp_fit_data['tau_final'])) + 0.5*max(abs(tau_fit_data['tau_final']-amp_fit_data['tau_final'])))
            plt.legend()
            plt.savefig(fit_dict['save_data_dir'] + '/delta_decay_times.'+ fit_dict['img_format'],format = fit_dict['img_format'])
        
        tau_norm_plot_fig = plt.figure()
        plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.scatter(np.arange(len(tau_fit_data['tau_final_norm'])),tau_fit_data['tau_final_norm'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
        plt.scatter(np.arange(len(amp_fit_data['tau_final_norm'])),amp_fit_data['tau_final_norm'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
        plt.title('Decay time normalised [\u03C4]')
        plt.xlabel('Waveform')
        plt.ylabel('\u03C4 [s]')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/Decay_times_norm.'+ fit_dict['img_format'],format = fit_dict['img_format'])
        
        if fit_dict['loss_metric'] == 2 or fit_dict['loss_metric'] == 3:
            
            c_plot_fig = plt.figure()
            ax_c_plot_fig = c_plot_fig.add_subplot(111)
            ax_c_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            ax_c_plot_fig.minorticks_on()
            ax_c_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax_c_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['tau_final_norm_std'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
            ax_c_plot_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['tau_final_norm_std'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
            plt.title('Standard deviation of $\sigma$\u03C4')
            ax_c_plot_fig.set_xlabel('Waveform')
            ax_c_plot_fig.set_ylabel('$\sigma$\u03C4[a.u]')
            plt.legend()
            plt.savefig(fit_dict['save_data_dir'] + '/Decay_times_norm_std.'+ fit_dict['img_format'],format = fit_dict['img_format'])

            
            c_plot_fig = plt.figure()
            ax_c_plot_fig = c_plot_fig.add_subplot(111)
            ax_c_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            ax_c_plot_fig.minorticks_on()
            ax_c_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax_c_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['tau_final_norm_std']/tau_fit_data['tau_final_norm'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
            ax_c_plot_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['tau_final_norm_std']/amp_fit_data['tau_final_norm'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
            plt.title('Coefficent of variation $\sigma$\u03C4/\u03C4')
            ax_c_plot_fig.set_xlabel('Waveform')
            ax_c_plot_fig.set_ylabel('$\sigma$\u03C4/\u03C4[a.u]')
            plt.legend()
            plt.savefig(fit_dict['save_data_dir'] + '/Decay_times_normcoef.'+ fit_dict['img_format'],format = fit_dict['img_format'])

        
        c_plot_fig = plt.figure()
        ax_c_plot_fig = c_plot_fig.add_subplot(111)
        ax_c_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        ax_c_plot_fig.minorticks_on()
        ax_c_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax_c_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['c_final_norm'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
        ax_c_plot_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['c_final_norm'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
        plt.title('Normilised Bias [c]')
        ax_c_plot_fig.set_xlabel('Waveform')
        ax_c_plot_fig.set_ylabel('c[a.u]')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/norm_biases.'+ fit_dict['img_format'],format = fit_dict['img_format'])


        # Plotting all the normalised Amplitudes
        amp_plot_fig = plt.figure()
        ax_amp_plot_fig = amp_plot_fig.add_subplot(111)
        ax_amp_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        ax_amp_plot_fig.minorticks_on()
        ax_amp_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax_amp_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['amplitude_final_norm'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
        ax_amp_plot_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['amplitude_final_norm'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
        plt.title('Normilised Amplitude')
        ax_amp_plot_fig.set_xlabel('Waveform')
        ax_amp_plot_fig.set_ylabel('Amplitude [a.u]')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/norm_amplitudes.'+ fit_dict['img_format'],format = fit_dict['img_format'])

        # Plot real amplitudes 
        real_amp_fig = plt.figure()
        ax_real_amp_fig = real_amp_fig.add_subplot(111)
        ax_real_amp_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        ax_real_amp_fig.minorticks_on()
        ax_real_amp_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax_real_amp_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['intensity_max'],alpha = 0.5, s = 5,marker ='v',c = 'k',label ='Max amplitude')
        ax_real_amp_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['intensity_start'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit Amplitude')
        ax_real_amp_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['intensity_start'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit Amplitude')
        plt.title('Max signal value')
        ax_real_amp_fig.set_xlabel('Waveform')
        ax_real_amp_fig.set_ylabel('Voltage [V]')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/amplitudes.'+ fit_dict['img_format'],format = fit_dict['img_format'])    

        # Plotting all the normalised reduced residuals
        chi_plot_fig = plt.figure()
        ax_chi_plot_fig = chi_plot_fig.add_subplot(111)
        ax_chi_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        ax_chi_plot_fig.minorticks_on()
        ax_chi_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax_chi_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['final_residual'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
        ax_chi_plot_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['final_residual'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
        ax_chi_plot_fig.set_ylim(np.min(tau_fit_data['final_residual'])-0.2*abs(np.min(tau_fit_data['final_residual'])),np.max(tau_fit_data['final_residual'])+0.2*abs(np.max(tau_fit_data['final_residual'])))
        plt.yscale('log')
        plt.title('Reduced chi squared')
        ax_chi_plot_fig.set_xlabel('Waveform')
        ax_chi_plot_fig.set_ylabel('Chi squared [a.u]')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/chi_squareds.'+ fit_dict['img_format'],format = fit_dict['img_format'])

        # Plotting the fitted data length
        data_length_plot_fig = plt.figure()
        ax_data_length_plot_fig = data_length_plot_fig.add_subplot(111)
        ax_data_length_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        ax_data_length_plot_fig.minorticks_on()
        ax_data_length_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax_data_length_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['end_idx'] - tau_fit_data['start_idx'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
        ax_data_length_plot_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['end_idx'] - amp_fit_data['start_idx'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
        plt.title('Fitted data length')
        plt.yscale('log')
        ax_data_length_plot_fig.set_xlabel('Waveform')
        ax_data_length_plot_fig.set_ylabel('Data length')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/dat_length.'+ fit_dict['img_format'],format = fit_dict['img_format'])

        # Plotting the fitted data length
        gIter_plot_fig = plt.figure()
        ax_gIter_plot_fig = gIter_plot_fig.add_subplot(111)
        ax_gIter_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        ax_gIter_plot_fig.minorticks_on()
        ax_gIter_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax_gIter_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['guess_iterations'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
        ax_gIter_plot_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['guess_iterations'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
        plt.title('Guess iterations')
        ax_gIter_plot_fig.set_xlabel('Waveform')
        ax_gIter_plot_fig.set_ylabel('Iterations')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/guess_iter.'+ fit_dict['img_format'],format = fit_dict['img_format'])

        fIter_plot_fig = plt.figure()
        ax_fIter_plot_fig = fIter_plot_fig.add_subplot(111)
        ax_fIter_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        ax_fIter_plot_fig.minorticks_on()
        ax_fIter_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax_fIter_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['final_iterations'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
        ax_fIter_plot_fig.scatter(np.arange(len(amp_fit_data['tau_final'])),amp_fit_data['final_iterations'],alpha = 0.5, s = 5,marker ='.',c= '0.5',label = 'Amp Fit')
        plt.title('Final iterations')
        ax_fIter_plot_fig.set_xlabel('Waveform')
        ax_fIter_plot_fig.set_ylabel('Iterations')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/final_iter.'+ fit_dict['img_format'],format = fit_dict['img_format'])

        # Plotting the while loops
        while_loop_plot_fig = plt.figure()
        ax_while_loop_plot_fig = while_loop_plot_fig.add_subplot(111)
        ax_while_loop_plot_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
        ax_while_loop_plot_fig.minorticks_on()
        ax_while_loop_plot_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        ax_while_loop_plot_fig.scatter(np.arange(len(tau_fit_data['tau_final'])),tau_fit_data['while_loops'],alpha = 0.5, s = 5,marker ='.',c = 'k',label ='Tau Fit')
        plt.title('While loops in tau fit')
        ax_while_loop_plot_fig.set_xlabel('Waveform')
        ax_while_loop_plot_fig.set_ylabel('While loops')
        plt.legend()
        plt.savefig(fit_dict['save_data_dir'] + '/while_loops.'+ fit_dict['img_format'],format = fit_dict['img_format'])


        if fit_dict['analyse_random'] == True:
            anal_idx = np.random.randint(len(decay_data))
        else:
            anal_idx = fit_dict['analyse_random']

        if fit_dict['load_data'] == True:
            anal_decay_data = decay_data[anal_idx]
            anal_time_data = time_data[anal_idx]
        else:
            anal_decay_data = decay_data[anal_idx]
            anal_time_data = time_data[anal_idx]
        
        if fit_dict['monte_carlo'] == True:
            logger.info('Starting Markov Chain Monte Carlo sampling...')

            logger.info('Analysing Tau Method...')
            tau_MCMC_sampler(anal_time_data,anal_decay_data,exp_func_MC,fit_dict)

            logger.info('Analysing Amplitude Method...')
            amplitude_MCMC_sampler(anal_time_data,anal_decay_data,exp_func_MC,fit_dict)

        ###############################################################################
        # Compairing fits
        ###############################################################################  
        
        data_points = np.linspace(0,len(decay_data)-1,fit_dict['num_data_fig'])

        for i in tqdm(range(fit_dict['num_data_fig'])):
            dat_point = int(data_points[i])

            if fit_dict['norm_data'] == 1:
                norm_time = time_data[dat_point]/time_data[dat_point][-1]
                norm_data = decay_data[dat_point]/decay_data[dat_point][0]
            elif fit_dict['norm_data'] == 2:
                norm_time = time_data[dat_point]/time_data[dat_point][-1]
                norm_data = decay_data[dat_point]/np.max(decay_data[dat_point])
            tau_plot_vec = norm_time[int(tau_fit_data['start_idx'][dat_point]):int(tau_fit_data['end_idx'][dat_point])]
            amp_plot_vec= norm_time[int(amp_fit_data['start_idx'][dat_point]):int(amp_fit_data['end_idx'][dat_point])]

            comp_fig = plt.figure()
            ax_comp_fig = comp_fig.add_subplot(121)
            ax_comp_fig.scatter(norm_time,abs(norm_data),marker ='.',c='0.75',alpha = 0.5, s = 10,label = 'Waveform: ' + str(dat_point))
            ax_comp_fig.plot(tau_plot_vec,exp_func_MC(tau_plot_vec,tau_fit_data['amplitude_final_norm'][dat_point],tau_fit_data['tau_final_norm'][dat_point],tau_fit_data['c_final_norm'][dat_point]),c = 'k',linestyle = '--',label = 'Tau Fit')
            ax_comp_fig.plot(amp_plot_vec,exp_func_MC(amp_plot_vec,amp_fit_data['amplitude_final_norm'][dat_point],amp_fit_data['tau_final_norm'][dat_point],amp_fit_data['c_final_norm'][dat_point]),color = '0.25',linestyle = ':',label = 'Amplitude Fit')
            ax_comp_fig.set_yscale('log')
            ax_comp_fig.set_ylim(norm_data.min() - 0.5*norm_data.min(),1.1)
            ax_comp_fig.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            ax_comp_fig.minorticks_on()
            ax_comp_fig.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax_comp_fig.set_ylabel('Normalised Amplitude [a.u]')
            ax_comp_fig.set_xlabel('Normalised Time [a.u]')
            ax_comp_fig.legend()

            ax_comp_fig_2 = comp_fig.add_subplot(122)
            ax_comp_fig_2.scatter(norm_time,abs(norm_data),marker ='.',c='0.75',alpha = 0.5, s = 10,label = 'Waveform: ' + str(dat_point))
            ax_comp_fig_2.plot(tau_plot_vec,exp_func_MC(tau_plot_vec,tau_fit_data['amplitude_final_norm'][dat_point],tau_fit_data['tau_final_norm'][dat_point],tau_fit_data['c_final_norm'][dat_point]),c = 'k',linestyle = '--',label = 'Tau Fit')
            ax_comp_fig_2.plot(amp_plot_vec,exp_func_MC(amp_plot_vec,amp_fit_data['amplitude_final_norm'][dat_point],amp_fit_data['tau_final_norm'][dat_point],amp_fit_data['c_final_norm'][dat_point]),color = '0.25',linestyle = ':',label = 'Amplitude Fit')

            ax_comp_fig_2.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            ax_comp_fig_2.minorticks_on()
            ax_comp_fig_2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            # ax_comp_fig_2.set_ylabel('Normalised Amplitude [a.u]')
            ax_comp_fig_2.set_xlabel('Normalised Time [a.u]')
            ax_comp_fig_2.legend()
            plt.tight_layout()
            plt.savefig(fit_dict['save_data_dir'] + '/comp_tau_amp'+str(dat_point)+'.'+ fit_dict['img_format'],format = fit_dict['img_format'])
        
        if fit_dict['interactive_plots'] == True:
            data_plots = []
            tau_plots = []
            amp_plots = []
            for i in range(len(decay_data)):
                if fit_dict['norm_data'] == 1:
                    norm_time = time_data[i]/time_data[i][-1]
                    norm_data = decay_data[i]/decay_data[i][0]
                elif fit_dict['norm_data'] == 2:
                    norm_time = time_data[i]/time_data[i][-1]
                    norm_data = decay_data[i]/np.max(decay_data[i])
                tau_plot_vec = norm_time[int(tau_fit_data['start_idx'][i]):int(tau_fit_data['end_idx'][i])]
                amp_plot_vec= norm_time[int(amp_fit_data['start_idx'][i]):int(amp_fit_data['end_idx'][i])]

                data_plots.append((norm_time,norm_data))
                tau_plots.append((tau_plot_vec,exp_func_MC(tau_plot_vec,tau_fit_data['amplitude_final_norm'][i],tau_fit_data['tau_final_norm'][i],tau_fit_data['c_final_norm'][i])))
                amp_plots.append((amp_plot_vec,exp_func_MC(amp_plot_vec,amp_fit_data['amplitude_final_norm'][i],amp_fit_data['tau_final_norm'][i],amp_fit_data['c_final_norm'][i])))
            
            
            def key_event(e):
                global curr_pos

                if e.key == "right":
                    curr_pos = curr_pos + 1
                elif e.key == "left":
                    curr_pos = curr_pos - 1
                else:
                    return
                curr_pos = curr_pos % len(data_plots)

                ax1.cla()
                ax1.scatter(data_plots[curr_pos][0], abs(data_plots[curr_pos][1]),marker ='.',c='0.75',alpha = 0.5, s = 10,label = 'Waveform: ' + str(curr_pos))
                ax1.plot(tau_plots[curr_pos][0], tau_plots[curr_pos][1],c = 'k',linestyle = '--',label = 'Tau Fit')
                ax1.plot(amp_plots[curr_pos][0], amp_plots[curr_pos][1],color = '0.25',linestyle = ':',label = 'Amplitude Fit')
                ax1.legend()
                ax1.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
                ax1.minorticks_on()
                ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                ax1.set_ylabel('Normalised Amplitude [a.u]')

                ax2.cla()
                ax2.scatter(data_plots[curr_pos][0], abs(data_plots[curr_pos][1]),marker ='.',c='0.75',alpha = 0.5, s = 10,label = 'Waveform: ' + str(curr_pos))
                ax2.plot(tau_plots[curr_pos][0], tau_plots[curr_pos][1],c = 'k',linestyle = '--',label = 'Tau Fit')
                ax2.plot(amp_plots[curr_pos][0], amp_plots[curr_pos][1],color = '0.25',linestyle = ':',label = 'Amplitude Fit')
                ax2.legend()
                ax2.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
                ax2.minorticks_on()
                ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                ax2.set_ylabel('Normalised Amplitude [a.u]')
                ax2.set_xlabel('Normalised Time [a.u]')
                ax2.set_yscale('log')
                interactive_fig.canvas.draw()
            
            curr_pos = fit_dict['current_pos']
            interactive_fig = plt.figure()
            interactive_fig.canvas.mpl_connect('key_press_event', key_event)
            ax1 = interactive_fig.add_subplot(211)
            ax1.scatter(data_plots[curr_pos][0], abs(data_plots[curr_pos][1]),marker ='.',c='0.75',alpha = 0.5, s = 10,label = 'Waveform: ' + str(curr_pos))
            ax1.plot(tau_plots[curr_pos][0], tau_plots[curr_pos][1],c = 'k',linestyle = '--',label = 'Tau Fit')
            ax1.plot(amp_plots[curr_pos][0], amp_plots[curr_pos][1],color = '0.25',linestyle = ':',label = 'Amplitude Fit')
            ax1.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            ax1.minorticks_on()
            ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax1.set_ylabel('Normalised Amplitude [a.u]')
            ax1.legend()

            ax2 = interactive_fig.add_subplot(212)
            ax2.scatter(data_plots[curr_pos][0], abs(data_plots[curr_pos][1]),marker ='.',c='0.75',alpha = 0.5, s = 10,label = 'Waveform: ' + str(curr_pos))
            ax2.plot(tau_plots[curr_pos][0], tau_plots[curr_pos][1],c = 'k',linestyle = '--',label = 'Tau Fit')
            ax2.plot(amp_plots[curr_pos][0], amp_plots[curr_pos][1],color = '0.25',linestyle = ':',label = 'Amplitude Fit')
            ax2.grid(b=True, which='major', color='#666666', linestyle='-',alpha = 0.8)
            ax2.minorticks_on()
            ax2.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            ax2.set_ylabel('Normalised Amplitude [a.u]')
            ax2.set_xlabel('Normalised Time [a.u]')
            ax2.set_yscale('log')
            ax2.legend()

            for fig_num in plt.get_fignums():
                if interactive_fig.number != fig_num:
                    plt.close(fig_num)
            plt.show()
    del tau_fit_data
    del amp_fit_data
    logger.removeHandler(stdout_handler)
    logger.removeHandler(output_file_handler)   

def paralell_eval_amp(x,y,fit_param,fit_dict,logger):
    '''
    Same functionality as tau_fit function however this one works for a singular value of C1 and C2 eg. C1 = 0.75 and C2 = 0.1, see fit_dict.
    This one is created to be able to paralellise the function for faster evaluation.
    However minor changes have been made, due to how the parlellisation of the package parmap works https://github.com/zeehio/parmap.
    As can be seen no allocations is made to the data and it only returns a tuple of all the data needed, which is returned by the parmap.starmap as 
    a list of tuples which have the same length as the input data.
    
    Example usage taken from fit_exp:

    exp_fit_params = lm.Parameters() # Createing a Parameter class, for detalied information see https://lmfit.github.io/lmfit-py/
    exp_fit_params.add('tau',value = fit_dict['tau'], min = fit_dict['tau_min'], max = fit_dict['tau_max'], vary = fit_dict['tau_vary']) # Adding Paramters to the class
    exp_fit_params.add('amp',value = fit_dict['amp'], min = fit_dict['amp_min'], max = fit_dict['amp_max'], vary = fit_dict['amp_vary']) # Adding Paramters to the class
    exp_fit_params.add('c',value = fit_dict['c'], min = fit_dict['c_min'], max = fit_dict['c_max'], vary = fit_dict['c_vary']) # Adding Paramters to the class
    amp_fit_list = pmap.starmap(paralell_eval_amp,list(zip(time_data,decay_data)),exp_fit_params,fit_dict,pm_pbar=True) # Paralell computing using parmap, time_data and decay data is list of arrays that have the same length.

    # Here im extracting the data from the amp_fit_list that is a list of tuples containg the data
    start_idx = []
    end_idx = []
    time_start = []
    time_stop = []
    intensity_start = []
    intensity_end = []
    amplitude_guess_norm = []
    tau_guess_norm = []
    c_guess_norm = []
    amplitude_final_norm = []
    tau_final_norm = []
    c_final_norm = []
    tau_final = []
    guess_residual = []
    guess_iterations = []
    final_residual = []
    final_iterations = []

    for i in range(len(amp_fit_list)):
        start_idx.append(amp_fit_list[i][0])
        end_idx.append(amp_fit_list[i][1])
        time_start.append(amp_fit_list[i][2])
        time_stop.append(amp_fit_list[i][3])
        intensity_start.append(amp_fit_list[i][4])
        intensity_end.append(amp_fit_list[i][5])
        amplitude_guess_norm.append(amp_fit_list[i][6])
        tau_guess_norm.append(amp_fit_list[i][7])
        c_guess_norm.append(amp_fit_list[i][8])
        amplitude_final_norm.append(amp_fit_list[i][9])
        tau_final_norm.append(amp_fit_list[i][10])
        c_final_norm.append(amp_fit_list[i][11])
        tau_final.append(amp_fit_list[i][12])
        guess_residual.append(amp_fit_list[i][13])
        guess_iterations.append(amp_fit_list[i][14])
        final_residual.append(amp_fit_list[i][15])
        final_iterations.append(amp_fit_list[i][16])

    amp_fit_data['start_idx'] = start_idx
    amp_fit_data['end_idx'] = end_idx
    amp_fit_data['time_start'] = time_start
    amp_fit_data['time_stop'] = time_stop
    amp_fit_data['intensity_start'] = intensity_start
    amp_fit_data['intensity_end'] = intensity_end
    amp_fit_data['amplitude_guess_norm'] = amplitude_guess_norm
    amp_fit_data['tau_guess_norm'] = tau_guess_norm
    amp_fit_data['c_guess_norm'] = c_guess_norm
    amp_fit_data['amplitude_final_norm'] = amplitude_final_norm
    amp_fit_data['tau_final_norm'] = tau_final_norm
    amp_fit_data['c_final_norm'] = c_final_norm
    amp_fit_data['tau_final'] = tau_final
    amp_fit_data['guess_residual'] = guess_residual
    amp_fit_data['guess_iterations'] = guess_iterations
    amp_fit_data['final_residual'] = final_residual
    amp_fit_data['final_iterations'] = final_iterations
    '''
    time_data = x
    decay_data = y

    time_norm_fit,decay_norm_fit,T_start,T_end = prep_data(time_data,decay_data,fit_dict,fit_dict['c1_amp'],fit_dict['c2_amp'],cut_data=True)

    estTau = super_rough_guess(time_norm_fit,decay_norm_fit,fit_dict)

    fit_param['tau'].value = estTau
    fit_param['amp'].value = decay_norm_fit[0]

    guess_result = lm.minimize(exp_func,fit_param,args=(time_norm_fit,decay_norm_fit,),method= fit_dict['rough_guess_method'],nan_policy=fit_dict['rough_guess_nan_policy'],reduce_fcn= fit_dict['rough_guess_reduce_func'])

    if guess_result.success != True:
        logger.warning('Guess method fit did not converge:\n %s,\n %s',lm.fit_report(guess_result),guess_result.params)

    fit_param['amp'].value = guess_result.params['amp'].value
    fit_param['tau'].value = guess_result.params['tau'].value

    result = lm.minimize(exp_func,fit_param,args=(time_norm_fit,decay_norm_fit,),method = fit_dict['final_guess_method'],nan_policy = fit_dict['final_guess_nan_policy'],reduce_fcn = fit_dict['final_guess_reduce_func'])
    if result.success != True:
        logger.warning('Final method fit did not converge:\n %s,\n %s',lm.fit_report(result),result.params)

    start_idx = T_start
    end_idx = T_end
    time_start = time_data[T_start]
    time_stop = time_data[T_end]
    intensity_start = decay_data[T_start]
    intensity_end = decay_data[T_end]
    amplitude_guess_norm = guess_result.params['amp'].value
    tau_guess_norm = guess_result.params['tau'].value
    c_guess_norm = guess_result.params['c'].value
    amplitude_final_norm = result.params['amp'].value
    try:
        amplitude_final_norm_std = result.params['amp'].stderr
    except (NameError ,RuntimeError,TypeError):
        logger.warning('Amplitude standard deviation could not be computed:\n %s,\n %s',lm.fit_report(result),result.params)
        amplitude_final_norm_std = 0
    tau_final_norm = result.params['tau'].value
    try:
        tau_final_norm_std = result.params['tau'].stderr
        tau_final_std = result.params['tau'].stderr*time_data[-1]
    except (NameError ,RuntimeError,TypeError):
        logger.warning('Tau standard deviation could not be computed:\n %s,\n %s',lm.fit_report(result),result.params)
        tau_final_norm_std = 0
        tau_final_std = 0
    
    c_final_norm = result.params['c'].value
    try:
        c_final_norm_std = result.params['c'].stderr
    except (NameError ,RuntimeError,TypeError):
        logger.warning('C standard deviation could not be computed:\n %s,\n %s',lm.fit_report(result),result.params)
        c_final_norm_std = 0
    tau_final = result.params['tau'].value*time_data[-1]
    guess_residual = guess_result.redchi
    guess_iterations = guess_result.nfev
    final_residual = result.redchi
    final_iterations = result.nfev
    return (start_idx,end_idx,time_start,time_stop,intensity_start,intensity_end,amplitude_guess_norm,tau_guess_norm,c_guess_norm,amplitude_final_norm,amplitude_final_norm_std,tau_final_norm,tau_final_norm_std,c_final_norm,c_final_norm_std,tau_final,tau_final_std,guess_residual,guess_iterations,final_residual,final_iterations)

def paralell_eval_tau(x,y,fit_param,fit_dict,logger):
    '''
    Same functionality as tau_fit function however this one works for a singular value of C1 and C2 eg. C1 = 0.75 and C2 = 0.1, see fit_dict.
    This one is created to be able to paralellise the function for faster evaluation.
    However minor changes have been made, due to how the parlellisation of the package parmap works https://github.com/zeehio/parmap.
    As can be seen no allocations is made to the data and it only returns a tuple of all the data needed, which is returned by the parmap.starmap as 
    a list of tuples which have the same length as the input data.
    
    Example usage taken from fit_exp:

    exp_fit_params = lm.Parameters() # Createing a Parameter class, for detalied information see https://lmfit.github.io/lmfit-py/
    exp_fit_params.add('tau',value = fit_dict['tau'], min = fit_dict['tau_min'], max = fit_dict['tau_max'], vary = fit_dict['tau_vary']) # Adding Paramters to the class
    exp_fit_params.add('amp',value = fit_dict['amp'], min = fit_dict['amp_min'], max = fit_dict['amp_max'], vary = fit_dict['amp_vary']) # Adding Paramters to the class
    exp_fit_params.add('c',value = fit_dict['c'], min = fit_dict['c_min'], max = fit_dict['c_max'], vary = fit_dict['c_vary']) # Adding Paramters to the class
    tau_fit_list = pmap.starmap(paralell_eval_tau,list(zip(time_data,decay_data)),exp_fit_params,fit_dict,pm_pbar=True) # Paralell computing using parmap, time_data and decay data is list of arrays that have the same length.
    '''
    time_data = x
    decay_data = y

    if fit_dict['norm_data'] == 1:
        time_data_cut_norm = time_data/time_data[-1]
        decay_data_cut_norm = decay_data/decay_data[0]
    elif fit_dict['norm_data'] == 2:
        time_data_cut_norm = time_data/time_data[-1]
        decay_data_cut_norm = decay_data/np.max(decay_data)

    giga_rough_guess = super_rough_guess(time_data_cut_norm,decay_data_cut_norm,fit_dict)
    new_tau = giga_rough_guess
        
    k = 0
    fit_param['amp'].value = decay_data_cut_norm[0]
    fit_param['tau'].value = new_tau
        
    guess_result = lm.minimize(exp_func,fit_param,args=(time_data_cut_norm,decay_data_cut_norm,),method= fit_dict['rough_guess_method'],nan_policy=fit_dict['rough_guess_nan_policy'],reduce_fcn= fit_dict['rough_guess_reduce_func'])

    if guess_result.success != True:
        logger.warning('Tau Method Guess method fit did not converge:\n %s,\n %s',lm.fit_report(guess_result),guess_result.params)

    fit_param['amp'].value = guess_result.params['amp'].value
    fit_param['tau'].value = guess_result.params['tau'].value
    fit_param['c'].value = guess_result.params['c'].value
    new_tau = guess_result.params['tau'].value

    old_tau = 100
    while np.abs(old_tau - new_tau) > fit_dict['tau_diff']:
        
        T_start = find_nearest(time_data_cut_norm,fit_dict['c1_tau']*new_tau)
        T_end = find_nearest(time_data_cut_norm,fit_dict['c2_tau']*new_tau)

        old_tau = new_tau
        time_norm_fit = time_data_cut_norm[T_start:T_end]
        decay_norm_fit = decay_data_cut_norm[T_start:T_end]

        if (T_end-T_start) < 30:
            # print('Short fit vector: ' + str(T_end-T_start))
            guess_result = lm.minimize(exp_func,fit_param,args=(time_data_cut_norm,decay_data_cut_norm,),method= fit_dict['rough_guess_method'],nan_policy=fit_dict['rough_guess_nan_policy'],reduce_fcn= fit_dict['rough_guess_reduce_func'])
            fit_param['amp'].value = guess_result.params['amp'].value
            fit_param['tau'].value = guess_result.params['tau'].value
            fit_param['c'].value = guess_result.params['c'].value
            new_tau = guess_result.params['tau'].value
            T_start = find_nearest(time_data_cut_norm,fit_dict['c1_tau']*new_tau)
            T_end = find_nearest(time_data_cut_norm,fit_dict['c2_tau']*new_tau)
            time_norm_fit = time_data_cut_norm[T_start:T_end]
            decay_norm_fit = decay_data_cut_norm[T_start:T_end]
            result = lm.minimize(exp_func,fit_param,args=(time_norm_fit,decay_norm_fit,),method = fit_dict['final_guess_method'],nan_policy = fit_dict['final_guess_nan_policy'],reduce_fcn = fit_dict['final_guess_reduce_func'])
            break

        result = lm.minimize(exp_func,fit_param,args=(time_norm_fit,decay_norm_fit,),method = fit_dict['final_guess_method'],nan_policy = fit_dict['final_guess_nan_policy'],reduce_fcn = fit_dict['final_guess_reduce_func'])

        new_tau = result.params['tau'].value
        k+= 1
        if k > fit_dict['while_loops']:
            break
    
    if result.success != True:
        logger.warning('Tau Method Guess method fit did not converge:\n %s,\n %s',lm.fit_report(guess_result),guess_result.params)
    
    start_idx = T_start
    end_idx = T_end
    time_start = time_data[T_start]
    time_stop = time_data[T_end]
    intensity_start = decay_data[T_start]
    intensity_end = decay_data[T_end]
    amplitude_guess_norm = guess_result.params['amp'].value
    tau_guess_norm = guess_result.params['tau'].value
    c_guess_norm = guess_result.params['c'].value
    amplitude_final_norm = result.params['amp'].value
    try:
        amplitude_final_norm_std = result.params['amp'].stderr
    except (NameError ,RuntimeError,TypeError):
        logger.warning('Amplitude standard deviation could not be computed:\n %s,\n %s',lm.fit_report(result),result.params)
        amplitude_final_norm_std = 0
    tau_final_norm = result.params['tau'].value
    try:
        tau_final_norm_std = result.params['tau'].stderr
        tau_final_std = result.params['tau'].stderr*time_data[-1]
    except (NameError ,RuntimeError,TypeError):
        logger.warning('Tau standard deviation could not be computed:\n %s,\n %s',lm.fit_report(result),result.params)
        tau_final_norm_std = 0
        tau_final_std = 0
    c_final_norm = result.params['c'].value
    try:
        c_final_norm_std = result.params['c'].stderr
    except (NameError ,RuntimeError,TypeError):
        logger.warning('C standard deviation could not be computed:\n %s,\n %s',lm.fit_report(result),result.params)
        c_final_norm_std = 0
    tau_final = result.params['tau'].value*time_data[-1]
    guess_residual = guess_result.redchi
    guess_iterations = guess_result.nfev
    final_residual = result.redchi
    final_iterations = result.nfev
    while_loops = k
    
    return (start_idx,end_idx,time_start,time_stop,intensity_start,intensity_end,amplitude_guess_norm,tau_guess_norm,c_guess_norm,amplitude_final_norm,amplitude_final_norm_std,tau_final_norm,tau_final_norm_std,c_final_norm,c_final_norm_std,tau_final,tau_final_std,guess_residual,guess_iterations,final_residual,final_iterations,while_loops)

