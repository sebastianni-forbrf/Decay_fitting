'''
Created by: Sebastian Nilsson
2020-11-06
'''
import fit_exp
import os
import tqdm
if __name__ == '__main__': 

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
        'check_data': True,
        'data_length_cut':200,
        'background_data': False,
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


    # if not os.path.exists(fit_dict['save_data_dir']):
    #     os.makedirs(fit_dict['save_data_dir'])
    curr_pos = fit_dict['current_pos']
    base = fit_dict['load_data_dir']
    for dir1 in tqdm.tqdm(os.scandir(base)):
        for dir2 in os.scandir(dir1.path):
            if dir2.path == dir1.path+str('/Background_data'):
                continue
            fit_dict['save_data_dir'] = dir2.path+str('/fitted_data')
            fit_dict['load_data_dir'] = dir2.path
            fit_dict['background_data_dir'] = dir1.path+str('/Background_data')
            fit_exp.fit_data(fit_dict)
            with open(fit_dict['save_data_dir'] + '/parameters.csv', 'w') as f:
                for key in fit_dict.keys():
                    f.write("%s,%s\n"%(key,fit_dict[key]))
                f.close()

            

