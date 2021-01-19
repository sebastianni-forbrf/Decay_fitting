# Decay fitting
The scripts in this GitHub is used to analyse decay data from phosphor data.
In the example data folder there is a data_kHz.mat file that can be used as a test file which the read_mat_to_csv.py converts to readble file for the main_decay_fit.py.

# Example usage.
## 1. 
Use the read_mat_to_csv.py to convert the calibartion_mV.mat files to readable
csv or HDF files. Make sure you specify what type of data you have accquired in the
calibratiom_mV.mat file, see the dictonary in the read_mat_to_csv.py.
Please note: if you have multiple files you like to convert you have to write a 
small piece of code that loops through the desired directorys.

## 2.
Processing the data is done by using the main_decay_fit.py.
Locate the directory where you saved the files from read_mat_to_csv.py
Then open the main_decay_fit.py, in the fit_dict dictionary add the directory to "load_data_dir"
You can change several settings but the most important ones are under " Fixed window fitting paramters, both for tau method and amplitude method".
In the post process dictionary section you can choose to do post processing and also choose which folder to save the fitted data.
Please note:  If you have several directorys you have to write a small piece of code that loops through the desired directorys.

## 3. 
The fitted data will be in a folder called "fitted_data" and all the relevant information from fitting will be saved in a csv file called either "amplitude_method_data.csv"
or "tau_method_data.csv". 
