# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:38:20 2020

@author: Henrik aka Boss man, modified by Sebastian Nilsson
"""
import pandas as pd
import numpy as np
import mat73
import os
from tqdm import tqdm

def convert_matlab_to_csv(fit_dict):

    if not os.path.exists(fit_dict['save_data_dir']): # Creates directory if it does not exist
        os.mkdir(fit_dict['save_data_dir'])

    mat_data = mat73.loadmat(fit_dict['load_data_dir'])

    wave_data=mat_data['wave_data']

    wave_I=wave_data['intensity']
    intensity = pd.DataFrame()
    for i in tqdm(range(len(wave_I))):
        if np.any(wave_I[i,:]):
            temp_in = pd.DataFrame()
            wavei = np.array(wave_I[i,:])
            temp_in[str(i)] = wavei.reshape(wavei.size)
            intensity = pd.concat([intensity,temp_in],ignore_index=True,axis=1)
    if fit_dict['hdf'] == True: 
        intensity_name='intensity.h5' 
        intensity.to_hdf(os.path.join(fit_dict['save_data_dir'],intensity_name),key= 'intensity',mode = 'w')
    else:
        intensity_name='intensity.csv' 
        intensity.to_csv(os.path.join(fit_dict['save_data_dir'],intensity_name))
    
    if fit_dict['temperature_data'] == True:
        temperature_name='temperature_data.csv'
        temp_data=mat_data['temp_therm']
        temperature=pd.DataFrame(temp_data,columns=['Temp 1', 'Temp 2', 'Temp 3', 'Temp 4'])
        temperature=temperature[temperature['Temp 1']>0]

        if fit_dict['hdf'] == True: 
            temperature_name='temperature_data.h5'
            temperature.to_hdf(os.path.join(fit_dict['save_data_dir'],temperature_name),key= 'temperature',mode = 'w')
        else:
            temperature_name='temperature_data.csv'
            temperature.to_csv(os.path.join(fit_dict['save_data_dir'],temperature_name))
    
    if fit_dict['time_data'] == True:
        time_name='time.csv'
        time_data=wave_data['time']
        time_point=pd.DataFrame(time_data,columns=['time'])
        time_point=time_point[time_point['time']>0]

        if fit_dict['hdf'] == True: 
            time_name='time.h5'
            time_point.to_hdf(os.path.join(fit_dict['save_data_dir'],time_name),key= 'time_point',mode = 'w')
        else:
            time_name='time.csv'
            time_point.to_csv(os.path.join(fit_dict['save_data_dir'],time_name))
    
    if fit_dict['dt_data'] == True:
        dt=wave_data['dt']
        deltat_name='dt.csv'
        if dt.size == 1:
            rows,colums = wave_data['intensity'].shape
            dt = np.ones(rows)*dt
            deltat_temp = pd.DataFrame(dt,columns=['dt'])
            deltat=pd.DataFrame(columns=['dt'])
            rows, colums = intensity.shape
            deltat['dt']=deltat_temp['dt']
        else:
            deltat_temp = pd.DataFrame(dt,columns=['dt'])
            deltat=pd.DataFrame(columns=['dt'])
            rows, colums = intensity.shape
            deltat['dt']=deltat_temp['dt'][1:colums]

        if fit_dict['hdf'] == True: 
            deltat_name='dt.h5'
            deltat.to_hdf(os.path.join(fit_dict['save_data_dir'],deltat_name),key= 'deltat',mode = 'w')
        else:
            deltat_name='dt.csv'
            deltat.to_csv(os.path.join(fit_dict['save_data_dir'],deltat_name))

    if fit_dict['window_data'] == True:
        window_settings_name='window_settings.csv'
        window_settings=mat_data['window_settings']
        del window_settings['VDIV']
        del window_settings['OFST']
        del window_settings['TDIV']
        del window_settings['TRDL']
        windows=pd.DataFrame.from_dict(window_settings,'columns')
        real_window = pd.DataFrame()
        ofs = []
        tdiv = []
        trdl = []
        vdiv = []
        for i in tqdm(range(len(wave_I))):
            if np.any(wave_I[i]):
                ofs.append(np.array(windows['OFST_actual'][i]).item())
                tdiv.append(np.array(windows['TDIV_actual'][i]).item())
                trdl.append(np.array(windows['TRDL_actual'][i]).item())
                vdiv.append(np.array(windows['VDIV_actual'][i]).item())
        real_window['OFST_actual'] = np.array(ofs)
        real_window['TDIV_actual'] = np.array(tdiv)
        real_window['TRDL_actual'] = np.array(trdl)
        real_window['VDIV_actual'] = np.array(vdiv)
        real_window.to_csv(os.path.join(fit_dict['save_data_dir'],window_settings_name))

        if fit_dict['hdf'] == True: 
            window_settings_name='window_settings.h5'
            real_window.to_hdf(os.path.join(fit_dict['save_data_dir'],window_settings_name),key= 'real_window',mode = 'w')
        else:
            window_settings_name='window_settings.csv'
            real_window.to_csv(os.path.join(fit_dict['save_data_dir'],window_settings_name))
   
if __name__ == '__main__':

    fit_dict = {
        'load_data_dir': '/home/sebastian/OneDrive/Research/Decay_Signals/Calibration_data/Data/MFG_data/calibration_mV.mat',
        'temperature_data': False,
        'hdf': True,
        'window_data':False,
        'time_data': True,
        'dt_data': True,
        'save_data_dir':'/home/sebastian/OneDrive/Research/Decay_Signals/Calibration_data/Data/MFG_data'
    }
    ## Read Several files
    # for dir in os.walk(fit_dict['load_data_dir']):
    #     print(dir[0] )
    # base = fit_dict['load_data_dir']
    # for dir1 in os.scandir(base):
    #     for dir2 in os.scandir(dir1.path):
    #         fit_dict['save_data_dir'] = dir2.path
    #         fit_dict['load_data_dir'] = dir2.path+str('/data_kHz.mat')
    #         print('\n'+fit_dict['load_data_dir']+'\n')
    #         convert_matlab_to_csv(fit_dict)

    ## Read Singular file
    convert_matlab_to_csv(fit_dict)