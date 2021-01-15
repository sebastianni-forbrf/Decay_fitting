import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


fit_dict = {
    'load_data_dir': '/home/sebastian/OneDrive/Research/Decay_Signals/Calibration_data/Data/LED_Study'
}


base = fit_dict['load_data_dir']
for dir1 in os.scandir(base):
    for dir2 in os.scandir(dir1.path):
        if dir2.path == dir1.path+str('/Background_data'):
            continue
        fit_dict['save_data_dir'] = dir2.path+str('/fitted_data')
        fit_dict['load_data_dir'] = dir2.path