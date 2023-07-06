import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from scipy.io import loadmat
from plotly.offline import iplot
import plotly.graph_objs as go
from skimage import io

import mne
from scipy.interpolate import interp1d


import h5py
import numpy as np# Functions & Dataset
import pandas as pd
from scipy.stats import spearmanr

import time
from joblib import Parallel, delayed


# Loading and parsing each trial video game feature
def structure_gamelog(gamelog):

    trials_data = {}

    reading_navigation_positions = False

    for line in gamelog:


        if line.startswith('Trial ended'):
            trials_data[trial_n]['Trial_ended_time'] = float(line.split(' ')[-1]) 

            trial_duration = trials_data[trial_n]['Trial_ended_time'] - trials_data[trial_n]['Trial_start_time']
            trials_data[trial_n]['trial_duration'] = trial_duration
            trials_data[trial_n]['n_of_objects'] = len(trials_data[trial_n]['Objects_location'])

            trials_data[trial_n]['Navigation'] = np.array(trials_data[trial_n]['Navigation'])
            
            ### Compute velocity from position (navigation) coordinates
            trial_duration = trials_data[trial_n]['Navigation'][:,0][-1] - trials_data[trial_n]['Navigation'][:,0][0]
            # trial_distance in meters
            trial_distance = np.sum(np.abs(np.diff(trials_data[trial_n]['Navigation'][:,1] +1j* trials_data[trial_n]['Navigation'][:,2])))
            trials_data[trial_n]['Velocity'] = trial_distance/trial_duration
            
            
            trials_data[trial_n]['Objects_location'] = np.array(trials_data[trial_n]['Objects_location'])

            # trial_distance in angles
            trial_distance = np.unwrap( np.angle( trials_data[trial_n]['Navigation'][:,1] +1j* trials_data[trial_n]['Navigation'][:,2] ) )
            trial_distance = trial_distance[-1] - trial_distance[0]
            trials_data[trial_n]['Trial_distance'] = trial_distance

            ### cue/sec needs to get how many laps were performed and ajust it to the number of items per every lap (2pi)
            this_trial_crossed_objects = trial_distance  * trials_data[trial_n]['n_of_objects'] / np.pi*2
            trial_cue_per_sec = trials_data[trial_n]['trial_duration'] / this_trial_crossed_objects
            trials_data[trial_n]['trial_cue_per_sec'] = trial_cue_per_sec


            reading_navigation_positions = False


        if reading_navigation_positions==False:

            if line.startswith('Trial number'): 
                trial_n = int(line.split(': ')[1][:-1])
                trials_data[trial_n] = {}
            if line.startswith('Speed'): 
                trials_data[trial_n]['Speed'] = float(line.split('Speed: ')[1][:-1])
            if line.startswith('Ring_size'): 
                trials_data[trial_n]['Ring_size'] = float(line.split('Ring_size: ')[1][:-1])

            if line.startswith('Objects location'):
                ### actualItem  +" "+ itemX +" "+ itemZ +" "+ itemAngle
                trials_data[trial_n]['Objects_location'] = []
            if line.startswith('Item:'):
                trials_data[trial_n]['Objects_location'].append( np.array(line.split('Item: ')[1][:-1].split(' ')).astype(float) )

            if line.startswith('Trial start'):
                trials_data[trial_n]['Navigation'] = []
                reading_navigation_positions = True
                trials_data[trial_n]['Trial_start_time'] = float(line.split(' ')[-1]) 


            if line.startswith('Question onset'):
                trials_data[trial_n]['Testing'] = {}
                trials_data[trial_n]['Testing']['Question_onset'] = float(line.split('Question onset ')[1][:-1])

            if line.startswith('Testing'):            
                tmp_line = line[:-1].split(' ')
                trials_data[trial_n]['Testing']['response_time'] = float(tmp_line[1])
                trials_data[trial_n]['Testing']['cued_object'] = int(tmp_line[3])
                trials_data[trial_n]['Testing']['option_1'] = int(tmp_line[5])
                trials_data[trial_n]['Testing']['option_2'] = int(tmp_line[7])
                trials_data[trial_n]['Testing']['asnwered_object'] = int(tmp_line[9])
                trials_data[trial_n]['Testing']['asnwered_key'] = tmp_line[11]

        elif reading_navigation_positions==True:
            trials_data[trial_n]['Navigation'].append( np.array(line[:-1].split(' ')).astype(float) )

    return trials_data


def interp_f(behavioural_signal, lfp_size):
    n_samples = len(behavioural_signal)
    x = np.linspace(0, n_samples, num=n_samples, endpoint=True)
    interpolate_f = interp1d(x, behavioural_signal, kind='linear')
    new_sampling_space = np.linspace(0, n_samples, num=lfp_size, endpoint=True)
    return interpolate_f(new_sampling_space)


def get_behavioral_position(trials_data, trial_number):
    position_array = []
    for count in range(len(trials_data[trial_number]['Navigation'])):
        time = trials_data[trial_number]['Navigation'][count][0]
        x = trials_data[trial_number]['Navigation'][count][1]
        y = trials_data[trial_number]['Navigation'][count][2]
        position_array.append([trial_number, time, x, y])
        
    return np.array(position_array)


'''
Extract navigation data from recording txt file
Return:
    data selected from hippocampol channel, 
    lfp_theta_filtered, applied norch and theta filter on raw data
    trials_data, extract from game log, see def struct_gamelog
    event_samples, TTL information and index for navigation start, end...etc
    int(raw.info['sfreq']) s_frequency info
'''
def get_subject_data(sub):

    # Load brain recordings
    raw = mne.io.read_raw_edf(file_paths[sub]['recordings'])
    print("all channel name:", raw.ch_names)
    #print(np.shape(raw))
    # Load TTL events
    TTL = pd.read_csv(file_paths[sub]['ttl'],
                      sep=' ',
                      names=['time', 'type', 'n'])
    #print(TTL)

    # Get hippocampal channels of this subject
    sub_hpc_chans = hpc_chan2[sub]
    data = raw.get_data(picks=sub_hpc_chans)

    # Load videoGame data and structure it by trials
    gamelog = []
    with open(file_paths[sub]['game'], 'r') as f:
        lines = f.readlines()
        for line in lines:
            gamelog.append(line)
    trials_data = structure_gamelog(gamelog)

    ### Load the signal
    # Here loading 2 signals in the Amygdala. This will be specific for each patient.
    lfps = raw.get_data(picks=sub_hpc_chans)

    ## Notch is done to remove the noise from the power line in the building. Since China operates on a 220V voltage and 50Hz and remove the 50Hz and its harmonics (100, 150, etc) to the signal.
    lfp_bp_notch = mne.filter.notch_filter(lfps,
                                           raw.info['sfreq'],
                                           [50., 100., 150., 200.],
                                           notch_widths=.1)
#   lfp_bp_notch = lfp_bp_notch[0] - lfp_bp_notch[1]
    low_freq = 2
    high_freq = 10

    lfp_theta_filtered = mne.filter.filter_data(lfp_bp_notch, int(raw.info['sfreq']), low_freq, high_freq, verbose=False )

    ### TTL markers
    event_samples = np.vstack((
        TTL[TTL['type'] == 2]['time'].values,
        TTL[TTL['type'] == 3]['time'].values,
        TTL[TTL['type'] == 4]['time'].values,
        TTL[TTL['type'] == 5]['time'].values,
    )).T

    return data, lfp_theta_filtered, trials_data, event_samples, int(raw.info['sfreq'])


'''
Deal with ieeg signal data
Return: 
a dict with essential information
physilogical data: a list of the ieeg signal between navigation for all trials
timestamps: a list of index matching physilogical data
position_2d: coordinate..ish matching each timestamp
event_marker: 
'''
def get_preprocess_data(trials_data,markers,physilogical_data,start_end_indice=[0,1],test=False): #trials, event_sample, notch_filtered_data
    
    '''
    start_end_indice:
        markers[:][0] the start of one trial
        markers[:][1] the end of one trial
        markers[:][2] the onset of the testing question
        markers[:][3] the onset of response
    test: 
        True: decode the recall part, no real output(position)
    '''
    physilogical_data_navi= {}

    for chan in range(physilogical_data.shape[0]):
        
        for trial_num in range(len(trials_data)):  
            behavioural_signal = get_behavioral_position(trials_data, trial_num)
            navi_start = int(markers[trial_num][start_end_indice[0]])
            navi_stop = int(markers[trial_num][start_end_indice[1]])
            lfp_size = abs(navi_stop - navi_start)
            print("lfp_size: ", lfp_size)
            
            if (chan == 0) & (trial_num == 0):
                start_array = np.array([lfp_size])
                timestamps = np.linspace(navi_start,navi_stop-1,lfp_size, dtype=int)
                if test == False :
                    position_2d = np.vstack(( interp_f(behavioural_signal.T[2], lfp_size) , interp_f(behavioural_signal.T[3], lfp_size)))
                
            if (chan == 0) & (trial_num > 0):
                start_array = np.hstack((start_array,np.array([int(start_array[trial_num-1])+lfp_size])))
                timestamps = np.hstack(( timestamps, np.linspace(navi_start,navi_stop-1,lfp_size, dtype=int) ))
                if test == False :
                    position_2d = np.hstack((position_2d, np.vstack(( interp_f(behavioural_signal.T[2], lfp_size) , interp_f(behavioural_signal.T[3], lfp_size)))  ))
                
            if trial_num == 0:
                physilogical_data_navi[chan]= physilogical_data[chan][navi_start:navi_stop]
                
            if trial_num > 0:
                physilogical_data_navi[chan]  = np.hstack((physilogical_data_navi[chan], physilogical_data[chan][navi_start:navi_stop] ))
    
    print(np.shape(timestamps))
    print(np.shape(start_array))
    print(np.shape(np.array([physilogical_data_navi[i] for i in range(len(physilogical_data_navi))]).T))
    if test == True :
        return {
            'physilogical_data': np.array([physilogical_data_navi[i] for i in range(len(physilogical_data_navi))]).T,
            'timestamps': timestamps,
            'event_marker_each_trial': start_array
        }
    if test == False :               
        return {
            'physilogical_data': np.array([physilogical_data_navi[i] for i in range(len(physilogical_data_navi))]).T,
            'timestamps': timestamps,
            'position_2d': position_2d,
            'event_marker_each_trial': start_array
        }