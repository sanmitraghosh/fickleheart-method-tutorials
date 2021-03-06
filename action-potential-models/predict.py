#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as m
import protocol

"""
Run prediction.
"""

model_list = ['tnnp-2004-w', 'fink-2008']
cal_list = ['stim1hz', 'stim2hz', 'randstim']
predict_list = ['stim1hz', 'stim2hz', 'randstim', 'hergblock']

try:
    which_model = sys.argv[1]
    which_cal = sys.argv[2]
    which_predict = sys.argv[3]
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__)
            + ' [str:which_calibration] [str:which_predict]')
    sys.exit()

if which_model not in model_list:
    raise ValueError('Input model %s is not available in the model list' \
            % which_model)

if which_cal not in cal_list:
    raise ValueError('Input calibration %s is not available in the data' \
            'list' % which_cal)

if which_predict not in predict_list:
    raise ValueError('Input data %s is not available in the predict list' \
            % which_predict)

data_dir = './data'

savedir = './fig/' + which_model
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = which_model + '-' + which_cal + '-' + which_predict

data_file_name = 'data-%s.csv' % which_predict
print('Predicting ', data_file_name)

# Load data
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1)  # headers
times = data[:, 0]
data = data[:, 1:]

# Protocol
stim_list = {
        'stim1hz': protocol.stim1hz,
        'stim2hz': protocol.stim2hz,
        'randstim': protocol.randstim,
        'hergblock': protocol.stim1hz_hergblock,
        }
stim_seq = stim_list[which_predict]

# Model
model = m.Model(
        './mmt-model-files/%s.mmt' % which_model,
        stim_seq=stim_seq,
        )
model.set_name(which_model)

# Calibrabted parameters
calloaddir = './out/' + which_model
load_seed = 542811797
fit_idx = [1, 2, 3]
param_file = '%s/%s-solution-%s-1.txt' % (calloaddir, which_cal, load_seed)
cal_param = np.loadtxt(param_file)

# Predict
if which_predict == 'hergblock':
    prediction = protocol.hergblock_simulate(model,
            cal_param, times)[[0, 3]]
    data = data.T[[0, 3]]
    legend = [' 0% block', ' 25% block', ' 50% block', ' 75% block',
            ' 100% block']
    legend = [legend[0], legend[3]]
else:
    prediction = [model.simulate(cal_param, times)]
    data = [data]
    legend = ['']

# Plot
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 4))
is_predict = 'Prediction' if which_cal != which_predict else 'Fitted model'
for i, (d, p) in enumerate(zip(data, prediction)):
    axes.plot(times, d, c='C' + str(i), alpha=0.5, label='Data' + legend[i])
    axes.plot(times, p, c='C' + str(i), ls='--',
            label=is_predict + legend[i])
axes.legend(loc=1)
axes.set_ylabel('Voltage (mV)')
axes.set_xlabel('Time (ms)')
plt.subplots_adjust(hspace=0)
plt.savefig('%s/%s.png' % (savedir, saveas),
        bbox_inches='tight')
plt.close()

