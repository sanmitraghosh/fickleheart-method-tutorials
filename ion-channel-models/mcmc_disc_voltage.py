#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('./method')
import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints
import pints.io
import pints.plot
import pymc3 as pm
import random 
import model as m
import parametertransform
import priors
from priors import HalfNormalLogPrior, InverseGammaLogPrior
from sparse_gp_custom_likelihood import DiscrepancyLogLikelihood, _create_theano_conditional_graph_voltage
"""
Run fit.
"""
print('Using PyMC3 version: ',str(pm.__version__))
model_list = ['A', 'B', 'C']

try:
    which_model = 'A'#sys.argv[1] 
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__))
    sys.exit()

if which_model not in model_list:
    raise ValueError('Input model %s is not available in the model list' \
            % which_model)

# Get all input variables
import importlib
sys.path.append('./mmt-model-files')
info_id = 'model_%s' % which_model
info = importlib.import_module(info_id)

data_dir = './data'

savedir = './out/mcmc-' + info_id
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_file_name = 'data-sinewave.csv'
print('Fitting to ', data_file_name)
print('Temperature: ', info.temperature)
saveas = info_id + '-' + data_file_name[5:][:-4]

# Protocol
protocol = np.loadtxt('./protocol-time-series/sinewave.csv', skiprows=1,
        delimiter=',')
protocol_times = protocol[:, 0]
protocol = protocol[:, 1]

# Control fitting seed
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)

np.random.seed(fit_seed)

# Set parameter transformation
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

# Load data
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1)  # headers
times = data[:, 0]
data = data[:, 1]
voltage = protocol#<-----New
noise_sigma = np.log(np.std(data[:500]))

print('Estimated noise level: ', noise_sigma)

model = m.Model(info.model_file,
        variables=info.parameters,
        current_readout=info.current_list,
        set_ion=info.ions_conc,
        transform=transform_to_model_param,
        temperature=273.15 + info.temperature,  # K
        )

LogPrior = {
        'model_A': priors.ModelALogPrior,
        'model_B': priors.ModelBLogPrior,
        }

# Update protocol
model.set_fixed_form_voltage_protocol(protocol, protocol_times)

# Create Pints stuffs
inducing_times = times[::1000] 
inducing_voltage = voltage[::1000] #<-----New
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = DiscrepancyLogLikelihood(problem, inducing_times, voltage=voltage, inducing_voltage=inducing_voltage, downsample=None) #<-----New
logmodelprior = LogPrior[info_id](transform_to_model_param,
        transform_from_model_param)


lognoiseprior = HalfNormalLogPrior(sd=25,transform=True) 
logrhoprior1 = InverseGammaLogPrior(alpha=5,beta=5,transform=True) #<-----New
logrhoprior2 = InverseGammaLogPrior(alpha=5,beta=5,transform=True) #<-----New
logrhoprior = pints.ComposedLogPrior(logrhoprior1, logrhoprior2) #<-----New
logkersdprior = InverseGammaLogPrior(alpha=5,beta=5,transform=True) 

 
logprior = pints.ComposedLogPrior(logmodelprior, lognoiseprior, logrhoprior, logkersdprior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
initial_rho = np.log([0.5,0.5]) #<-----New
initial_ker_sigma = np.log(5.0) 

priorparams = np.copy(info.base_param)
transform_priorparams = transform_from_model_param(priorparams)
priorparams = np.hstack((priorparams, np.exp(noise_sigma), np.exp(initial_rho), np.exp(initial_ker_sigma)))
transform_priorparams = np.hstack((transform_priorparams, noise_sigma, initial_rho, initial_ker_sigma))
print('Posterior at prior parameters: ',
        logposterior(transform_priorparams))
for _ in range(10):
    assert(logposterior(transform_priorparams) ==\
            logposterior(transform_priorparams))

# Load fitting results
calloaddir = './out/' + info_id
load_seed = 542811797
fit_idx = [1, 2, 3]
transform_x0_list = []

print('MCMC starting point: ')
for i in fit_idx:
    f = '%s/%s-solution-%s-%s.txt' % (calloaddir, 'sinewave', load_seed, i)
    p = np.loadtxt(f)
    transform_x0_list.append(np.hstack((transform_from_model_param(p),
            noise_sigma, initial_rho, initial_ker_sigma)))
    print(transform_x0_list[-1])
    print('Posterior: ', logposterior(transform_x0_list[-1]))

# Run
mcmc = pints.MCMCController(logposterior, len(transform_x0_list),
        transform_x0_list, method=pints.AdaptiveCovarianceMCMC)
n_iter = 100
mcmc.set_max_iterations(n_iter)
mcmc.set_initial_phase_iterations(int(200)) # Note for Chon: Only use 100/200 iterations maximum for random walk and then switch to Adaptive
mcmc.set_parallel(True)
mcmc.set_chain_filename('%s/%s-chain.csv' % (savedir, saveas))
mcmc.set_log_pdf_filename('%s/%s-pdf.csv' % (savedir, saveas))
chains = mcmc.run()

# De-transform parameters
chains_param = np.zeros(chains.shape)
for i, c in enumerate(chains):
    c_tmp = np.copy(c)
    chains_param[i, :, :-4] = transform_to_model_param(c_tmp[:, :-4]) #<-----New 
    chains_param[i, :, -4:] = np.exp((c_tmp[:, -4:])) #<-----New
    
    del(c_tmp)

# Save (de-transformed version)
pints.io.save_samples('%s/%s-chain.csv' % (savedir, saveas), *chains_param)

# Plot
# burn in and thinning
chains_final = chains[:, int(0.5 * n_iter)::1, :]
chains_param = chains_param[:, int(0.5 * n_iter)::1, :]

transform_x0 = transform_x0_list[0]
x0 = np.append(transform_to_model_param(transform_x0[:-4]), np.exp(transform_x0[-4:]))#<-----New

pints.plot.pairwise(chains_param[0], kde=False, ref_parameters=x0)
plt.savefig('%s/%s-fig1.png' % (savedir, saveas))
plt.close('all')

pints.plot.trace(chains_param, ref_parameters=x0)
plt.savefig('%s/%s-fig2.png' % (savedir, saveas))
plt.close('all')


import scipy.stats as stats

ppc_samples = chains_param[0]
gp_ppc_mean =[]
gp_ppc_var = []

gp_ppc_sim = []
training_data = data.reshape((-1,))
t_training_protocol = times.reshape((-1,1)) 
v_training_protocol = voltage.reshape((-1,1)) #<-----New
x_training_protocol = np.concatenate((t_training_protocol, v_training_protocol),axis=1)#<-----New
ind_t = inducing_times.reshape((-1,1))
ind_v = inducing_voltage.reshape((-1,1))#<-----New
ind_x = np.concatenate((ind_t, ind_v),axis=1)#<-----New

valid_times = times  
valid_voltage = voltage  #<-----New, need to pass here the AP protocol
t_valid_protocol = valid_times.reshape((-1,1)) 
v_valid_protocol = valid_voltage.reshape((-1,1)) 
x_valid_protocol = np.concatenate((t_valid_protocol, v_valid_protocol),axis=1)

ppc_sampler_mean, ppc_sampler_var = _create_theano_conditional_graph_voltage(training_data, x_training_protocol, ind_x, x_valid_protocol)
nds = 4 #<-----New
for ind in random.sample(range(0, np.size(ppc_samples, axis=0)), 40):
        ode_params = transform_from_model_param(ppc_samples[ind, :-nds]) 
        _sigma, _rho1, _rho2, _ker_sigma = ppc_samples[ind,-nds:] #<-----New
        _rho = np.append(_rho1, _rho2)#<-----New
        current_training_protocol = model.simulate(ode_params, times)
        current_valid_protocol = model.simulate(ode_params, valid_times)
        ppc_mean = ppc_sampler_mean(current_training_protocol,current_valid_protocol,_rho,_ker_sigma,_sigma)
        ppc_var = ppc_sampler_var(current_training_protocol,current_valid_protocol,_rho,_ker_sigma,_sigma)
        gp_ppc_mean.append(ppc_mean)
        gp_ppc_var.append(ppc_var)

gp_ppc_mean = np.array(gp_ppc_mean)
gp_ppc_var = np.array(gp_ppc_var)
ppc_mean = np.mean(gp_ppc_mean, axis=0)
var1, var2, var3 = np.mean(gp_ppc_var, axis=0), np.mean(np.power(gp_ppc_mean,2), axis=0), np.power(np.mean(gp_ppc_mean, axis=0),2)
ppc_sd = np.sqrt(var1 + var2 - var3)


plt.figure(figsize=(8, 6))
plt.plot(times, data, label='Model C')
plt.plot(times, ppc_mean, label='Mean')

plt.plot(times, ppc_mean + 2*ppc_sd, '-', color='blue',
             lw=0.5, label='conf_int')
plt.plot(times, ppc_mean - 2*ppc_sd, '-', color='blue',
             lw=0.5)

plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Current (pA)')
plt.show()