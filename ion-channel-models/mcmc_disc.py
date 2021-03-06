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
from sparse_gp_custom_likelihood import DiscrepancyLogLikelihood, _create_theano_conditional_graph
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
noise_sigma = np.log(np.std(data[:500]))
np.savetxt('../../time.txt',times)
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
inducing_times = times[::1000] #Note to Chon: These are the inducing or speudo training points for the FITC GP
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = DiscrepancyLogLikelihood(problem, inducing_times, downsample=None) #Note to Chon: downsample=100<--change this to 1 or don't use this option
logmodelprior = LogPrior[info_id](transform_to_model_param,
        transform_from_model_param)

# Priors for discrepancy
# I have transformed all the discrepancy priors as well
lognoiseprior = HalfNormalLogPrior(sd=25,transform=True) # This will have considerable mass at the initial value
logrhoprior = InverseGammaLogPrior(alpha=5,beta=5,transform=True) # As suggested in STAN manual
logkersdprior = InverseGammaLogPrior(alpha=5,beta=5,transform=True) # As suggested in STAN manual

 
logprior = pints.ComposedLogPrior(logmodelprior, lognoiseprior, logrhoprior, logkersdprior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Check logposterior is working fine
initial_rho = np.log(0.5) # Initialise Kernel hyperparameter \rho
initial_ker_sigma = np.log(5.0) # Initialise Kernel hyperparameter \ker_sigma

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
    chains_param[i, :, :-3] = transform_to_model_param(c_tmp[:, :-3]) # First the model ones 
    chains_param[i, :, -3:] = np.exp((c_tmp[:, -3:])) # Then the discrepancy ones
    
    del(c_tmp)

# Save (de-transformed version)
pints.io.save_samples('%s/%s-chain.csv' % (savedir, saveas), *chains_param)

# Plot
# burn in and thinning
chains_final = chains[:, int(0.5 * n_iter)::1, :]
chains_param = chains_param[:, int(0.5 * n_iter)::1, :]

transform_x0 = transform_x0_list[0]
x0 = np.append(transform_to_model_param(transform_x0[:-3]), np.exp(transform_x0[-3:]))

pints.plot.pairwise(chains_param[0], kde=False, ref_parameters=x0)
plt.savefig('%s/%s-fig1.png' % (savedir, saveas))
plt.close('all')

pints.plot.trace(chains_param, ref_parameters=x0)
plt.savefig('%s/%s-fig2.png' % (savedir, saveas))
plt.close('all')

##############  GP DISCREPANCY PREDICTION FOR CHON #########################################################
# Posterior predictive in light of the discontinuity GP. Like the ARMAX case we use the variance identity here.
# Currently this is setup to use the same protocol for training and validation. Please change this accordingly.
# How this works: Basically we want \int GP(Current_validation|Current_training, Data, gp_params, ode_params) d gp_params d ode_params .
# We obtain this--->GP(Current_validation|Current_training, Data, gp_params, ode_params) as  Normal distribution, see ppc_mean, ppc_var, 
# for a single sample of (gp_params, ode_params). To propagate th uncertainty fully I then use the same Variance identity that I used for ARMAX
# to integrate out (gp_params, ode_params).
import scipy.stats as stats

ppc_samples = chains_param[0]
gp_ppc_mean =[]
gp_ppc_var = []

gp_ppc_sim = []
gp_rmse = []
training_data = data.reshape((-1,))
t_training_protocol = times.reshape((-1,1)) 
ind_t = inducing_times.reshape((-1,1))
valid_times = times  #### <---Create this variable accordingly, now I am using the training protocol time
t_valid_protocol = valid_times.reshape((-1,1)) 
ppc_sampler_mean, ppc_sampler_var = _create_theano_conditional_graph(training_data, t_training_protocol, ind_t, t_valid_protocol)
nds = 3 ### Number of discrepancy params
for ind in random.sample(range(0, np.size(ppc_samples, axis=0)), 40):
        ode_params = transform_from_model_param(ppc_samples[ind, :-nds]) ### Expecting these parameters have been "transformed to model"
        _sigma, _rho, _ker_sigma = ppc_samples[ind,-nds:] ### Expecting these to Untransformed by exponentiation earlier"
        current_training_protocol = model.simulate(ode_params, times)
        current_valid_protocol = model.simulate(ode_params, valid_times)
        ppc_mean = ppc_sampler_mean(current_training_protocol,current_valid_protocol,_rho,_ker_sigma,_sigma)
        ppc_var = ppc_sampler_var(current_training_protocol,current_valid_protocol,_rho,_ker_sigma,_sigma)
        gp_ppc_mean.append(ppc_mean)
        gp_ppc_var.append(ppc_var)
        #### This bits are for E[rmse] calculation ###
        ### I am assuming you have a function rmse(data, prediction)
        ppc_sample_given_all_params = stats.norm(ppc_mean,np.sqrt(ppc_var)).rvs()
        gp_rmse.append(rmse(current_training_protocol,ppc_sample_given_all_params))


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

### Calculate rmse ###
gp_rmse = np.array(gp_rmse)
expected_gp_rmse = np.mean(gp_rmse, axis=0)
print("The calculated RMSE is: ", expected_gp_rmse)