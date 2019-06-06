#!/usr/bin/env python2
from __future__ import print_function
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pints
import myokit
import myokit.pacing as pacing

from default import *

vhold = -80  # mV
DT = 0.2

#
# Defining AP simulation
#

def simulate_aps(scales, model_file, beats=2, cl=1000, prepace=100,
        stimulate=True):
    """
        Generate APs using the given scalings.
    """
    # Load model
    model = myokit.load_model(model_file)

    # Apply scalings
    for var in scales:
        scale = scales[var]
        v = model.get(var)
        v.set_rhs(myokit.Multiply(myokit.Number(scale), v.rhs()))

    # Simulate with modified model
    sim = myokit.Simulation(model)

    # Add stimulus
    if stimulate:
        protocol = myokit.pacing.blocktrain(period=cl, duration=1,
                offset=50)
        sim.set_protocol(protocol)
    
    # Pre-pace for some beats
    sim.pre(prepace * cl)

    # Log some beats and return
    log = ['engine.time', 'membrane.V']
    log = ['environment.time', 'membrane.V']
    log = ['membrane.V']
    return sim.run(beats * cl, log=log, log_interval=DT).npview()


#
# Time out handler
#
class Timeout(myokit.ProgressReporter):
    """
    A :class:`myokit.ProgressReporter` that halts the simulation after
    ``max_time`` seconds.
    """
    def __init__(self, max_time):
        self.max_time = float(max_time)
    def enter(self, msg=None):
        self.b = myokit.Benchmarker()
    def exit(self):
        pass
    def update(self, progress):
        return self.b.time() < self.max_time


#
# Create ForwardModel
#

class Model(pints.ForwardModel):
    parameters = [
        'ina.s', 'ical.s', 'ikr.s', 'iks.s', 'ito.s', 'inaca.s', 'ik1.s',
        'inak.s', #'if.s', 
        ]
    
    def __init__(self, model_file, prepace=10, stimulate=True, cl=None,
            stim_dur=None, stim_offset=None, stim_amp=None, transform=None,
            max_evaluation_time=5, norm=False):
        """
        # model_file: mmt model file for myokit; main units: mV, ms, pA.
        # prepace: number of pre-pace before recording the simulated AP.
        # stimulate: bool, if True, apply stimulus.
        # cl: cycle length of stimulus.
        # stim_dur: stimulus duration.
        # stim_offset: stimulus offset (after each pace).
        # stim_amp: stimulus (normalised) amplitude (based on the model
        #           default stimulus amplitude).
        # transform: transform search space parameters to model parameters.
        # max_evaluation_time: maximum time (in second) allowed for one
        #                      simulate() call.
        # norm: bool, if True, normalise output.
        """
        self._model = myokit.load_model(model_file)
        self._model_file = model_file
        self._model_file_name = os.path.basename(model_file)
        print('Initialising model %s...' % self._model_file_name)
        self._prepace = prepace
        self._stimulate = stimulate
        self.transform = transform
        self.simulation = myokit.Simulation(self._model)
        # Add stimulus
        if self._stimulate:
            if stim_dur is None:
                stim_dur = model_stim_setup[self._model_file_name][0]
            if stim_offset is None:
                stim_offset = model_stim_setup[self._model_file_name][1]
            if cl is None:
                cl = model_stim_setup[self._model_file_name][2]
            if stim_amp is None:
                stim_amp = model_stim_setup[self._model_file_name][3]
            self._cl = cl
            protocol = myokit.pacing.blocktrain(period=self._cl, 
                                                duration=stim_dur, 
                                                offset=stim_offset,
                                                level=stim_amp)
            self.simulation.set_protocol(protocol)
        else:
            if cl is None:
                self._cl = 1000
            else:
                self._cl = cl
        # Create a order-matched conductance list
        try:
            self._conductance = []
            p2g = model_conductance[self._model_file_name]
            for p in self.parameters:
                self._conductance.append(p2g[p])
            assert(len(self._conductance) == len(self.parameters))
            del(p, p2g)
        except:
            raise ValueError('Model conductances do not match parameters')
        # Get original parameters
        try:
            self.original = []
            for name in self._conductance:
                v = self._model.get(name)
                self.original.append(np.float(v.rhs()))
            assert(len(self.original) == len(self.parameters))
        except:
            raise ValueError('Model conductances do not exist in the given ' \
                    + 'model')
        # Set stimulus
        if self._stimulate:
            try:
                s = model_stim_amp[self._model_file_name]
                self.simulation.set_constant(s[0], s[1])
            except:
                raise ValueError('Model stimulus do not exist in the given ' \
                        + 'model')
        # Store model original state -- can be something else later!
        self.original_state = self.simulation.state()
        # if normalise
        self.norm = norm
        # maximum time allowed
        self.max_evaluation_time = max_evaluation_time
        print('Done')
    
    def dimension(self):
        return len(self.parameters)

    def simulate(self, parameter, times):
        """
        Generate APs using the given scalings.
        """
        parameter = np.array(parameter)
        # Update model parameters
        if self.transform is not None:
            parameters = self.transform(parameters)
        # Simulate with modified model
        for i, name in enumerate(self._conductance):
            self.simulation.set_constant(name, 
                                         parameter[i] * self.original[i])
        # Run
        self.simulation.reset()
        # As myokit.org specified, in AP simulation mode, simulation.pre()
        # sorts the end of simulation state as the new default state, so
        # simulation.reset() only reset to the 'new' default state. It need
        # a manual reset of the state using simulation.set_state() to the 
        # originally stored state.
        self.simulation.set_state(self.original_state)
        try:
            # Pre-pace for some beats
            self.simulation.pre(self._prepace * self._cl)
            # Log some beats
            d = self.simulation.run(np.max(times)+0.02, 
                log_times = times, 
                log = [
                       'membrane.V',
                      ],
                ).npview()
        except myokit.SimulationError:
            return float('inf')
        if self.norm:
            return self.normalise(d['membrane.V'])
        else:
            return d['membrane.V']

    def current(self, parameter, voltage, times):
        """
        Generate current of voltage clamp using the given scalings.
        - Not fitting on this, so voltage resolution is not as important.
        - Not running this often, so can setup everything here...
        """
        parameter = np.array(parameter)
        # Update model parameters
        if self.transform is not None:
            parameters = self.transform(parameters)
        model = myokit.load_model(self._model_file)
        # Simulate with modified model
        for i, name in enumerate(self._conductance):
            '''
            # normal way of doing it...
            self.simulation.set_constant(name, 
                                         parameter[i] * self.original[i])
            '''
            # try to set conductance for non-literal...
            model.get(name).set_rhs(
                parameter[i] * self.original[i]
            #'''
            )
        # Get current names of output
        current = []
        m_cur = model_current[self._model_file_name]
        for name in self.parameters:
            current.append(m_cur[name])
        # Set up voltage clamp
        #for ion_var, ion_conc in model_ion[self._model_file_name]:
        #    self._fix_concentration(model, ion_var, ion_conc)
        # Detach voltage for voltage clamp(?)
        model_v = model.get('membrane.V')
        model_v.demote()
        tmp_vhold = vhold
        model_v.set_rhs(tmp_vhold)
        model.get('engine.pace').set_binding(None)
        model_v.set_binding('pace')
        
        # Create pre-pacing protocol
        
        protocol = pacing.constant(tmp_vhold)
        # Create pre-pacing simulation
        simulation1 = myokit.Simulation(model, protocol)
        simulation2 = myokit.Simulation(model)
        simulation2.set_fixed_form_protocol(
            times, 
            voltage
        )
        simulation2.set_tolerance(1e-8, 1e-10)
        simulation2.set_max_step_size(1e-2)  # ms
        # Run
        simulation1.reset()
        simulation2.reset()
        simulation1.set_state(self.original_state)
        simulation1.pre(100)
        simulation2.set_state(simulation1.state())
        # Log some beats
        d = simulation2.run(np.max(times)+0.02, 
            log_times = times, 
            log = current,
            ).npview()
        # rename output names
        d_out = myokit.DataLog()
        for s, c in m_cur.iteritems():
            if s in self.parameters:
                d_out[s[:-2]] = d[c]
        del(d)
        return d_out

    def _fix_concentration(self, model, variable, concentration):
        v = model.get(variable)
        if v.is_state():
            v.demote()
        v.set_rhs(concentration)
    
    def parameter(self):
        # return the name of the parameters
        return self.parameters

    def name(self):
        # name
        return self._name
    
    def set_name(self, name):
        # set name
        self._name = name

    def separate_current(self, parameter, times):
        parameter = np.array(parameter)
        # Update model parameters
        if self.transform is not None:
            parameters = self.transform(parameters)
        return self.currents * parameter[self.ss]

    def normalise(self, v):
        # Do whatever fancy normalisation here
        # For example to mimic optical mapping data
        method = 1
        if method == 1:
            # 5, 95 percentiles
            minimum = np.percentile(v, 5)
            maximum = np.percentile(v, 95)
            return (v - minimum) / (maximum - minimum)
        elif method == 2:
            # RMSD normalisation
            return v / np.sqrt(np.mean(v ** 2))
        elif method == 3:
            # Use minimisation to fit the two curve
            # Actually it should not be here but in
            # the error function...
            raise NotImplementedError
