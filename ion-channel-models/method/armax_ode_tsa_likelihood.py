import os
import pints
import numpy as np


class DiscrepancyLogLikelihood(pints.ProblemLogLikelihood):
    """
    This class defines a custom loglikelihood which implements a
    discrepancy model where the noise follows an ARMA(p,q) process
    """

    def __init__(self, problem, armax_result_obj, transparams = False, temperature=None): ################# <-----Changed 21/10 #####################
        super(DiscrepancyLogLikelihood, self).__init__(problem)

        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._armax_result = armax_result_obj
        self._nds = len(self._armax_result.params) - 1
        self._n_parameters = self._np + self._nds
        self._nt = problem.n_times() 
        self._temperature = temperature   
        self._transparams = transparams  ################# <-----Changed 21/10 #####################

    def __call__(self, x):
        
        armax_params = x[-self._nds:]
        armax_params = np.append(1.0,armax_params)
        model_params = x[:-self._nds]
        ion_current = self._problem.evaluate(model_params) 
        
        self._armax_result.model.transparams = self._transparams ################# <-----Changed 21/10 #####################
        self._armax_result.model.exog = ion_current[:,None]
        
        if self._temperature==None:
            ll = self._armax_result.model.loglike_kalman(armax_params)
            assert np.sum(self._armax_result.model.exog) == np.sum(ion_current)       
            return ll
        else:
            return self._temperature*self._armax_result.model.loglike_kalman(armax_params)
    