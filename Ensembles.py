import numpy as np
from SynapticDynamics import FiringMask


class Ensemble(object):
    def __init__(self, simenv, num_neurons, **ensemble_params):
        """

        Args:
            simenv (Environments.Simulation):
            num_neurons (int):
            **ensemble_params (dict):
                u_rest (numpy.darray): (num_neurons, )
                tau_m (numpy.darray): (num_neurons, )
                u_threshold (numpy.darray): (num_neurons, )
                u_reset (numpy.darray): (num_neurons, )
                g_excite_const (numpy.darray): (num_neurons, num_neurons)
                tau_excite (numpy.darray): (num_neurons, num_neurons)
                g_inhib_const (numpy.darray): (num_neurons, num_neurons)
                tau_inhib (numpy.darray): (num_neurons, num_neurons)
                E_excite (numpy.darray): (num_neurons, num_neurons)
                E_inhib (numpy.darray): (num_neurons, num_neurons)
                E_run (float):
                E_theta (float)
                weights_excite (numpy.darray): (num_neurons, num_neurons)
                weights_inhib (numpy.darray): (num_neurons, num_neurons)
        """
        self.num_neurons = num_neurons
        self.simenv = simenv
        self.ensemble_params = ensemble_params

        # Initialize ensemble
        self._initialise_ensembles_params()

        # Initialize synaptic parameters
        self.g_excite, self.g_inhib = np.zeros((num_neurons, num_neurons)), np.zeros((num_neurons, num_neurons))
        self._initialise_synaptic_current_params()
        self.synaptic_currents = np.zeros((num_neurons,))  # spike-input related

        # Firing
        self.firing_mask = FiringMask(num_neurons)

        # External input parameters
        self.E_run = ensemble_params["E_run"]
        self.E_theta = ensemble_params["E_theta"]

        # Weights
        self.weights_excite = ensemble_params['weights_excite']
        self.weights_inhib = ensemble_params['weights_inhib']


    def state_update(self):
        self._membrane_potential_dyanmics_update()
        self._threshold_crossing()  # Register which neurons fire and reset potentials
        # if np.sum(self.firing_mask.get_mask()) > 0:
        #     import pdb
        #     pdb.set_trace()
        self._calc_synaptic_current()
        self._syn_current_dynamics_update()
        self.simenv.increment()
        return self.simenv.check_end()

    def _membrane_potential_dyanmics_update(self):
        inputs_dict = self.simenv.get_inputs()
        du_dt = (self.u_rest-self.u)/self.tau_m \
                + self.synaptic_currents \
                + (self.E_run - self.u) * inputs_dict["run"] \
                + (self.E_theta - self.u) * inputs_dict["theta"]

        self.u += du_dt * self.simenv.get_dt()

    def _threshold_crossing(self):
        self.firing_mask.update_mask(self.u, self.u_threshold)
        self.u[self.firing_mask.get_mask()] = self.u_reset[self.firing_mask.get_mask()]

    def _calc_synaptic_current(self):

        u_2d = np.repeat(self.u.reshape(1, -1), self.num_neurons, axis=0)  # Expand row-wise
        current_2d_excite = self.g_excite * (self.E_excite - u_2d) * self.weights_excite
        current_2d_inhib = self.g_inhib * (self.E_inhib - u_2d) * self.weights_inhib
        current_1d = np.sum(current_2d_excite, axis=1) + np.sum(current_2d_inhib, axis=1)
        self.synaptic_currents = current_1d

    def _syn_current_dynamics_update(self):
        dg_excite = -self.g_excite/(self.tau_excite+1e-9) + self.g_excite_const * self.firing_mask.get_2d_rows()
        dg_inhib = -self.g_inhib/(self.tau_inhib+1e-9) + self.g_inhib_const * self.firing_mask.get_2d_rows()
        self.g_excite += dg_excite * self.simenv.get_dt()
        self.g_inhib += dg_inhib * self.simenv.get_dt()

    def _initialise_ensembles_params(self):
        self.u_rest, self.tau_m, self.u_threshold, self.u_reset = self.ensemble_params["u_rest"],\
                                                                  self.ensemble_params["tau_m"],\
                                                                  self.ensemble_params["u_threshold"],\
                                                                  self.ensemble_params["u_reset"]
        self.u = self.u_rest.copy()

    def _initialise_synaptic_current_params(self):
        self.g_excite_const, self.tau_excite, self.g_inhib_const, self.tau_inhib =  self.ensemble_params["g_excite_const"],\
                                                                        self.ensemble_params["tau_excite"],\
                                                                        self.ensemble_params["g_inhib_const"],\
                                                                        self.ensemble_params["tau_inhib"]
        self.E_excite, self.E_inhib = self.ensemble_params["E_excite"], self.ensemble_params["E_inhib"]


