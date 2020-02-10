import numpy as np


class WeightGenerator:
    # 0-5 = Field A, 6-11 = Field B, 12-13 = PlaceCell A and B
    def __init__(self, num_neurons, excite_overlap=0):
        self.num_neurons = num_neurons
        self.weights_excite = np.zeros((self.num_neurons , self.num_neurons ))
        self.weights_inhib = np.ones((self.num_neurons , self.num_neurons )) - np.eye(self.num_neurons )
        self.excite_overlap = excite_overlap
        self._weight_config()

    def _weight_config(self):
        num_noplace = int(self.num_neurons-2)
        num_half = int((self.num_neurons-2)/2)
        self.weights_excite[self.num_neurons-2, 0:(num_half+self.excite_overlap)] = 0.6  # Connect to A
        self.weights_excite[self.num_neurons-1, (num_half-self.excite_overlap):num_noplace] = 0.6  # Connect to B
        self.weights_inhib[num_noplace:self.num_neurons, :] = 0
        self.weights_inhib[:, num_noplace:self.num_neurons] = 0

    def get_weights_excite(self):
        return self.weights_excite.copy()

    def get_weights_inhib(self):
        return self.weights_inhib.copy()


def Overlap0(num_neurons, exicte_overlap=0):


    wg = WeightGenerator(num_neurons, exicte_overlap)
    weights_excite = wg.get_weights_excite()
    weights_inhib = wg.get_weights_inhib()
    g_excite_const=weights_excite * 5e3
    tau_excite=weights_excite * 20e-3
    E_excite = 0

    g_inhib_const=weights_inhib * 5e3
    tau_inhib=weights_inhib * 40e-3

    E_inhib = -70e-3

    return weights_excite, g_excite_const, tau_excite, E_excite, weights_inhib, g_inhib_const, tau_inhib, E_inhib
