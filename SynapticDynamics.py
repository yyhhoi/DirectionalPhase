import numpy as np


class FiringMask():
    def __init__(self, num_neurons):
        self.firing_mask_2d_template = np.zeros((num_neurons, num_neurons))
        self.firing_mask = np.zeros(num_neurons)

    def update_mask(self, u, u_threshold):
        self.firing_mask = u > u_threshold  # firing mask ~ (num_fired, )

    def get_mask(self):
        return self.firing_mask

    def get_2d_rows(self):  # Expanded to multiple rows
        firing_mask_2d = self.firing_mask_2d_template.copy()
        firing_mask_2d[:, self.firing_mask == 1] = 1
        return firing_mask_2d

    def get_2d_cols(self):
        firing_mask_2d = self.firing_mask_2d_template.copy()

        firing_mask_2d[self.firing_mask == 1, :] = 1
        return firing_mask_2d
