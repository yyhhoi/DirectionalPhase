

class Simulation():
    def __init__(self, t, **external_inputs):
        """

        Args:
            t (numpy.darray): with shape (time, )
            **external_inputs (numpy.darray): (t, num_neurons), entry = input values. Expected keys = 'theta', 'run'
        """
        self.t = t
        self.timelength = self.t.shape[0]
        self.current_idx = 0
        self.current_time = self.getCurrentTime()
        self.external_inputs_dict = external_inputs


    def getCurrentTime(self):
        return self.t[self.current_idx]


    def get_dt(self):
        return self.t[self.current_idx + 1] - self.t[self.current_idx]

    def get_inputs(self):
        return {key:val[self.current_idx] for key, val in self.external_inputs_dict.items()}

    def increment(self):
        self.current_idx += 1
        self.current_time = self.getCurrentTime()

    def check_end(self):
        if self.current_idx < (self.timelength - 1):
            return True
        else:
            return False


