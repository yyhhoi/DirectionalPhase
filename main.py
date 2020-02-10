import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean
from pycircstat.descriptive import mean as pycirc_mean
from pycircstat.descriptive import median as pycirc_median
from astropy.stats.circstats import circmean as astro_mean
from Ensembles import Ensemble
from Environments import Simulation
from params import Overlap0


class SpikeTimeRecorder:
    def __init__(self, num_neurons):
        self.record_list = [[] for i in range(num_neurons)]

    def record(self, current_time, firing_mask):
        indexes = np.where(firing_mask > 0)
        if indexes[0].shape[0] > 0:
            for ind in indexes[0]:
                self.record_list[ind].append(current_time)

def pair_dist(arrA, arrB):
    sizeA = arrA.shape[0]
    sizeB = arrB.shape[0]

    dist = np.matmul(arrA.reshape(-1, 1),np.ones((1, sizeB))) - np.matmul(np.ones((sizeA, 1)), arrB.reshape(1, -1))
    return dist


def gauss(x, mean, sd):
    const = 1/(sd * np.sqrt(2*np.pi))
    exponent = (-1/2) * np.power( ((x-mean)/sd), 2)
    return const * np.exp(exponent)

def gen_sinusoid(t, amp):
    sinusoid = np.sin(2 * np.pi * 12 * t) * amp + amp
    return sinusoid

def gen_external_input(num_neurons, theta_amp, t, runs_amp, runs_sd):
    # Theta
    sinusoid = gen_sinusoid(t, theta_amp)

    # Run
    means = np.linspace(0, 1, num_neurons-2)
    runs = [gauss(t, mean, runs_sd) for mean in means]
    runs.append(np.zeros(t.shape))
    runs.append(np.zeros(t.shape))

    runs_np = np.stack(runs).T * runs_amp  # (num_neurons, t) -> (t, num_neurons)

    return sinusoid, runs_np


if __name__ == '__main__':

    t = np.linspace(0,  1, 1000)
    theta_amp = 12
    runs_amp = 3
    num_neurons = 100
    overlap = 40

    sptA_list = []
    sptB_list = []
    dist_mat_list = []
    for trial_idx in range(1):
        print(trial_idx)
        sinusoid, runs_np = gen_external_input(num_neurons=num_neurons,
                                               theta_amp=theta_amp,
                                               t=t,
                                               runs_amp=runs_amp,
                                               runs_sd=0.05)
        # runs_np = np.ones((t.shape[0], num_neurons))
        weights_excite, g_excite_const, tau_excite, E_excite, weights_inhib, g_inhib_const, tau_inhib, E_inhib = Overlap0(num_neurons=num_neurons,
                                                                                                                          exicte_overlap=overlap)
        theta_input = np.ones((t.shape[0], num_neurons)) * sinusoid.reshape(-1, 1)
        theta_input[:, -2:] = 0
        sp_recorder = SpikeTimeRecorder(num_neurons)
        external_inputs = dict(
            run=runs_np,
            theta=theta_input
        )
        simenv = Simulation(t, **external_inputs)
        ensemble_params = dict(
            u_rest=np.ones(num_neurons, ) * -60e-3,
            tau_m=np.ones(num_neurons, ) * 50e-3,
            u_threshold=np.random.uniform(-43e-3, -47e-3, size=(num_neurons, )),
            u_reset=np.ones(num_neurons, ) * -60e-3,
            g_excite_const=g_excite_const,
            tau_excite=tau_excite,
            g_inhib_const=g_inhib_const,
            tau_inhib=tau_inhib,
            E_excite=E_excite,
            E_inhib=E_inhib,
            weights_excite=weights_excite,
            weights_inhib=weights_inhib,
            E_run=0,
            E_theta=0,
        )
        ensem = Ensemble(simenv, num_neurons, **ensemble_params)

        soma = []
        syn = []

        while ensem.simenv.check_end():
            sp_recorder.record(ensem.simenv.getCurrentTime(), ensem.firing_mask.get_mask())
            soma.append(ensem.u.copy())
            syn.append(ensem.synaptic_currents.copy())
            ensem.state_update()

        soma_np = np.stack(soma)
        syn_np = np.stack(syn)

        sptA = np.array(sp_recorder.record_list[-2])
        sptB = np.array(sp_recorder.record_list[-1])
        dist_mat = pair_dist(sptA, sptB).flatten()
        dist_mat_within = dist_mat[np.abs(dist_mat) < (1/12)]
        dist_mat_list.append(dist_mat_within)
    # all_dists = np.concatenate(dist_mat_list)
    # all_dists = (all_dists/(1/12)) * np.pi
    # histinfo = plt.hist(all_dists, bins=60)
    # mean_phase = astro_mean(all_dists)
    # plt.plot([mean_phase, mean_phase], [0, np.max(histinfo[0])], c='r')
    # plt.xlim(-np.pi, np.pi)
    # plt.show()



    # Visualization

    figsize = (16, 8)

    # fig_soma, ax_soma = plt.subplots(num_neurons+1, figsize=figsize)
    # for i in range(num_neurons):
    #     ax_soma[i].plot(t[:-1], soma_np[:, i])
    #     ax_soma[i].set_ylabel('%d'%(i+1))
    # ax_soma[-1].plot(t[:-1], sinusoid[:-1])
    # fig_soma.suptitle('Soma')
    #
    # fig_syn_current, ax_syn_current = plt.subplots(num_neurons+1, figsize=figsize)
    # for i in range(num_neurons):
    #     ax_syn_current[i].plot(t[:-1], syn_np[:, i])
    #     ax_syn_current[i].set_ylabel('%d'%(i+1))
    # ax_syn_current[-1].plot(t[:-1], sinusoid[:-1])
    # fig_syn_current.suptitle('Synaptic current')

    num_neurons_to_plot = num_neurons
    fig_rastor, ax_rastor = plt.subplots(1, figsize=figsize)
    y_map = np.linspace(0, num_neurons_to_plot, 100)
    xx, yy = np.meshgrid(t, y_map)
    theta_modulation_color = gen_sinusoid(xx, theta_amp)
    ax_rastor.pcolormesh(xx, yy, theta_modulation_color, cmap='gray')
    rastor_c = ['b'] * (num_neurons_to_plot-2) + ['r'] * 2
    ax_rastor.eventplot(sp_recorder.record_list, linelengths=0.5, color = rastor_c)



    # fig_run, ax_run = plt.subplots(num_neurons+1, figsize=figsize)
    # for i in range(num_neurons):
    #     ax_run[i].plot(t[:-1], runs_np[:-1, i])
    #     ax_run[i].set_ylabel('%d'%(i+1))
    # ax_run[-1].plot(t[:-1], sinusoid[:-1])
    # fig_run.suptitle('Runs')


    plt.show()