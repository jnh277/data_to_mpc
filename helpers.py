###############################################################################
#    Data to Controller for Nonlinear Systems: An Approximate Solution
#    Copyright (C) 2021  Johannes Hendriks < johannes.hendriks@newcastle.edu.a >
#    and James Holdsworth < james.holdsworth@newcastle.edu.au >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

""" Some helper functions """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import gaussian_kde as kde
import os

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

def col_vec(x):
    return np.reshape(x, (-1,1))

def row_vec(x):
    return np.reshape(x, (1, -1))

def calculate_acf(x):
    lags = np.arange(0, 51, 1)
    acf = np.zeros(np.shape(lags))
    tmp = x - np.mean(x)
    for i in range(len(lags)):
        acf[i] = sum(tmp[:-len(lags)] * tmp[lags[i]:(-len(lags) + lags[i])])
    acf = acf / np.max(acf)
    return acf

def calc_MAP(x):
    min_x = np.min(x)
    max_x = np.max(x)
    pos = np.linspace(min_x, max_x, 100)
    kernel = kde(x)
    z = kernel(pos)
    return pos[np.argmax(z)]

def build_phi_matrix(obs,order,inputs):
    "Builds the regressor matrix"
    no_obs = len(obs)
    max_delay = np.max((order[0],order[1]-1))
    phi = np.zeros((no_obs-max_delay, np.sum(order)))
    for i in range(order[0]):
        phi[:,i] = obs[max_delay-i-1:-i-1]
    for i in range(order[1]):
        phi[:,i+order[0]] = inputs[max_delay-i:no_obs-i]
    return phi

def build_input_matrix(inputs, input_order):
    no_obs = len(inputs)
    max_delay = input_order - 1
    phi = np.zeros((no_obs-max_delay, input_order))
    for i in range(input_order):
        phi[:, i] = inputs[max_delay - i:no_obs - i]
    return phi

def build_obs_matrix(obs, output_order):
    no_obs = len(obs)
    max_delay = output_order
    phi = np.zeros((no_obs - max_delay, output_order))
    for i in range(output_order):
        phi[:,i] = obs[max_delay-i-1:-i-1]
    return phi


def generate_data(no_obs, a, b, sigmae):
    order_a = len(a)
    order_b = len(b)
    order_ab = order_a + order_b
    order_max = np.max((order_a, order_b))
    y = np.zeros(no_obs)
    u = np.random.normal(size=no_obs)

    y[0] = 0.0
    Phi = np.zeros((no_obs - order_max, order_ab))

    for t in range(order_max, no_obs):
        y[t] = np.sum(a * y[range(t-1, t-order_a, -1)])
        y[t] += np.sum(b * u[range(t-1, t-order_b, -1)])
        y[t] += sigmae * np.random.normal()
        Phi[t - order_max, :] = np.hstack((y[range(t-1, t-order_a-1, -1)], u[range(t-1, t-order_b-1, -1)]))

    return y, u, Phi


def plot_trace(param,num_plots,pos, param_name='parameter',save=False):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(num_plots, 1, pos)
    plt.hist(param, 30, density=True);
    sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()


def plot_bode(A_smps,B_smps,C_smps,D_smps,A_t,B_t,C_t,D_t,omega,no_plot=300,max_samples=1000, save=False):
    """plot bode diagram from estimated system samples and true sys"""
    no_samples = np.shape(A_smps)[0]
    n_states= np.shape(A_smps)[1]
    no_eval = max(no_samples,max_samples)
    sel = np.random.choice(np.arange(no_samples), no_eval, False)
    omega_res = max(np.shape(omega))

    mag_samples = np.zeros((omega_res, no_eval))
    phase_samples = np.zeros((omega_res, no_eval))

    count = 0
    for s in sel:
        A_s = A_smps[s]
        B_s = B_smps[s].reshape(n_states,-1)
        C_s = C_smps[s].reshape(-1,n_states)
        D_s = D_smps[s]
        w, mag_samples[:, count], phase_samples[:, count] = signal.bode((A_s,B_s,C_s,float(D_s)), omega)
        count = count + 1

    w, mag_true, phase_true = signal.bode((A_t, B_t, C_t, float(D_t)), omega)

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w.flatten(), mag_samples[:, 0], color='green', alpha=0.1, label='samples')  # Bode magnitude plot
    plt.semilogx(w.flatten(), mag_samples[:, 1:no_plot], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='blue', label='True system')  # Bode magnitude plot
    hm, = plt.semilogx(w.flatten(), np.mean(mag_samples, 1), '-.', color='orange', label='mean')  # Bode magnitude plot
    # hu, = plt.semilogx(w.flatten(), np.percentile(mag_samples, 97.5, axis=1),'--',color='orange',label='Upper CI')    # Bode magnitude plot

    plt.legend(handles=[h1, h2, hm])
    plt.legend()
    plt.title('Bode diagram')
    plt.ylabel('Magnitude (dB)')
    plt.xlim((min(omega), max(omega)))

    plt.subplot(2, 1, 2)
    plt.semilogx(w.flatten(), phase_samples[:,:no_plot], color='green', alpha=0.1)  # Bode phase plot
    plt.semilogx(w.flatten(), phase_true, color='blue')  # Bode phase plot
    plt.semilogx(w.flatten(), np.mean(phase_samples, 1), '-.', color='orange',
                       label='mean')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim((min(omega), max(omega)))

    if save:
        plt.savefig('bode_plot.png',format='png')

    plt.show()

def plot_dbode(num_samples,den_samples,num_true,den_true,Ts,omega,no_plot=300, max_samples=1000, save=False):
    """plot bode diagram from estimated discrete time system samples and true sys"""
    no_samples = np.shape(num_samples)[0]
    no_eval = min(no_samples,max_samples)
    sel = np.random.choice(np.arange(no_samples), no_eval, False)
    omega_res = max(np.shape(omega))

    mag_samples = np.zeros((omega_res, no_eval))
    phase_samples = np.zeros((omega_res, no_eval))



    count = 0
    for s in sel:
        den_sample = np.concatenate(([1.0], den_samples[s,:]), 0)
        num_sample = num_samples[s, :]
        w, mag_samples[:, count], phase_samples[:, count] = signal.dbode((num_sample, den_sample, Ts), omega)
        count = count + 1

    # calculate the true bode diagram
    # plot the true bode diagram
    w, mag_true, phase_true = signal.dbode((num_true.flatten(), den_true.flatten(), Ts), omega)

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w.flatten(), mag_samples[:, 0], color='green', alpha=0.1, label='samples')  # Bode magnitude plot
    plt.semilogx(w.flatten(), mag_samples[:, 1:no_plot], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='blue', label='True system')  # Bode magnitude plot
    hm, = plt.semilogx(w.flatten(), np.mean(mag_samples, 1), '-.', color='orange', label='mean')  # Bode magnitude plot
    # hu, = plt.semilogx(w.flatten(), np.percentile(mag_samples, 97.5, axis=1),'--',color='orange',label='Upper CI')    # Bode magnitude plot

    plt.legend(handles=[h1, h2, hm])
    plt.legend()
    plt.title('Bode diagram')
    plt.ylabel('Magnitude (dB)')
    plt.xlim((min(omega),min(max(omega),1/Ts*3.14)))

    plt.subplot(2, 1, 2)
    plt.semilogx(w.flatten(), phase_samples[:,:no_plot], color='green', alpha=0.1)  # Bode phase plot
    plt.semilogx(w.flatten(), phase_true, color='blue')  # Bode phase plot
    plt.semilogx(w.flatten(), np.mean(phase_samples, 1), '-.', color='orange',
                       label='mean')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim((min(omega), min(max(omega),1/Ts*3.14)))

    if save:
        plt.savefig('bode_plot.png',format='png')

    plt.show()

def plot_bode_ML(A_smps, B_smps, C_smps, D_smps, A_t, B_t, C_t, D_t, A_ML, B_ML, C_ML, D_ML, omega, no_plot=300, max_samples=1000, save=False):
    """plot bode diagram from estimated system samples and true sys and maximum likelihood estiamte"""
    no_samples = np.shape(A_smps)[0]
    n_states = np.shape(A_smps)[1]
    no_eval = max(no_samples, max_samples)
    sel = np.random.choice(np.arange(no_samples), no_eval, False)
    omega_res = max(np.shape(omega))

    mag_samples = np.zeros((omega_res, no_eval))
    phase_samples = np.zeros((omega_res, no_eval))

    count = 0
    for s in sel:
        A_s = A_smps[s]
        B_s = B_smps[s].reshape(n_states, -1)
        C_s = C_smps[s].reshape(-1, n_states)
        D_s = D_smps[s]
        w, mag_samples[:, count], phase_samples[:, count] = signal.bode((A_s, B_s, C_s, float(D_s)), omega)
        count = count + 1

    # now what if i also want to show the MAP estimate
    # no_freqs = np.shape(mag_samples)[0]
    # mag_MAP = np.zeros((no_freqs))
    # phase_MAP = np.zeros((no_freqs))
    # for k in range(no_freqs):
    #     mag_MAP[k] = calc_MAP(mag_samples[k, :])
    #     phase_MAP[k] = calc_MAP(phase_samples[k, :])

    w, mag_true, phase_true = signal.bode((A_t, B_t, C_t, float(D_t)), omega)
    w, mag_ML, phase_ML = signal.bode((A_ML, B_ML, C_ML, float(D_ML)), omega)

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w.flatten(), mag_samples[:, 0], color='green', alpha=0.1,
                       label='hmc samples')  # Bode magnitude plot
    plt.semilogx(w.flatten(), mag_samples[:, 1:no_plot], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='black', label='True system')  # Bode magnitude plot
    hml, = plt.semilogx(w.flatten(), mag_ML,'--', color='purple', label='ML estimate')  # Bode magnitude plot
    hm, = plt.semilogx(w.flatten(), np.mean(mag_samples, 1), '-.', color='orange',label='hmc mean')  # Bode magnitude plot
    # hmap = plt.semilogx(w.flatten(), mag_MAP, '-.', color='blue',label='hmc MAP')  # Bode magnitude plot

    # hu, = plt.semilogx(w.flatten(), np.percentile(mag_samples, 97.5, axis=1),'--',color='orange',label='Upper CI')    # Bode magnitude plot

    # plt.legend(handles=[h1, h2, hml, hm, hmap])
    plt.legend()
    plt.title('Bode diagram')
    plt.ylabel('Magnitude (dB)')
    plt.xlim((min(omega), max(omega)))

    plt.subplot(2, 1, 2)
    plt.semilogx(w.flatten(), phase_samples[:, :no_plot], color='green', alpha=0.1)  # Bode phase plot
    plt.semilogx(w.flatten(), phase_true, color='black')  # Bode phase plot
    plt.semilogx(w.flatten(), phase_ML,'--', color='purple')  # Bode phase plot
    plt.semilogx(w.flatten(), np.mean(phase_samples, 1), '-.', color='orange',
                 label='mean')  # Bode magnitude plot
    # plt.semilogx(w.flatten(), phase_MAP, '-.', color='blue',
    #              label='map')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim((min(omega), max(omega)))

    if save:
        plt.savefig('figures/example4_bode.png',format='png')

    plt.show()


def plot_dbode_ML(num_samples,den_samples,num_true,den_true,num_ML,den_ML,Ts,omega,no_plot=300, max_samples=1000, save=False):
    """plot bode diagram from estimated discrete time system samples and true sys"""
    no_samples = np.shape(num_samples)[0]
    no_eval = min(no_samples,max_samples)
    sel = np.random.choice(np.arange(no_samples), no_eval, False)
    omega_res = max(np.shape(omega))

    mag_samples = np.zeros((omega_res, no_eval))
    phase_samples = np.zeros((omega_res, no_eval))



    count = 0
    for s in sel:
        den_sample = np.concatenate(([1.0], den_samples[s,:]), 0)
        num_sample = num_samples[s, :]
        w, mag_samples[:, count], phase_samples[:, count] = signal.dbode((num_sample, den_sample, Ts), omega)
        count = count + 1

    # calculate the true bode diagram
    # plot the true bode diagram
    w, mag_true, phase_true = signal.dbode((num_true.flatten(), den_true.flatten(), Ts), omega)
    w, mag_ML, phase_ML = signal.dbode((num_ML.flatten(), den_ML.flatten(), Ts), omega)

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w.flatten(), mag_samples[:, 0], color='green', alpha=0.1, label='hmc samples')  # Bode magnitude plot
    plt.semilogx(w.flatten(), mag_samples[:, 1:no_plot], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='blue', label='True system')  # Bode magnitude plot
    h_ML, = plt.semilogx(w.flatten(), mag_ML,'--', color='purple', label='ML Estimate')  # Bode magnitude plot
    hm, = plt.semilogx(w.flatten(), np.mean(mag_samples, 1), '-.', color='orange', label='hmc mean')  # Bode magnitude plot
    # hu, = plt.semilogx(w.flatten(), np.percentile(mag_samples, 97.5, axis=1),'--',color='orange',label='Upper CI')    # Bode magnitude plot

    plt.legend(handles=[h1, h2, hm, h_ML])
    plt.legend()
    plt.title('Bode diagram')
    plt.ylabel('Magnitude (dB)')
    plt.xlim((min(omega),min(max(omega),1/Ts*3.14)))

    plt.subplot(2, 1, 2)
    plt.semilogx(w.flatten(), phase_samples[:,:no_plot], color='green', alpha=0.1)  # Bode phase plot
    plt.semilogx(w.flatten(), phase_true, color='blue')  # Bode phase plot
    plt.semilogx(w.flatten(), phase_ML,'--', color='purple')  # Bode phase plot
    plt.semilogx(w.flatten(), np.mean(phase_samples, 1), '-.', color='orange',
                       label='mean')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim((min(omega), min(max(omega),1/Ts*3.14)))
    # plt.ylim(-300,10)

    if save:
        plt.savefig('bode_plot.png',format='png')

    plt.show()

def plot_firfreq(b_samples,num_true,den_true,b_ML,Ts=1.0,no_plot=300, max_samples=1000, save=False):
    """plot bode diagram from estimated discrete time system samples and true sys"""
    no_samples = np.shape(b_samples)[0]
    no_eval = min(no_samples,max_samples)
    sel = np.random.choice(np.arange(no_samples), no_eval, False)
    # omega_res = max(np.shape(omega))
    omega_res = 512
    mag_samples = np.zeros((omega_res, no_eval))
    phase_samples = np.zeros((omega_res, no_eval))



    count = 0
    for s in sel:
        b_sample = b_samples[s, :]
        w_hmc, h = signal.freqz(b_sample)
        mag_samples[:,count] = 20 * np.log10(abs(h))
        phase_samples[:,count] = np.unwrap(np.angle(h))*180.0/3.14
        count = count + 1

    # calculate the true bode diagram
    # plot the true bode diagram
    w, mag_true, phase_true = signal.dbode((num_true.flatten(), den_true.flatten(), Ts))
    w_ML, h = signal.freqz(b_ML)
    mag_ML = 20 * np.log10(abs(h))
    phase_ML = np.unwrap(np.angle(h))*180.0/3.14

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w_hmc.flatten(), mag_samples[:, 0], color='green', alpha=0.1, label='hmc samples')  # Bode magnitude plot
    plt.semilogx(w_hmc.flatten(), mag_samples[:, 1:no_plot], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='blue', label='True system')  # Bode magnitude plot
    hml, = plt.semilogx(w_ML.flatten(), mag_ML,'--', color='purple', label='ML Estimate')  # Bode magnitude plot
    hm, = plt.semilogx(w_hmc.flatten(), np.mean(mag_samples, 1), '-.', color='orange', label='hmc mean')  # Bode magnitude plot
    # hu, = plt.semilogx(w.flatten(), np.percentile(mag_samples, 97.5, axis=1),'--',color='orange',label='Upper CI')    # Bode magnitude plot

    plt.legend(handles=[h1, h2, hm, hml])
    plt.legend()
    plt.title('Bode diagram')
    plt.ylabel('Magnitude (dB)')
    # plt.xlim((min(omega),min(max(omega),1/Ts*3.14)))

    plt.subplot(2, 1, 2)
    plt.semilogx(w_hmc.flatten(), phase_samples[:,:no_plot], color='green', alpha=0.1)  # Bode phase plot
    plt.semilogx(w.flatten(), phase_true, color='blue')  # Bode phase plot
    plt.semilogx(w_ML.flatten(), phase_ML,'--', color='purple')  # Bode magnitude plot
    plt.semilogx(w_hmc.flatten(), np.mean(phase_samples, 1), '-.', color='orange',
                       label='mean')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/sample)')
    # plt.xlim((min(omega), min(max(omega),1/Ts*3.14)))

    if save:
        plt.savefig('fir_bode_plot.png',format='png')

    plt.show()