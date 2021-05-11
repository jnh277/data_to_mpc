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

""" This script runs simulation A) Pedagogical Example from the paper and saves the results """
""" This overwrites the preincluded results that match the paper plots """
""" The results can then be plotted using the script 'plot_single_state_demo.py' """

import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")

# general imports
import pystan
import numpy as np
from helpers import col_vec, suppress_stdout_stderr
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# jax related imports
import jax.numpy as jnp
from jax import grad, jit,  jacfwd, jacrev
from jax.ops import index, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy


# Control parameters
x_star = np.array([1.0])        # desired set point
M = 200                             # number of samples we will use for MC MPC
N = 10                              # horizonline of MPC algorithm
sqc = np.array([[1.0]])             # square root cost on state error
src = np.array([[0.01]])             # square root cost on control action
delta = 0.05                        # desired maximum probability of not satisfying the constraint

x_ub = 1.2
u_ub = np.pi/2
state_constraints = (lambda x: x_ub - x,lambda x: x)
input_constraints = (lambda u: u_ub - u,lambda u:u+u_ub)


# simulation parameters
T = 30              # number of time steps to simulate and record measurements for
x0 = 0.5            # initial time step
r_true = 0.01       # measurement noise standard deviation
q_true = 0.1       # process noise standard deviation
nu_true = 3.        # student t degrees of freedom

#----------------- Simulate the system-------------------------------------------#
def ssm(x, u, a=0.9, b=0.2):
    return a*x + b*np.sin(u)

x = np.zeros(T+1)
x[0] = x0                                   # initial state
w = np.random.normal(0.0, q_true, T+1)        # make a point of predrawing noise
y = np.zeros((T,))

# create some inputs that are random but held for 10 time steps
u = np.zeros((T+1,))     # first control action will be zero

### hmc parameters and set up the hmc model
warmup = 1000
chains = 4
iter = warmup + int(M/chains)
model_name = 'non_affine'
path = 'stan/'
if Path(path+model_name+'.pkl').is_file():
    model = pickle.load(open(path+model_name+'.pkl', 'rb'))
else:
    model = pystan.StanModel(file=path+model_name+'.stan')
    with open(path+model_name+'.pkl', 'wb') as file:
        pickle.dump(model, file)

## define jax friendly function for simulating the system during mpc
def simulate(xt, u, w, theta):
    a = theta['a']
    b = theta['b']
    [o, M, N] = w.shape
    x = jnp.zeros((o, M, N+1))
    x = index_update(x, index[:, :,0], xt)
    for k in range(N):
        x = index_update(x, index[:, :, k+1], a * x[:, :, k] + b * jnp.sin(u[:, k]) + w[:, :, k])
    return x[:, :, 1:]

# define MPC cost, gradient and hessian function
cost = jit(log_barrier_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
gradient = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))


xt_est_save = np.zeros((1,M,T))
a_est_save = np.zeros((M,T))
b_est_save = np.zeros((M,T))
q_est_save = np.zeros((M,T))
r_est_save = np.zeros((M,T))
mpc_result_save = []

### SIMULATE SYSTEM AND PERFORM MPC CONTROL
for t in tqdm(range(T),desc='Simulating system, running hmc, calculating control'):
    # simulate system
    # apply u_t
    # this takes x_t to x_{t+1}
    # measure y_t
    x[t+1] = ssm(x[t], u[t]) + x[t]*q_true * np.random.randn()
    y[t] = x[t] + r_true * np.random.standard_t(nu_true,1)

    # estimate system (estimates up to x_t)
    stan_data = {
        'N': t + 1,
        'y': y[:t + 1],
        'u': u[:t + 1],
        'prior_mu': np.array([0.8, 0.05, 0.1, 0.1]),
        'prior_std': np.array([0.2, 0.2, 0.2, 0.2]),
        'prior_state_mu': 0.3,
        'prior_state_std': 0.2,
        'nu': nu_true,
    }
    with suppress_stdout_stderr():
        fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains)
    traces = fit.extract()
    #
    # # state samples
    z = traces['z']
    #
    # parameter samples
    a = traces['a']
    b = traces['b']
    r = traces['r']
    q = traces['q']

    # current state samples (expanded to [o,M]
    xt = np.expand_dims(z[:, -1], 0)  # inferred state for current time step
    # we also need to sample noise
    w = np.expand_dims(col_vec(q) * np.random.randn(M, N + 1), 0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
    ut = np.expand_dims(np.array([u[t]]), 0)  # control action that was just applied
    theta = {'a': a,
             'b': b, }

    # save some things for later plotting
    xt_est_save[0,:,t] = z[:, -1]
    a_est_save[:, t] = a
    b_est_save[:, t] = b
    q_est_save[:, t] = q
    r_est_save[:, t] = r

    # calculate next control action
    result = solve_chance_logbarrier(np.zeros((1, N)), cost, gradient, hessian, ut, xt, theta, w, x_star, sqc, src,
                                     delta, simulate, state_constraints, input_constraints, verbose=False)

    mpc_result_save.append(result)
    uc = result['uc']
    u[t+1] = uc[0,0]


run = 'test'
with open('results/'+run+'/xt_est_save.pkl','wb') as file:
    pickle.dump(xt_est_save, file)
with open('results/'+run+'/a_est_save.pkl','wb') as file:
    pickle.dump(a_est_save, file)
with open('results/'+run+'/b_est_save.pkl','wb') as file:
    pickle.dump(b_est_save, file)
with open('results/'+run+'/q_est_save.pkl','wb') as file:
    pickle.dump(q_est_save, file)
with open('results/'+run+'/r_est_save.pkl','wb') as file:
    pickle.dump(r_est_save, file)
with open('results/'+run+'/u.pkl','wb') as file:
    pickle.dump(u, file)
with open('results/'+run+'/x.pkl','wb') as file:
    pickle.dump(x, file)
with open('results/'+run+'/mpc_result_save.pkl', 'wb') as file:
    pickle.dump(mpc_result_save, file)