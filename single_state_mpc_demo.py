"""
Simulate a first order state space model given by
x_{t+1} = a*x_t + b*u_t + w_t
y_t = x_t + e_t
with q and r the standard deviations of w_t and q_t respectively

Jointly estimate the smoothed state trajectories and parameters theta = {a, b, q, r}
to give p(x_{1:T}, theta | y_{1:T})

GOAL:
Use a Monte Carlo style approach to perform MPC where the state constraints are satisfied
with a given probability.

Current set up: Uses MC to give an expected cost and then satisfies,
Chance state constraints using a log barrier formulation
Chance state constraints using a log barrier formulation
Input constraints using a log barrier formulation

Implementation:
Uses custom newton method to solve
Uses JAX to compile and run code on GPU/CPU and provide gradients and hessians
"""

# general imports
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace, col_vec, row_vec, suppress_stdout_stderr
from pathlib import Path
import pickle
from tqdm import tqdm

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
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
# x_ub = 1.2
x_ub = 1.
u_ub = 1.5
state_constraints = (lambda x: x_ub - x,)
input_constraints = (lambda u: u_ub - u,)


# simulation parameters
T = 30              # number of time steps to simulate and record measurements for
# x0 = 0.5            # initial time step
x0 = -1.
r_true = 0.01       # measurement noise standard deviation
q_true = 0.05       # process noise standard deviation

#----------------- Simulate the system-------------------------------------------#
def ssm(x, u, a=0.9, b=0.1):
    return a*x + b*u

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
model_name = 'single_state_gaussian_priors'
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
    # x[:, 0] = xt
    x = index_update(x, index[:, :,0], xt)
    for k in range(N):
        # x[:, k+1] = a * x[:, k] + b * u[k] + w[:, k]
        x = index_update(x, index[:, :, k+1], a * x[:, :, k] + b * u[:, k] + w[:, :, k])
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
### SIMULATE SYSTEM AND PERFORM MPC CONTROL
for t in tqdm(range(T),desc='Simulating system, running hmc, calculating control'):
    # simulate system
    # apply u_t
    # this takes x_t to x_{t+1}
    # measure y_t
    x[t+1] = ssm(x[t], u[t]) + q_true * np.random.randn()
    y[t] = x[t] + r_true * np.random.randn()

    # estimate system (estimates up to x_t)
    stan_data = {
        'N': t+1,
        'y': y[:t+1],
        'u': u[:t+1],
        # 'prior_mu': np.array([0.8, 0.05, 0.1, 0.1]),
        # 'prior_std': np.array([0.2, 0.2, 0.2, 0.2]),
        'prior_mu': np.array([-0.5, 0.05, 0.1, 0.1]),
        'prior_std': np.array([1.0, 0.2, 0.2, 0.2]),
        'prior_state_mu': 0.3,
        'prior_state_std': 0.2,
    }
    with suppress_stdout_stderr():
        fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains)
    traces = fit.extract()

    # state samples
    z = traces['z']

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

    uc = result['uc']
    u[t+1] = uc[0,0]


plt.subplot(2,1,1)
plt.plot(x,label='True', color='k')
plt.plot(xt_est_save[0,:,:].mean(axis=0), color='b',label='mean')
plt.plot(np.percentile(xt_est_save[0,:,:],97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
plt.plot(np.percentile(xt_est_save[0,:,:],2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel('x')
plt.axhline(x_ub,linestyle='--',color='r',linewidth=2.,label='constraint')
plt.axhline(x_star,linestyle='--',color='g',linewidth=2.,label='target')
plt.legend()

plt.subplot(2,1,2)
plt.plot(u)
plt.axhline(u_ub,linestyle='--',color='r',linewidth=2.,label='constraint')
plt.ylabel('u')
plt.xlabel('t')

plt.tight_layout()
plt.show()

plt.subplot(2,2,1)
plt.plot(a_est_save.mean(axis=0),label='mean')
plt.plot(np.percentile(a_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
plt.plot(np.percentile(a_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel('a')
plt.axhline(0.9,linestyle='--',color='k',linewidth=2.,label='true')
plt.legend()

plt.subplot(2,2,2)
plt.plot(b_est_save.mean(axis=0),label='mean')
plt.plot(np.percentile(b_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
plt.plot(np.percentile(b_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel('b')
plt.axhline(0.1,linestyle='--',color='k',linewidth=2.,label='true')
plt.legend()

plt.subplot(2,2,3)
plt.plot(q_est_save.mean(axis=0),label='mean')
plt.plot(np.percentile(q_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
plt.plot(np.percentile(q_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel('q')
plt.axhline(q_true,linestyle='--',color='k',linewidth=2.,label='true')
plt.legend()
plt.xlabel('t')

plt.subplot(2,2,4)
plt.plot(r_est_save.mean(axis=0),label='mean')
plt.plot(np.percentile(r_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
plt.plot(np.percentile(r_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel('r')
plt.axhline(r_true,linestyle='--',color='k',linewidth=2.,label='true')
plt.legend()
plt.xlabel('t')

plt.show()
