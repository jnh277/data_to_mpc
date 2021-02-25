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

x_ub = 1.2
u_ub = 2.
state_constraints = (lambda x: x_ub - x,)
input_constraints = (lambda u: u_ub - u,)


# simulation parameters
T = 30              # number of time steps to simulate and record measurements for
x0 = 0.5            # initial time step
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




run = 'single_state_run2'
with open('results/'+run+'/xt_est_save.pkl','rb') as file:
    xt_est_save = pickle.load(file)
with open('results/'+run+'/a_est_save.pkl','rb') as file:
    a_est_save = pickle.load(file)
with open('results/'+run+'/b_est_save.pkl','rb') as file:
    b_est_save = pickle.load(file)
with open('results/'+run+'/q_est_save.pkl','rb') as file:
    q_est_save = pickle.load(file)
with open('results/'+run+'/r_est_save.pkl','rb') as file:
    r_est_save = pickle.load(file)
with open('results/'+run+'/x.pkl','rb') as file:
    x = pickle.load(file)
with open('results/'+run+'/u.pkl','rb') as file:
    u = pickle.load(file)
with open('results/'+run+'/mpc_result_save100.pkl', 'rb') as file:
    mpc_result_save = pickle.load(file)



plt.subplot(2,1,1)
#print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.fill_between(np.arange(T),np.percentile(xt_est_save[0,:,:],97.5,axis=0),np.percentile(xt_est_save[0,:,:],2.5,axis=0),alpha=0.2,label='95% CI',color=u'#1f77b4')
plt.plot(x,label='True', color='k')
plt.plot(xt_est_save[0,:,:].mean(axis=0),label='mean',color=u'#1f77b4',linestyle='--')
# plt.plot(np.percentile(xt_est_save[0,:,:],97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(xt_est_save[0,:,:],2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel('x')
plt.axhline(x_ub,linestyle='--',color='r',linewidth=2.,label='constraint')
plt.axhline(x_star,linestyle='--',color='g',linewidth=2.,label='target')
plt.legend()

plt.subplot(2,1,2)
plt.plot(u)
plt.axhline(u_ub,linestyle='--',color='r',linewidth=2.,label='constraint')
plt.ylabel('u')
plt.xlabel(r'$t$')

plt.tight_layout()
plt.show()

plt.subplot(2,2,1)
plt.plot(a_est_save.mean(axis=0),label='mean')
plt.fill_between(np.arange(T),np.percentile(a_est_save,97.5,axis=0),np.percentile(a_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
# plt.plot(np.percentile(a_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(a_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel(r'$a$')
plt.axhline(0.9,linestyle='--',color='k',linewidth=2.,label='true')
plt.legend()
plt.xlabel(r'$t$')

plt.subplot(2,2,2)
plt.plot(b_est_save.mean(axis=0),label='mean')
plt.fill_between(np.arange(T),np.percentile(b_est_save,97.5,axis=0),np.percentile(b_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
# plt.plot(np.percentile(b_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(b_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel(r'$b$')
plt.axhline(0.1,linestyle='--',color='k',linewidth=2.,label='true')
plt.legend()
plt.xlabel(r'$t$')

plt.subplot(2,2,3)
plt.plot(q_est_save.mean(axis=0),label='mean')
plt.fill_between(np.arange(T),np.percentile(q_est_save,97.5,axis=0),np.percentile(q_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
# plt.plot(np.percentile(q_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(q_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel(r'$q$')
plt.axhline(q_true,linestyle='--',color='k',linewidth=2.,label='true')
plt.legend()
plt.xlabel(r'$t$')

plt.subplot(2,2,4)
plt.plot(r_est_save.mean(axis=0),label='mean')
plt.fill_between(np.arange(T),np.percentile(r_est_save,97.5,axis=0),np.percentile(r_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
# plt.plot(np.percentile(r_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(r_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel(r'$r$')
plt.axhline(r_true,linestyle='--',color='k',linewidth=2.,label='true')
plt.legend()
plt.xlabel(r'$t$')

plt.tight_layout()
plt.show()
