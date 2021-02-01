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
NO state constraints are implemented,
input constraints are implemented

Implementation:
This implementation uses scipy's optimisation routines
Uses JAX to compile and run code on GPU/CPU and provide gradients for the optimisation routine
"""

# general imports
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace, col_vec, row_vec
from pathlib import Path
import pickle
from scipy.optimize import minimize

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put
from jax.ops import index, index_add, index_update
from jax.config import config
config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy

# Control parameters
x_star = 1.0        # desired set point
M = 200             # number of samples we will use for MC MPC
N = 20              # horizonline of MPC algorithm
qc = 1.0            # cost on state error
rc = 1.             # cost on control action
x_ub = 1.05         # upper bound constraint on state

# simulation parameters
T = 100             # number of time steps to simulate and record measurements for
x0 = 3.0            # initial x
r_true = 0.1        # measurement noise standard deviation
q_true = 0.05       # process noise standard deviation

#----------------- Simulate the system-------------------------------------------#
def ssm(x, u, a=0.9, b=0.1):
    return a*x + b*u

x = np.zeros(T+1)
x[0] = x0                                   # initial state
w = np.random.normal(0.0, q_true, T)        # make a point of predrawing noise

# create some inputs that are random but held for 10 time steps
u = np.random.uniform(-0.5,0.5, T)
u = np.reshape(u, (-1,10))
u[:,1:] = np.atleast_2d(u[:,0]).T * np.ones((1,9))
u = u.flatten()

for k in range(T):
    x[k+1] = ssm(x[k], u[k]) + w[k]

# simulate measurements
y = np.zeros(T)
y = x[:T] + np.random.normal(0.0, r_true, T)

plt.subplot(2,1,1)
plt.plot(u)
plt.title('Simulated inputs used for inference')

plt.subplot(2, 1, 2)
plt.plot(x)
plt.plot(y,linestyle='None',color='r',marker='*')
plt.title('Simulated state and measurements used for inferences')
plt.tight_layout()
plt.show()

#----------- USE HMC TO PERFORM INFERENCE ---------------------------#
# avoid recompiling
model_name = 'LSSM_demo'
path = 'stan/'
if Path(path+model_name+'.pkl').is_file():
    model = pickle.load(open(path+model_name+'.pkl', 'rb'))
else:
    model = pystan.StanModel(file=path+model_name+'.stan')
    with open(path+model_name+'.pkl', 'wb') as file:
        pickle.dump(model, file)

stan_data = {
    'N':T,
    'y':y,
    'u':u,
}

fit = model.sampling(data=stan_data, warmup=1000, iter=2000)
traces = fit.extract()

# state samples
z = traces['z']

# parameter samples
a = traces['a']
b = traces['b']
r = traces['r']
q = traces['q']

# plot the initial parameter marginal estimates
plot_trace(a,4,1,'a')
plt.title('HMC inferred parameters')
plot_trace(b,4,2,'b')
plot_trace(r,4,3,'r')
plot_trace(q,4,4,'q')
plt.show()

# plot some of the initial marginal state estimates
for i in range(4):
    if i==1:
        plt.title('HMC inferred states')
    plt.subplot(2,2,i+1)
    plt.hist(z[:, i*20+1],bins=30, label='p(x_'+str(i+1)+'|y_{1:T})', density=True)
    plt.axvline(x[i*20+1], label='True', linestyle='--',color='k',linewidth=2)
    plt.xlabel('x_'+str(i+1))
plt.tight_layout()
plt.legend()
plt.show()

# ----- Solve the MC MPC control problem ------------------#

# jax compatible version of function to simulate forward a sample / scenario
def simulate(xt, u, a, b, w):
    N = len(u)
    M = len(a)
    x = jnp.zeros((M, N+1))
    # x[:, 0] = xt
    x = index_update(x, index[:,0], xt)
    for k in range(N):
        # x[:, k+1] = a * x[:, k] + b * u[k] + w[:, k]
        x = index_update(x, index[:, k+1], a * x[:, k] + b * u[k] + w[:, k])
    return x[:, 1:]

# jax compatible version of function to compute cost
def expectation_cost(uc, ut, xt, x_star, a, b, w, qc, rc):
    u = jnp.hstack([ut, uc]) # u_t was already performed, so uc is N-1
    x = simulate(xt, u, a, b, w)
    V = jnp.sum((qc*(x - x_star)) ** 2) + jnp.sum((rc * uc)**2)
    return V

# input bounds
bnds = ((-3.0, 3.0),)*(N-1)
uc = jnp.zeros(N-1)  # initialise uc

# compile cost function
cost_jit = jit(expectation_cost)
gradient = jit(grad(cost_jit, argnums=0))    # get compiled function to return gradients with respect to uc

# downsample the the HMC output since for illustration purposes we sampled > M
ind = np.random.choice(len(a), M, replace=False)
a = a[ind]  # same indices for all to ensure they correpond to the same realisation from dist
b = b[ind]
q = q[ind]
r = r[ind]

xt = z[ind, -1]  # inferred state for current time step

# we also need to sample noise
w = col_vec(q) * np.random.randn(M, N)  # uses the sampled stds

ut = u[-1]      # control action that was just applied

# put everything we want to call onto the gpu
args = (device_put(ut), device_put(xt), device_put(x_star), device_put(a),
        device_put(b), device_put(w), device_put(qc), device_put(rc))

# wrap up the jit functions
def np_cost_jit(uc, *args):
    return np.array(cost_jit(device_put(uc), *args))

def np_gradient(uc, *args):  #
    return np.array(gradient(device_put(uc), *args))

res = minimize(np_cost_jit, uc, jac=np_gradient, bounds=bnds, args=(ut,xt,x_star,a,b,w,qc,rc))
# NOTE: the args command is the better way of dealing with extra inputs to cost and gradient functions
print(res)
uc = res.x


x_mpc = simulate(xt, np.hstack([ut, uc]), a, b, w)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.hist(x_mpc[:, i*3], label='MC forward sim')
    if i==1:
        plt.title('MPC solution over horizon')
    plt.axvline(1.0, linestyle='--', color='g', linewidth=2, label='target')
    plt.xlabel('t+'+str(i*3+1))
plt.tight_layout()
plt.legend()
plt.show()

plt.plot(uc)
plt.title('MPC determined control action')
plt.show()







