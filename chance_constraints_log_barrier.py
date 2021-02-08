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
Input constraints using a log barrier formulation

Implementation:
Uses custom newton method to solve
Uses JAX to compile and run code on GPU/CPU and provide gradients and hessians
"""

# general imports
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace, col_vec, row_vec
from pathlib import Path
import pickle



# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config
from jax.scipy.special import expit
# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy



# Control parameters
x_star = 1.0        # desired set point
M = 200             # number of samples we will use for MC MPC
N = 20              # horizonline of MPC algorithm
sqc = np.array([[1.0]])            # square root cost on state error
src = np.array([[1.0]])             # square root cost on control action

# simulation parameters
T = 100             # number of time steps to simulate and record measurements for
x0 = 3.0            # initial x
r_true = 0.01       # measurement noise standard deviation
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


# downsample the the HMC output since for illustration purposes we sampled > M
ind = np.random.choice(len(a), M, replace=False)
a = a[ind]  # same indices for all to ensure they correpond to the same realisation from dist
b = b[ind]
q = q[ind]
r = r[ind]

xt = np.expand_dims(z[ind, -1],0)  # inferred state for current time step

# we also need to sample noise
w = np.expand_dims(col_vec(q) * np.random.randn(M, N+1), 0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
ut = np.expand_dims(np.array([u[-1]]), 0)      # control action that was just applied


# ----- Solve the MC MPC control problem ------------------#

# jax compatible version of function to simulate forward the samples / scenarios
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


# compile cost and create gradient and hessian functions
cost = jit(log_barrier_cost, static_argnums=(11,12,13, 14))  # static argnums means it will recompile if N changes
gradient = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14))


# define some optimisation settings
mu = 1e4
gamma = 1
delta = 0.05
max_iter = 1000


# define some state constraints, (these need to be tuples (so trailing comma))
x_ub = 1.25
state_constraints = (lambda x: x_ub - x,)
input_constraints = (lambda u: 5.0 - u,)

theta = {'a':a,
         'b':b,}


# solve mpc optimisation problem
result = solve_chance_logbarrier(np.zeros((1,N)), cost, gradient, hessian, ut, xt, theta, w, x_star, sqc, src,
                            delta, simulate, state_constraints, input_constraints)

uc = result['uc']

x_mpc = simulate(xt, np.hstack([ut, uc]), w, theta)
hx = jnp.concatenate([state_constraint(x_mpc[:, :, 1:]) for state_constraint in state_constraints], axis=2)
cx = np.mean(hx > 0, axis=1)
cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints],axis=1)
print('State constraint satisfaction')
print(cx)
print('Input constraint satisfaction')
print(cu >= 0)
#
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.hist(x_mpc[0,:, i*3], label='MC forward sim')
    if i==1:
        plt.title('MPC solution over horizon')
    plt.axvline(x_star, linestyle='--', color='g', linewidth=2, label='target')
    plt.axvline(x_ub, linestyle='--', color='r', linewidth=2, label='upper bound')
    plt.xlabel('t+'+str(i*3+1))
plt.tight_layout()
plt.legend()
plt.show()

plt.plot(uc[0,:])
plt.title('MPC determined control action')
plt.show()







