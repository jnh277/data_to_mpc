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
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config
from jax.scipy.special import expit
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

#
def logbarrier(z, mu):       # log barrier for the constraint z >= 0
    return jnp.sum(-mu * jnp.log(z))

def chance_constraint(x, s, x_ub, gamma, delta):    # upper bounded chance constraint on the state
    return jnp.mean(expit((x_ub - x) / gamma), axis=0) - (1 - s)     # take the sum over the samples (M)

# jax compatible version of function to compute cost
def cost(z, ut, xt, x_star, a, b, w, qc, rc, mu, gamma, x_ub, delta):
    # TODO: this should not be hard coded but having trouble with compilation right now when it is dynamic indexing
    uc = z[:19]              # control input variables  #
    s = z[19:]               # slack variables

    u = jnp.hstack([ut, uc]) # u_t was already performed, so uc is N-1
    x = simulate(xt, u, a, b, w)
    # state error and input penalty cost and cost that drives slack variables down
    V1 = jnp.sum((qc*(x - x_star)) ** 2) + jnp.sum((rc * uc)**2) + jnp.sum(10 * (s + 1e3)**2)
    # need a log barrier on each of the slack variables to ensure they are positve
    V2 = logbarrier(s - delta, mu)       # aiming for 95% accuracy
    # now the chance constraints
    cx = chance_constraint(x[:,1:], s, x_ub, gamma, delta)        # cx = c(u,s) - delta
    V3 = logbarrier(cx, mu)
    return V1 + V2 + V3


# compile cost and create gradient and hessian functions
cost_jit = jit(cost)
gradient = jit(grad(cost, argnums=0))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(cost, argnums=0)))

# define some optimisation settings
mu = 1e4
gamma = 1
x_ub = 1
delta = 0.05

# put everything we want to call onto the gpu
# args = (device_put(ut), device_put(xt), device_put(x_star), device_put(a),
#         device_put(b), device_put(w), device_put(qc), device_put(rc),
#         device_put(mu), device_put(gamma), device_put(x_ub), device_put(delta), device_put(N))
args = (device_put(ut), device_put(xt), device_put(x_star), device_put(a),
        device_put(b), device_put(w), device_put(qc), device_put(rc),
        device_put(mu), device_put(gamma), device_put(x_ub), device_put(delta))

# test
z0 = jnp.hstack([jnp.zeros((N-1)), np.ones((N-1,))])
test = cost_jit(z0, *args)
testgrad = gradient(z0, *args)
testhessian = hessian(z0, *args)

max_iter = 1000

z = np.hstack([np.zeros((N-1,)), np.ones((N-1,))]) # only using one slack variable for all constraints right now
mu = 1e4
gamma = 1.0
args = (device_put(ut), device_put(xt), device_put(x_star), device_put(a),
        device_put(b), device_put(w), device_put(qc), device_put(rc),
        device_put(mu), device_put(gamma), device_put(x_ub), device_put(delta))

for i in range(max_iter):
    # compute cost, gradient, and hessian
    jz = device_put(z)
    c = np.array(cost_jit(jz, *args))
    g = np.array(gradient(jz, *args))
    h = np.array(hessian(jz, *args))

    # compute search direction
    p = - np.linalg.solve(h, g)

    # perform line search
    alpha = 1.0
    for k in range(52):
        # need to use the wolfe conditions to ensure a bit of a better decrease
        ztest = z + alpha * p
        ctest = np.array(cost_jit(device_put(ztest), *args))
        # if np.isnan(ctest) or np.isinf(ctest):
        #     continue
        # nan and inf checks should be redundant
        if ctest < c:
            z = ztest
            break

        alpha = alpha / 2

    if k == 51:
        print('Failed line search')
        break

    print('Iter:', i+1, 'Cost: ', c, 'nd:',np.dot(g,p),'alpha: ', alpha, 'mu: ', mu, 'gamma: ', gamma)

    if np.abs(np.dot(g,p)) < 1e-2: # if search direction was really small, then decrease mu and s for next iteration
        if mu < 1e-6 and gamma < 1e-3:
            break

        mu = max(mu / 2, 0.999e-6)
        gamma = max(gamma / 1.25, 0.999e-3)
        # need to adjust the slack after changing gamma
        x_new = simulate(xt, np.hstack([ut, z[:N-1]]), a, b, w)
        cx = chance_constraint(x_new[:, 1:], z[N-1:], x_ub, gamma, delta)
        z[N-1:] += -np.minimum(cx, 0)
        args = (device_put(ut), device_put(xt), device_put(x_star), device_put(a),
                device_put(b), device_put(w), device_put(qc), device_put(rc),
                device_put(mu), device_put(gamma), device_put(x_ub), device_put(delta))


x_mpc = simulate(xt, np.hstack([ut, z[:N-1]]), a, b, w)
cx = chance_constraint(x_mpc, 0.0, x_ub, gamma, delta)
print('Constraint satisfaction')
print(1 + cx)
#
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

plt.plot(z[:N-1])
plt.title('MPC determined control action')
plt.show()







