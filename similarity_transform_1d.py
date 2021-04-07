

# general imports
import pystan
import numpy as np
from helpers import col_vec, plot_trace
from pathlib import Path
import pickle
from matplotlib import pyplot as plt

# jax related imports
import jax.numpy as jnp
from jax import grad, jit,  jacfwd, jacrev
from jax.ops import index, index_update
from jax.config import config
config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier


# Control parameters
x_star = np.array([1.0])        # desired set point
M = 2400                             # number of samples we will use for MC MPC
N = 20                              # horizonline of MPC algorithm
sqc = np.array([[1.0]])             # square root cost on state error
src = np.array([[0.01]])             # square root cost on control action
delta = 0.05                        # desired maximum probability of not satisfying the constraint

x_ub = np.Inf
u_ub = 10.
state_constraints = (lambda x: x_ub - x,)
input_constraints = (lambda u: u_ub - u,)

# simulation parameters
T = 20              # number of time steps to simulate and record measurements for
Nu = 1
Nx = 1
Ny = 1
r_true = 0.1       # measurement noise standard deviation
q_true = 0.1       # process noise standard deviation

#----------------- Simulate the system-------------------------------------------#
def ssm(x, u, a=0.5, b=0.1):
    return a*x + b*u

x = np.zeros(T+1)
x[0] = 0.                                 # initial state
w = np.random.normal(0.0, q_true, T+1)        # make a point of predrawing noise
y = np.zeros((T,))

# create some inputs that are random but held for 10 time steps
u = np.random.uniform(-10,10, T)

### hmc parameters and set up the hmc model
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
    x = index_update(x, index[:, :,0], xt)
    for k in range(N):
        x = index_update(x, index[:, :, k+1], a * x[:, :, k] + b * u[:, k] + w[:, :, k])
    return x[:, :, 1:]

# define MPC cost, gradient and hessian function
cost = jit(log_barrier_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
gradient = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))

for t in range(T):
    x[t+1] = ssm(x[t], u[t]) + q_true * np.random.randn()
    y[t] = 2*x[t] + r_true * np.random.randn()

# estimate system (estimates up to x_t)
stan_data = {
    'N': T,
    'y': y,
    'u': u
}
fit = model.sampling(data=stan_data, warmup=2000, iter=2000+M//4, chains=4)
traces = fit.extract()

# state samples
z = traces['z']

# parameter samples
a = traces['a']
b = traces['b']
c = traces['c']
prod1 = c*b
# r = traces['r']

plot_trace(a,4,1,'a')
plt.title('HMC inferred parameters')
plot_trace(b,4,2,'b')
plot_trace(c,4,3,'c')
plot_trace(prod1,4,4,'c*b')
plt.show() # this plot hopefulyl shows a nicely bimodal estimate of b and c, but a constant multiple bc

# current state samples (expanded to [o,M]
xt = np.expand_dims(z[:, 0], 0)  # inferred state for current time step
# we also need to sample noise
w = np.expand_dims(col_vec(0.1) * np.random.randn(M, N + 1), 0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
ut = np.expand_dims(np.array([u[-1]]), 0)  # control action that was just applied
theta = {'a': a,
         'b': b }

# calculate next control action
result = solve_chance_logbarrier(np.zeros((1, N)), cost, gradient, hessian, ut, xt, theta, w, x_star, sqc, src,
                                    delta, simulate, state_constraints, input_constraints, verbose=True)

uc = result['uc']

x_mpc = simulate(xt, np.hstack([ut, uc]), w,theta)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.hist(x_mpc[:, i*2], label='MC forward sim')
    if i==1:
        plt.title('MPC solution over horizon')
    plt.axvline(1.0, linestyle='--', color='g', linewidth=2, label='target')
    plt.xlabel('t+'+str(i*2+1))
plt.tight_layout()
plt.legend()
plt.show()

print(uc)
plt.plot(uc)
plt.title('MPC determined control action')
plt.show()