

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
x[0] = 3.0                                  # initial state
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

for t in range(T):
    x[t+1] = ssm(x[t], u[t]) + q_true * np.random.randn()
    y[t] = 2*x[t] + r_true * np.random.randn()

# estimate system (estimates up to x_t)
stan_data = {
    'N': T,
    'y': y,
    'u': u
}
fit = model.sampling(data=stan_data, warmup=2000, iter=10000, chains=4)
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

