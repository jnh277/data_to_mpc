import os
import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")
# general imports
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace_grid
from pathlib import Path
import pickle

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev
from jax.lax import scan
from jax.ops import index, index_update
from jax.config import config
config.update("jax_enable_x64", True)


N = 200
e = np.random.normal(0.5,0.3,(N,))
x = np.linspace(0.1,np.pi,N)
theta = 1.5
y = theta + (x + e)**2


plt.plot(x, y)
plt.show()

# plt.hist(y-x)
# plt.show()

stan_data = {
    'N': N,
    'y':y,
    'x':x,
}

model = pystan.StanModel(file='stan/squared_transform.stan')
fit = model.sampling(data=stan_data, iter=4000, chains=4)
traces = fit.extract()

mu_fit = traces['mu']
sigma_fit = traces['sigma']
theta_fit = traces['theta']

plt.subplot(3,1,1)
plt.hist(mu_fit)

plt.subplot(3,1,2)
plt.hist(sigma_fit)

plt.subplot(3,1,3)
plt.hist(theta_fit)
plt.show()