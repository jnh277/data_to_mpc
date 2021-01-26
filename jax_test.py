import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy.optimize import minimize

import jax.numpy as jnp
from jax import grad, jit, device_put


a = jnp.array([-2,1.5])
c = 0.5
def cost(x, a, c):
    return jnp.power(x-a,2).sum() + c


cost_jit = jit(cost, static_argnums=(2,))
gradient = grad(cost_jit, argnums=0)


def npgradient(x, *args): # need this wrapper for scipy.optimize.minimize
    return 0+np.asarray(gradient(x, *args))  # adding 0 since 'L-BFGS-B' otherwise complains about contig. problems ...


x0 = jnp.ones((2,))


res = minimize(cost_jit, x0, jac=npgradient, args=(a, c))

print(res.x)



