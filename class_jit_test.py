
import numpy as np
import quadprog
import matplotlib.pyplot as plt

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev
from jax import device_put
from jax.config import config
from jax.scipy.special import expit

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import solve_chance_logbarrier, log_barrier_cosine_cost
import time
import osqp
from scipy import sparse
config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy



x = np.ones((5,1))
y = np.zeros((5,1))

def error_func(x, y):
    print('compiling')
    return x - y


class Test_Class:
    def __init__(self, error_func):

        self.error_func = error_func
        self.cost = jit(self.cost_func)
        self.grad = jit(grad(self.cost_func, argnums=0))

    def cost_func(self, x, y):
        return np.sum(self.error_func(x, y)**2)


test_obj = Test_Class(error_func)

c = test_obj.cost(x,y)
g = test_obj.grad(x,y)

c = test_obj.cost(x,y)
g = test_obj.grad(x,y)



def myfunc(x,y):
    c = 5 * np.sum(x**2) + 2 * np.sum(y**2)
    return c
x = np.ones((5,1))
y = np.ones((6,1))

J_myfunc = jacfwd(myfunc, argnums=(0,1))

print(myfunc(x,y))

J = J_myfunc(x,y)

