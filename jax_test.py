"""
Script for demonstrating and testing ways to use JAX to compute gradients, jacobians,
and Hessians and use them within optimisation routines

"""

import numpy as np
from scipy.optimize import minimize, least_squares
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev, jvp


a = jnp.array([-2,1.5])
c = 0.5
def cost(x, a, c):
    return jnp.power(x-a,2).sum() + c


cost_jit = jit(cost)
gradient = grad(cost_jit, argnums=0)
hessian = jacfwd(jacrev(cost_jit, argnums=0))

def npgradient(x, *args): # need this wrapper for scipy.optimize.minimize
    return 0+np.asarray(gradient(x, *args))  # adding 0 since 'L-BFGS-B' otherwise complains about contig. problems ...


x0 = jnp.ones((2,))

# providing gradient and hessian to optimisation routine
res = minimize(cost_jit, x0, jac=npgradient, args=(a, c), method='Newton-CG', hess=hessian, options={'disp':True})
print(res.x)


# look at computing the jacobian and how this might be used in a least squares setting
a = device_put(np.random.uniform(0.5, 1.5, 5))
A = jnp.stack([a, a**2]).T
x = jnp.array([1.0, 2.0])
b = A @ x

def error_func(x, A, b):
    return b - jnp.matmul(A, x)

error_func_jit = jit(error_func)

# use forward mode differentiation when jacobian will be tall
jac1 = jacfwd(error_func_jit, argnums=0)

# use reverse mode differentiation when jacobian will be fat
jac2 = jacrev(error_func_jit, argnums=0)
x0 = np.array([0,0])

res2 = least_squares(error_func_jit, x0, jac=jac1, args=(A,b))
print(res2)


