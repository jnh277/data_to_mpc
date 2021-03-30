'''
The problem to be solved
find u that
minimise        (x-10)^T (x-10)
subject to      P(-x + 10 >= 0) > 1-epsilon

where  x = u + w
and w ~ N(mu,sigma)


'''


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
# import osqp
from scipy import sparse
config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy

qpsolver = 'quadprog'
# qpsolver = 'OSQP'

M = 1000            # number of samples
w = np.zeros((2,M))
w[0,:] = np.random.randn(M,)
w[1,:] = 0.5 * np.random.randn(M,)


# plt.scatter(w[0,:], w[1,:])
# plt.show()



# define some constraints
upper_bound = 10.0
constraints = (lambda z: upper_bound - z[[0],:],lambda z: upper_bound - z[[1],:])

# define the cost function
def cost(z, w):
    u = jnp.expand_dims(z[:2], 1)
    epsilon = z[2]
    x = u + w
    V1 = jnp.sum(((x - 10) **2)) + 1e3*(1e3 + epsilon)**2
    return V1

def chance_constraint(hu, epsilon, gamma):    # Pr(h(u) >= 0 ) >= (1-epsilon)
    return jnp.mean(expit(hu / gamma), axis=1) - (1 - epsilon)     # take the sum over the samples (M)

def constraint(z, w, gamma):
    u = jnp.expand_dims(z[:2],1)
    x = u + w
    epsilon = z[2]
    C = jnp.hstack((chance_constraint(upper_bound - x[[0], :], epsilon, gamma)[0], epsilon - 0.05))
    return C

def lagrangian(z,lams, gamma, w):
    L1 = cost(z,w)
    L2 = jnp.sum(- lams * constraint(z,w,gamma))
    return L1+L2

def merit_function(z, w, inv_mu, gamma):
    return cost(z, w) - inv_mu * np.sum(np.minimum(constraint(z,w,gamma), 0))


Hfunc = jit(jacfwd(jacrev(lagrangian, argnums=0)))
# ddFfunc = jit(jacfwd(jacrev(cost, argnums=0)))
Ffunc = jit(cost)
dFfunc = jit(grad(cost, argnums=0))
dCfunc = jit(jacfwd(constraint, argnums=0))
ddCfunc = jit(jacfwd(jacrev(constraint, argnums=0)))

u0 = 5*np.ones((2,1))
epsilon0 = 1.0*np.ones((1,1))
# u0 = 0*np.ones((2,1))
# epsilon0 = np.ones((1,1))
lams0 = 100*np.ones((2,))
gamma0 = 1
z0 = np.vstack((u0,epsilon0)).flatten()

## do some stuff
z = device_put(z0)
lams = device_put(lams0)
gamma = device_put(gamma0)
w = device_put(w)
upper_bound = device_put(upper_bound)

if qpsolver == 'OSQP':
    prob = osqp.OSQP()

ts = time.time()
count = 0
old_nd = 1e6
c1 = 0.01
for i in range(30):
    H = np.array(Hfunc(z,lams,gamma,w))
    dF = np.array(dFfunc(z,w))
    dC = np.array(dCfunc(z, w, gamma)).T
    C = np.array([constraint(z, w, gamma)]).flatten()


    [d, v] = np.linalg.eig(H)
    if (d < 1e-5).any():
        Hold = 1.0*H
        if np.iscomplex(d).any():  # stop complex eigenvalues
            d = np.real(d)
            v = np.real(v)
        ind = d < 1e-5
        d[ind] = 1e-5 + np.abs(d[ind])
        H = v @ np.diag(d) @ v.T
        # print('here')



    if qpsolver == 'OSQP': ## solve using OSQP

        q = np.array(dF)

        l = np.array(-C)
        u = np.array([np.inf, np.inf])
        if i==0:
            P = sparse.csc_matrix(H)
            A = sparse.csc_matrix(dC.T)
            prob.setup(P, q, A, l, u, verbose=False)
        else:
            prob.update(q=q, l=l, u=u,Px=H.flatten(),Ax=(dC.T).flatten())
        res = prob.solve()
        p = res.x
        eta = res.y


    if qpsolver == 'quadprog': ## solve using quadprog
        meq = 0
        result = quadprog.solve_qp(H,-dF,dC,-C,meq)
        p = result[0]
        eta = result[4]


    gradL = dF - dC @ eta
    nd = p.dot(gradL)
    V = np.round(Ffunc(z, w)-1e3*(1e3 + z[2])**2,2)

    str = 'SQP iter: {0:3d} | gamma = {1:1.4f} | Cost = {2:2.3e} | ' \
          'norm(grad L) = {3:2.3e} | grad = {4:2.3e} | ' \
          'min(C) = {5:2.3e}'.format(i,gamma,V,np.round(gradL.dot(gradL),2),np.abs(nd),np.min(C))
    print(str)

    if (np.abs(nd) < 1e-2) and np.min(C) > -1e-6 and gamma < 1e-3:
        break

    inv_mu = np.max(np.abs(eta)) + 10

    c = merit_function(z, w, inv_mu, gamma)
    cind = constraint(z, w, gamma)
    alpha = 1.0
    for k in range(52):
        ztest = z + alpha * p
        ctest = merit_function(ztest, w, inv_mu, gamma)
        cun = dC.T @ ztest + C
        # if np.isnan(ctest) or np.isinf(ctest): nan and inf checks should be redundant
        # if ctest < c:  # #
        if ctest < c + 0.01 * alpha * (nd - inv_mu * np.sum(np.minimum(cun,0))):    # check this is correct?
            z = ztest
            lams = device_put(eta)

            break

        alpha = alpha / 2

    if np.abs(nd) > old_nd*0.99:
        count += 1
    else:
        count = 0
    old_nd = np.abs(nd)

    if count >= 3:
        gamma = min(gamma * 2,0.1)
    else:
        gamma = max(gamma/2,0.99e-3)
    #
    #
u = jnp.expand_dims(z[:2],1)
x = u + w
print('true violation', np.mean(x[0,:] > upper_bound))
tf = time.time()
print('Time taken = ', tf-ts)
#
