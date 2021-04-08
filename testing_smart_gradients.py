# general imports
import pystan
import numpy as np
from helpers import col_vec, suppress_stdout_stderr
from pathlib import Path
import pickle
from jax.scipy.special import expit
from tqdm import tqdm

# jax related imports
import jax.numpy as jnp
from jax import grad, jit,  jacfwd, jacrev
from jax.ops import index, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy


# Control parameters
x_star = np.array([1.0])        # desired set point
M = 200                             # number of samples we will use for MC MPC
N = 5                              # horizonline of MPC algorithm
sqc = np.array([[1.0]])             # square root cost on state error
src = np.array([[0.5]])             # square root cost on control action
delta = 0.05                        # desired maximum probability of not satisfying the constraint

x_ub = 1.2
u_ub = 2.
state_constraints = (lambda x: x_ub - x,)
input_constraints = (lambda u: u_ub - u,)


# simulation parameters
T = 30              # number of time steps to simulate and record measurements for
x0 = 0.5            # initial time step
r_true = 0.01       # measurement noise standard deviation
q_true = 0.05       # process noise standard deviation

#----------------- Simulate the system-------------------------------------------#
def ssm(x, u, a=0.9, b=0.1):
    return a*x + b*u

x = np.zeros(T+1)
x[0] = x0                                   # initial state
w = np.random.normal(0.0, q_true, T+1)        # make a point of predrawing noise
y = np.zeros((T,))

# create some inputs that are random but held for 10 time steps
u = np.zeros((T+1,))     # first control action will be zero

### hmc parameters and set up the hmc model
warmup = 1000
chains = 4
iter = warmup + int(M/chains)
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




xt_est_save = np.zeros((1,M,T))
a_est_save = np.zeros((M,T))
b_est_save = np.zeros((M,T))
q_est_save = np.zeros((M,T))
r_est_save = np.zeros((M,T))
mpc_result_save = []
### SIMULATE SYSTEM AND PERFORM MPC CONTROL
# for t in tqdm(range(T),desc='Simulating system, running hmc, calculating control'):

t = 0       # look at just one timestep

# simulate system
# apply u_t
# this takes x_t to x_{t+1}
# measure y_t
x[t+1] = ssm(x[t], u[t]) + q_true * np.random.randn()
y[t] = x[t] + r_true * np.random.randn()

# estimate system (estimates up to x_t)
stan_data = {
    'N': t+1,
    'y': y[:t+1],
    'u': u[:t+1],
    'prior_mu': np.array([0.8, 0.05, 0.1, 0.1]),
    'prior_std': np.array([0.2, 0.2, 0.2, 0.2]),
    'prior_state_mu': 0.3,
    'prior_state_std': 0.2,
}
# with suppress_stdout_stderr():
fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains)
traces = fit.extract()

# state samples
z = traces['z']

# parameter samples
a = traces['a']
b = traces['b']
r = traces['r']
q = traces['q']

# current state samples (expanded to [o,M]
xt = np.expand_dims(z[:, -1], 0)  # inferred state for current time step
# we also need to sample noise
w = np.expand_dims(col_vec(q) * np.random.randn(M, N + 1), 0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
ut = np.expand_dims(np.array([u[t]]), 0)  # control action that was just applied
theta = {'a': a,
         'b': b, }


Nu = 1
uc = np.ones((Nu, N))
ncx = 1
o = 1
epsilon = np.ones((N * ncx,))
z = np.hstack([uc.flatten(), epsilon])
x_star = 2.*np.ones((o,1))

""" Things we wish to replicate """
def complete_cost(z, xt, ut, w, theta, x_star,sqc,src, Nu, N, ncx, o):
    uc = jnp.reshape(z[:Nu*N], (Nu, N))                 # control input variables  #,
    epsilon = z[Nu*N:N*Nu+ncx*N]                        # slack variables on state constraints

    u = jnp.hstack([ut, uc]) # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)
    V = jnp.sum(jnp.matmul(sqc, jnp.reshape(x[:, :, 1:] - jnp.reshape(x_star, (o, 1, -1)), (o, -1))) ** 2) \
        + 0.5*jnp.sum(jnp.matmul(src, uc) ** 2) + 0.5*jnp.sum(300 * (epsilon + 1e3) ** 2)
    return V

V = complete_cost(z, xt, ut, w, theta, x_star, sqc, src, Nu, N, ncx, o)

# compute dVdz
grad_complete_cost = grad(complete_cost, argnums=0)
dVdz = grad_complete_cost(z, xt, ut, w, theta, x_star, sqc, src, Nu, N, ncx, o)
dVdu = dVdz[:N*Nu]

# hessian
hessian_complete_cost = jacfwd(jacrev(complete_cost, argnums=0))
H = hessian_complete_cost(z, xt, ut, w, theta, x_star, sqc, src, Nu, N, ncx, o)
Hu = H[:N*Nu,:N*Nu]

""" how we are going to replicate them by breaking them up """
# create a wrapper around simulate that adds in u_t and takes flat uc
def simulate_wrapper(uc_bar, xt, ut, w, theta, Nu, N):
    uc = jnp.reshape(uc_bar, (Nu, N))  # control input variables  #,
    u = jnp.hstack([ut, uc])  # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)
    xbar = x.flatten()
    return xbar

dxdu_func = jacfwd(simulate_wrapper, argnums=0)
# dxdu_func = jit(jacfwd(simulate_wrapper, argnums=0),static_argnums=(5,6))

xbar = simulate_wrapper(uc.flatten(), xt, ut, w, theta, Nu, N)      # The simulated x trajectory
dxdu = dxdu_func(uc.flatten(), xt, ut, w, theta, Nu, N)             # gradient trajectory wrt uc

def cost(xbar, uc_bar, x_star, sqc, src, o,M,N,Nu):
    x = jnp.reshape(xbar,(o,M,N+1))
    uc = jnp.reshape(uc_bar, (Nu, N))  # control input variables  #,
    V = jnp.sum(jnp.matmul(sqc, jnp.reshape(x[:, :, 1:] - jnp.reshape(x_star, (o, 1, -1)), (o, -1))) ** 2) \
        + 0.5*jnp.sum(jnp.matmul(src, uc) ** 2)

    return V

def cost_epsilon(epsilon):
    return 0.5*jnp.sum(300 * (epsilon + 1e3) ** 2)

def grad_cost_epsilon(epsilon):
    return 300*(epsilon + 1e3)

def hess_cost_epsilon(N,ncx):
    return 300*np.eye(N*ncx)

V2 = cost(xbar, uc.flatten(),x_star, sqc, src, o,M,N,Nu) + cost_epsilon(epsilon)

dVdxu_func = grad(cost, argnums=(0,1))
dVdxu = dVdxu_func(xbar, uc.flatten(), x_star, sqc, src, o,M,N,Nu)

# d cost with respect to u
dVdu_2 = dVdxu[0] @ dxdu + dVdxu[1]

# d cost with respect to u and eps
dVdz_2 = np.hstack([dVdu, grad_cost_epsilon(epsilon)])

# hessian
d2xdu2_func = jacfwd(jacrev(simulate_wrapper, argnums=0))
d2xdu2 = d2xdu2_func(uc.flatten(), xt, ut, w, theta, Nu, N)
d2Vdxu2_func = jacfwd(jacrev(cost, argnums=(0,1)),argnums=(0,1))
d2Vdxu2 = d2Vdxu2_func(xbar, uc.flatten(), x_star, sqc, src, o,M,N,Nu)

# d2Vdu2p_func = jacfwd(jacrev(cost, argnums=(0)))
# d2Vd = d2Vdx2_func(xbar, uc.flatten(), x_star, sqc, src, o,M,N,Nu)

# dd cost with respect to u
Hu_2 = d2Vdxu2[1][1] + d2xdu2.T @ dVdxu[0] + dxdu.T @ d2Vdxu2[0][0] @ dxdu

# dd cost with respect to eps
Heps_2 = hess_cost_epsilon(N, ncx)

# dd cost with respect to u and eps
H_2 = np.zeros((N*Nu+N*ncx,N*Nu+N*ncx))
H_2[:N*Nu,:N*Nu] = Hu_2
H_2[N*Nu:,N*Nu:] = Heps_2

# gradients of constraints
gamma = 0.1

def chance_constraint(hu, epsilon, gamma):    # Pr(h(u) >= 0 ) >= (1-epsilon)
    return jnp.mean(expit(hu / gamma), axis=1) - (1 - epsilon)     # take the sum over the samples (M)
#
# simulate and calculate state constraint with respect to u and epsilon
def complete_constraint_calc(z, xt, ut, w, theta, gamma, state_constraints, Nu, N, ncx):
    uc = jnp.reshape(z[:Nu * N], (Nu, N))  # control input variables  #,
    epsilon = z[Nu * N:N * Nu + ncx * N]  # slack variables on state constraints

    u = jnp.hstack([ut, uc])  # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)

    hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = chance_constraint(hx, epsilon, gamma).flatten()
    return cx

dCfunc = jacfwd(complete_constraint_calc,argnums=0)

C_check = complete_constraint_calc(z, xt, ut, w, theta, gamma, state_constraints, Nu, N, ncx)
dC_check = dCfunc(z, xt, ut, w, theta, gamma, state_constraints, Nu, N, ncx)

def state_constraint_wrapper(xbar, epsilon, gamma, state_constraints, o, M, N):
    x = jnp.reshape(xbar, (o, M, N + 1))
    hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = chance_constraint(hx, epsilon, gamma).flatten()
    return cx

dCdxepsfunc = jacfwd(state_constraint_wrapper, argnums=(0,1))

dCdxeps = dCdxepsfunc(xbar, epsilon, gamma, state_constraints, o, M, N)

dC = np.concatenate([dCdxeps[0] @ dxdu, dCdxeps[1]],axis=1)

np.max(np.abs(dC - dC_check))

# Lagrangian and its second derivative
def complete_lagrangian(z, lams, xt, ut, w, theta, x_star, sqc, src, state_constraints, Nu, N, ncx, o):
    uc = jnp.reshape(z[:Nu * N], (Nu, N))  # control input variables  #,
    epsilon = z[Nu * N:N * Nu + ncx * N]  # slack variables on state constraints

    u = jnp.hstack([ut, uc])  # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)
    V = jnp.sum(jnp.matmul(sqc, jnp.reshape(x[:, :, 1:] - jnp.reshape(x_star, (o, 1, -1)), (o, -1))) ** 2) \
        + 0.5 * jnp.sum(jnp.matmul(src, uc) ** 2) + 0.5 * jnp.sum(300 * (epsilon + 1e3) ** 2)

    hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = chance_constraint(hx, epsilon, gamma).flatten()

    L = V - np.sum(lams * cx)
    return L

lams = np.ones((5,))



L = complete_lagrangian(z, lams, xt, ut, w, theta, x_star, sqc, src, state_constraints, Nu, N, ncx, o)

def lams_constraints(z, lams, xt, ut, w, theta, state_constraints, Nu, N, ncx):
    uc = jnp.reshape(z[:Nu * N], (Nu, N))  # control input variables  #,
    epsilon = z[Nu * N:N * Nu + ncx * N]  # slack variables on state constraints

    u = jnp.hstack([ut, uc])  # u_t was already performed, so uc is the next N control actions
    x = simulate(xt, u, w, theta)

    hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = chance_constraint(hx, epsilon, gamma).flatten()
    return np.sum(lams * cx)


ddlamCfunc = jacfwd(jacrev(lams_constraints,argnums=0))

d2lamC_check = ddlamCfunc(z, lams, xt, ut, w, theta, state_constraints, Nu, N, ncx)

d2Cdx2func = jacfwd(jacrev(state_constraint_wrapper,argnums=0),argnums=0)

d2Cdx2 = d2Cdx2func(xbar, epsilon, gamma, state_constraints, o, M, N)

test = dxdu.T @ np.sum(np.reshape(lams,(-1,1,1))*d2Cdx2,axis=0) @ dxdu
test2 = (lams @ dCdxeps[0]) @ d2xdu2.transpose((2,0,1))
d2lamC_11 = dxdu.T @ np.sum(np.reshape(lams,(-1,1,1))*d2Cdx2,axis=0) @ dxdu + (lams @ dCdxeps[0]) @ d2xdu2.transpose((2,0,1))
d2lamC = np.vstack((np.hstack((d2lamC_11,np.zeros((N,N*ncx)))),np.zeros((N*ncx,N+N*ncx))))

# now check lagrangian double deriv
Hfunc = jacfwd(jacrev(complete_lagrangian,argnums=0),argnums=0)
HL_check = Hfunc(z, lams, xt, ut, w, theta, x_star, sqc, src, state_constraints, Nu, N, ncx, o)

HL = H_2 - d2lamC


# define MPC cost, gradient and hessian function
# cost = jit(log_barrier_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
# gradient = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
# hessian = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))