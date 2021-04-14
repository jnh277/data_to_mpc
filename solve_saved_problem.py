# fixing multithreading on osx
import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")

# general imports
import pystan
import numpy as np
from helpers import col_vec, suppress_stdout_stderr
from pathlib import Path
import pickle
from tqdm import tqdm
import quadprog
import time
import matplotlib.pyplot as plt

# jax related imports
from jax.scipy.special import expit
import jax.numpy as jnp
from jax import grad, jit,  jacfwd, jacrev
from jax.ops import index, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy


# Control parameters
x_star = np.array([1.0])            # desired set point
M = 200                             # number of samples we will use for MC MPC
N = 10                              # horizonline of MPC algorithm
sqc = np.array([[1.0]])             # square root cost on state error
src = np.array([[0.01]])             # square root cost on control action
delta = 0.05                        # desired maximum probability of not satisfying the constraint

x_ub = 1.2
u_ub = 2.
state_constraints = (lambda x: x_ub - x,)
input_constraints = (lambda u: u_ub - u,)


# simulation parameters
T = 15              # number of time steps to simulate and record measurements for
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

# define MPC cost, gradient and hessian function
# this is the stuff for the log barrier approach
cost_lb = jit(log_barrier_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
gradient_lb = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
hessian_lb = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))

# THIS IS FUNCTION DEFINITIONS FOR THE SQP APPROACH
def simulate_wrapper(uc_bar, xt, ut, w, theta, simulate_func, Nu, N):

    uc = jnp.reshape(uc_bar, (Nu, N))  # control input variables  #,
    u = jnp.hstack([ut, uc])  # u_t was already performed, so uc is the next N control actions
    x = simulate_func(xt, u, w, theta)
    xbar = x.flatten()
    return xbar


def cost(xbar, uc_bar, x_star, sqc, src, o, M, N, Nu):
    x = jnp.reshape(xbar,(o,M,N+1))
    uc = jnp.reshape(uc_bar, (Nu, N))  # control input variables  #,
    V = jnp.sum(jnp.matmul(sqc, jnp.reshape(x[:, :, 1:] - jnp.reshape(x_star, (o, 1, -1)), (o, -1))) ** 2) \
        + 0.5*jnp.sum(jnp.matmul(src, uc) ** 2)
    return V


def cost_epsilon(epsilon):
    return 0.5*jnp.sum(300 * (epsilon + 1e3) ** 2)


def grad_cost_epsilon(epsilon):
    return 300*(epsilon + 1e3)


def hess_cost_epsilon(N, ncx):
    return 300*jnp.eye(N*ncx)


def chance_constraint(hu, epsilon, gamma):    # Pr(h(u) >= 0 ) >= (1-epsilon)
    return jnp.mean(expit(hu / gamma), axis=1) - (1 - epsilon)     # take the sum over the samples (M)


def state_constraint_wrapper(xbar, epsilon, gamma, state_constraints, o, M, N):
    x = jnp.reshape(xbar, (o, M, N + 1))
    hx = jnp.concatenate([state_constraint(x[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = chance_constraint(hx, epsilon, gamma).flatten()
    return cx


def input_constraint_wrapper(ubar, input_constraints, N, Nu):
    uc = jnp.reshape(ubar, (Nu, N))  # control input variables  #
    cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints], axis=1).flatten()
    return cu


# todo: constraints on epsilon???

# THIS IS COMPILED FUNCTION AND DERIVATIVES
sim_wrap = jit(simulate_wrapper, static_argnums=(5, 6, 7))      # computes xbar given ubar
dxdu_func = jit(jacfwd(simulate_wrapper, argnums=0),static_argnums=(5, 6, 7))   # computes dxdu
d2xdu2_func = jit(jacfwd(jacrev(simulate_wrapper, argnums=0)),static_argnums=(5, 6, 7))
dVdxu_func = jit(grad(cost, argnums=(0,1)), static_argnums=(5, 6, 7, 8))
d2Vdxu2_func = jit(jacfwd(jacrev(cost, argnums=(0,1)),argnums=(0,1)),static_argnums=(5, 6, 7, 8))
Cx_func = jit(state_constraint_wrapper, static_argnums=(3, 4, 5, 6))
Cu_func = jit(input_constraint_wrapper, static_argnums=(1, 2, 3))
dCxdxepsfunc = jit(jacfwd(state_constraint_wrapper, argnums=(0,1)), static_argnums=(3, 4, 5, 6))
dCudufunc = jit(jacfwd(input_constraint_wrapper, argnums=0), static_argnums=(1, 2, 3))
d2Cxdx2func = jit(jacfwd(jacrev(state_constraint_wrapper,argnums=0),argnums=0), static_argnums=(3, 4, 5, 6))
d2Cudu2func = jit(jacfwd(jacrev(input_constraint_wrapper, argnums=0), argnums=0), static_argnums=(1, 2, 3))

def build_SQP_parts(z, lams, gamma, sqc, src, o, Nu, N, ncx, ncu):
    # Todo: push things to jax and pull things from jax properly
    ubar = z[:Nu*N]
    epsilon = z[Nu*N:N*Nu+ncx*N]                        # slack variables on state constraints
    # first calculate components
    xbar = sim_wrap(ubar, xt, ut, w, theta, simulate, Nu, N)  # The simulated x trajectory
    dxdu = dxdu_func(ubar, xt, ut, w, theta, simulate, Nu, N)  # gradient trajectory wrt uc
    d2xdu2 = d2xdu2_func(ubar, xt, ut, w, theta, simulate, Nu, N)
    dVdxu = dVdxu_func(xbar, ubar, x_star, sqc, src, o, M, N, Nu)
    d2Vdxu2 = d2Vdxu2_func(xbar, ubar, x_star, sqc, src, o, M, N, Nu)
    dCxdxeps = dCxdxepsfunc(xbar, epsilon, gamma, state_constraints, o, M, N)
    dCudu = dCudufunc(ubar, input_constraints, N, Nu)
    d2Cxdx2 = d2Cxdx2func(xbar, epsilon, gamma, state_constraints, o, M, N)
    d2Cudu2 = d2Cudu2func(ubar, input_constraints, N, Nu)
    Cx = Cx_func(xbar, epsilon, gamma, state_constraints, o, M, N)
    Cu = Cu_func(ubar, input_constraints, N, Nu)
    C = np.hstack([Cx, Cu, epsilon - 0.05])     # todo, derivative of constraints w.r.t epsilon

    # then combine components
    # put together derivative of cost with respect to z
    dVdu = dVdxu[0] @ dxdu + dVdxu[1]
    dV = np.hstack([dVdu, grad_cost_epsilon(epsilon)])

    # put together derivative of constraints with respect to z
    dCx = np.concatenate([dCxdxeps[0] @ dxdu, dCxdxeps[1]], axis=1)
    dCu = np.hstack([dCudu, np.zeros((ncu * N, ncx * N))])
    dCeps = np.hstack([np.zeros((ncx*N,Nu * N)),np.eye(ncx * N)])
    dC = np.vstack([dCx, dCu, dCeps])

    # second derivative of second half of lagrangian (constraint part)
    d2lamC_11 = dxdu.T @ np.sum(np.reshape(lams[:ncx * N], (-1, 1, 1)) * d2Cxdx2, axis=0) @ dxdu + (
                lams[:ncx * N] @ dCxdxeps[0]) @ d2xdu2.transpose((2, 0, 1)) + lams[ncx * N:ncx * N+N*ncu] @ d2Cudu2
    d2lamC = np.vstack((np.hstack((d2lamC_11, np.zeros((N, N * ncx)))), np.zeros((N * ncx, N + N * ncx))))

    # second derivative of first half of lagrangian
    ddVu = d2Vdxu2[1][1] + d2xdu2.T @ dVdxu[0] + dxdu.T @ d2Vdxu2[0][0] @ dxdu
    ddVeps = hess_cost_epsilon(N, ncx)
    ddV = np.zeros((N * Nu + N * ncx, N * Nu + N * ncx))
    ddV[:N * Nu, :N * Nu] = ddVu
    ddV[N * Nu:, N * Nu:] = ddVeps

    # combined second derivative of lagrangian
    H = ddV - d2lamC

    return H, dV, C, dC, xbar

def merit_function(z, inv_mu, theta, sqc, src, x_star, o, Nu, N, ncx):
    ubar = z[:Nu*N]
    epsilon = z[Nu*N:N*Nu+ncx*N]                        # slack variables on state constraints
    xbar = sim_wrap(ubar, xt, ut, w, theta, simulate, Nu, N)  # The simulated x trajectory
    V = cost(xbar, ubar, x_star, sqc, src, o, M, N, Nu) + cost_epsilon(epsilon)
    Cx = Cx_func(xbar, epsilon, gamma, state_constraints, o, M, N)
    Cu = Cu_func(ubar, input_constraints, N, Nu)
    C = np.hstack([Cx,Cu, epsilon - 0.05])
    return V - inv_mu * np.sum(np.minimum(C, 0))

# def merit_function(z, w, inv_mu, gamma):
#     return cost(z, w) - inv_mu * np.sum(np.minimum(constraint(z,w,gamma), 0))

lb_times = []
sqp_times = []


## LOADING AND SOLVING PROBLEM FROM HERE

with open('results/sqp_debug/saved_optims.pkl','rb') as file:
    saved_problems = pickle.load(file)

ind = 3
problem = saved_problems[ind]

theta = problem['problem_data']['theta']
xt = problem['problem_data']['xt']
w = problem['problem_data']['w']
ut = problem['problem_data']['ut']


# SOLVE USING SQP
# todo: should also be pushing stuff onto jax properly here
uc0 = np.zeros((1, N))
o,N = w.shape[0],w.shape[2]-1
Nu = uc0.shape[0]             # input dimension
ncu = len(input_constraints)  # number of input constraints
ncx = len(state_constraints)  # number of state constraints
epsilon0 = 1.0
lams = 1e6 * np.ones((ncx * N + ncu * N + ncx*N,))      # second ncx*N is for constraints placed on epsilon


gamma = 1
z = np.hstack([uc0.flatten(), epsilon0 * np.ones((ncx * N,))])


count = 0
old_nd = 1e6
c1 = 0.01
subproblem_save = []
t2s = time.time()
for i in range(30):
    # solve sqp subproblem
    H, dV, C, dC, xbar = build_SQP_parts(z, lams, gamma, sqc, src, o, Nu, N, ncx, ncu)


    subproblem = {'H':H,
                  'dV':dV,
                  'C':C,
                  'dC':dC,
                  'xbar':xbar,
                  'gamma':gamma}

    if len(state_constraints) > 0:
        x_mpc = jnp.reshape(xbar, (o, M, N + 1))
        hx = np.concatenate([state_constraint(x_mpc[:, :, 1:]) for state_constraint in state_constraints], axis=2)
        cx = 1 - np.mean(hx > 0, axis=1)
    else:
        cx = 0

    [d, v] = np.linalg.eig(H)
    min_d = np.min(d)
    if (d < 1e-5).any():
        Hold = 1.0 * H
        if np.iscomplex(d).any():  # stop complex eigenvalues
            d = np.real(d)
            v = np.real(v)
        ind = d < 1e-5
        d[ind] = 1e-5 + np.abs(d[ind])
        H = v @ np.diag(d) @ v.T

    meq = 0
    result = quadprog.solve_qp(H, -dV, dC.T, -C, meq)
    p = result[0]
    eta = result[4]

    gradL = dV - eta @ dC
    nd = p.dot(gradL)

    V = cost(xbar, z[:Nu*N], x_star, sqc, src, o, M, N, Nu)

    subproblem['p'] = p
    subproblem['eta'] = eta
    subproblem['gradL'] = gradL
    subproblem['cost'] = V
    subproblem['nd'] = nd
    subproblem['C'] = C
    subproblem['cx'] = cx

    str = 'SQP iter: {0:3d} | gamma = {1:1.4f} | Cost = {2:2.3e} | ' \
          'norm(grad L) = {3:2.3e} | grad = {4:2.3e} | ' \
          'min(C) = {5:2.3e} | ' \
          'max % cx < 0 = {6:1.3f} | ' \
        'min eig H {7:2.3e}'.format(i, gamma, V, np.round(gradL.dot(gradL), 2), np.abs(nd), np.min(C[ncx*N:]), np.max(cx), min_d)
    print(str)

    if (np.abs(nd) < 1e-2) and np.min(C[ncx*N:]) > -1e-6 and (cx < 0.0501).all and gamma < 1e-3:    # todo: remove hardcoding of delta
        break

    inv_mu = np.max(np.abs(eta)) + 10

    c = merit_function(z, inv_mu, theta, sqc, src, x_star, o, Nu, N, ncx)
    # cind = constraint(z, w, gamma)
    alpha = 1.0
    for k in range(52):
        ztest = z + alpha * p
        ctest = merit_function(ztest, inv_mu, theta, sqc, src, x_star, o, Nu, N, ncx)
        cun = dC @ ztest + C

        if len(state_constraints) > 0:
            xbar_test = sim_wrap(ztest[:Nu * N], xt, ut, w, theta, simulate, Nu, N)  # The simulated x trajectory
            x_mpc_test = jnp.reshape(xbar_test, (o, M, N + 1))
            hx = np.concatenate([state_constraint(x_mpc_test[:, :, 1:]) for state_constraint in state_constraints],
                                axis=2)
            cxtest = 1 - np.mean(hx > 0, axis=1)
        else:
            cx = 0


        # if ctest < c:  # #
        if ctest < c + c1 * alpha * (nd - inv_mu * np.sum(np.minimum(cun, 0))):  # check this is correct?
            # if ctest < c + c1 * alpha * (nd - inv_mu * np.sum(np.minimum(cun, 0))) and ((cxtest <= cx) + (cxtest < 0.0501)).all():  # check this is correct?
            z = ztest
            # lams = device_put(eta)
            lams = eta
            break

        alpha = alpha / 2

    subproblem['alpha'] = alpha
    subproblem_save.append(subproblem)

    # always decrease by 2
    gamma = max(gamma / 2, 0.99e-3)

    # below is for decreasing only when newton decrement is small
    # if (np.abs(nd) < 1e0):
    #     if count > 0:
    #         gamma = max(gamma / 5, 0.99e-3)
    #     else:
    #         gamma = max(gamma / 2, 0.99e-3)
    #     count += 1
    # else:
    #     count=0

    # below is to try to increase if things stop working so well (sometimes helps...)
    # if np.abs(nd) > old_nd * 0.99 and nd > 1e-2:
    #     count += 1
    # else:
    #     count = 0
    # old_nd = np.abs(nd)
    #
    # if count >= 3:
    #     gamma = 1.0
    # else:
    #     gamma = max(gamma / 1.5, 0.99e-3)










## load using
# with open('results/sqp_debug/cx_constrained_saved_optims.pkl','rb') as file:
#     saved_optims = pickle.load(file)

