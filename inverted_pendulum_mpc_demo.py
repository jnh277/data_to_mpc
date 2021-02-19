"""
Simulate a nonlinear inverted pendulum (QUBE SERVO 2)
with independent process and measurement noise

TODO: UPDATE THIS INFO

GOAL:
Use a Monte Carlo style approach to perform MPC where the state constraints are satisfied
with a given probability.

Current set up: Uses MC to give an expected cost and then satisfies,
Chance state constraints using a log barrier formulation
Chance state constraints using a log barrier formulation
Input constraints using a log barrier formulation

Implementation:
Uses custom newton method to solve
Uses JAX to compile and run code on GPU/CPU and provide gradients and hessians
"""

# general imports
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace, col_vec, row_vec, suppress_stdout_stderr
from pathlib import Path
import pickle
from tqdm import tqdm

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy


# Control parameters
z_star = np.array([[0],[np.pi],[0.0],[0.0]],dtype=float)        # desired set point in z1
Ns = 200             # number of samples we will use for MC MPC
Nh = 25              # horizonline of MPC algorithm
sqc_v = np.array([1,10.0,1e-5,1e-5],dtype=float)            # cost on state error
sqc = np.diag(sqc_v)
src = np.array([[0.001]])

# define the state constraints, (these need to be tuples)
state_constraints = (lambda z: 0.75*np.pi - z[[0],:,:],lambda z: z[[0],:,:] + 0.75*np.pi)
# define the input constraints
input_constraints = (lambda u: 18. - u, lambda u: u + 18.)


# simulation parameters
# TODO: WARNING DONT MAKE T > 100 due to size of saved inv_metric
T = 10             # number of time steps to simulate and record measurements for
Ts = 0.025
z1_0 = -np.pi/4            # initial states
z2_0 = np.pi/4
z3_0 = 0.0
z4_0 = 0.0

# got these values from running HMC on some real data
r1_true = 0.0011        # measurement noise standard deviation
r2_true = 0.001
r3_true = 0.175
q1_true = 3e-4       # process noise standard deviation
q2_true = 1e-4      # process noise standard deviation
q3_true = 0.013       # process noise standard deviation
q4_true = 0.013      # process noise standard deviation

# got these values from the data sheet
mr_true = 0.095 # kg
mp_true = 0.024 # kg
Lp_true = 0.129 # m
Lr_true = 0.085 # m
Jr_true = mr_true * Lr_true * Lr_true / 3 # kgm^2
Jp_true = mp_true * Lp_true * Lp_true / 3 # kgm^2
Km_true = 0.042 # Vs/rad / Nm/A
Rm_true = 8.4 # ohms
Dp_true = 5e-5 # Nms/rad
Dr_true = 1e-3 # Nms/rad
grav = 9.81

theta_true = {
        'Mp': mp_true,
        'Lp': Lp_true,
        'Lr': Lr_true,
        'Jr': Jr_true,
        'Jp': Jp_true,
        'Km': Km_true,
        'Rm': Rm_true,
        'Dp': Dp_true,
        'Dr': Dr_true,
        'g': grav,
        'h': Ts
    }
Nx = 4
Ny = 3
Nu = 1


## --------------------- FUNCTION DEFINITIONS ---------------------------------- ##
# ----------------- Simulate the system-------------------------------------------#
def fill_theta(t):
    Jr = t['Jr']
    Jp = t['Jp']
    Mp = t['Mp']
    Lr = t['Lr']
    Lp = t['Lp']
    g = t['g']

    t['Jr + Mp * Lr * Lr'] = Jr + Mp * Lr * Lr
    t['0.25 * Mp * Lp * Lp'] = 0.25 * Mp * Lp * Lp
    t['0.5 * Mp * Lp * Lr'] = 0.5 * Mp * Lp * Lr
    t['m22'] = Jp + 0.25 * Mp * Lp * Lp
    t['0.5 * Mp * Lp * g'] = 0.5 * Mp * Lp * g
    return t


def qube_gradient(xt, u, t):  # t is theta, this is for QUBE
    cos_xt1 = jnp.cos(xt[1, :])  # there are 5 of these
    sin_xt1 = jnp.sin(xt[1, :])  # there are 4 of these
    m11 = t['Jr + Mp * Lr * Lr'] + t['0.25 * Mp * Lp * Lp'] - t['0.25 * Mp * Lp * Lp'] * cos_xt1 * cos_xt1
    m12 = t['0.5 * Mp * Lp * Lr'] * cos_xt1
    m22 = t['m22']  # this should be a scalar anyway - can be vector
    sc = m11 * m22 - m12 * m12
    tau = (t['Km'] * (u[0] - t['Km'] * xt[2, :])) / t['Rm']  # u is a scalr
    d1 = tau - t['Dr'] * xt[2, :] - 2 * t['0.25 * Mp * Lp * Lp'] * sin_xt1 * cos_xt1 * xt[2, :] * xt[3, :] + t[
        '0.5 * Mp * Lp * Lr'] * sin_xt1 * xt[3, :] * xt[3, :]
    d2 = -t['Dp'] * xt[3, :] + t['0.25 * Mp * Lp * Lp'] * cos_xt1 * sin_xt1 * xt[2, :] * xt[2, :] - t[
        '0.5 * Mp * Lp * g'] * sin_xt1
    dx = jnp.zeros_like(xt)
    dx = index_update(dx, index[0, :], xt[2, :])
    dx = index_update(dx, index[1, :], xt[3, :])
    dx = index_update(dx, index[2, :], (m22 * d1 - m12 * d2) / sc)
    dx = index_update(dx, index[3, :], (m11 * d2 - m12 * d1) / sc)
    return dx


def rk4(xt, ut, theta):
    h = theta['h']
    k1 = qube_gradient(xt, ut, theta)
    k2 = qube_gradient(xt + k1 * h / 2, ut, theta)
    k3 = qube_gradient(xt + k2 * h / 2, ut, theta)
    k4 = qube_gradient(xt + k3 * h, ut, theta)
    return xt + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * h  # should handle a 2D x just fine


def pend_simulate(xt, u, w,
                  theta):  # w is expected to be 3D. xt is expected to be 2D. ut is expected to be 2d but also, should handle being a vector (3d)
    [Nx, Ns, Np1] = w.shape
    # if u.ndim == 2:  # conditional checks might slow jax down
    #     Nu = u.shape[1]
    # if Nu != Np1:
    #     print('wyd??')
    x = jnp.zeros((Nx, Ns, Np1 + 1))
    x = index_update(x, index[:, :, 0], xt)
    for ii in range(Np1):
        x = index_update(x, index[:, :, ii + 1], rk4(x[:, :, ii], u[:, ii], theta) + w[:, :,
                                                                                     ii])  # slicing creates 2d x into 3d x. Also, Np1 loop will consume all of w
    return x[:, :, 1:]  # return everything except xt


# compile cost and create gradient and hessian functions
sim = jit(pend_simulate)  # static argnums means it will recompile if N changes

# Pack true parameters
theta_true = fill_theta(theta_true)

# declare simulations variables
z_sim = np.zeros((Nx, 1, T+1), dtype=float) # state history
u = np.zeros((Nu, T+1), dtype=float)
w_sim = np.reshape(np.array([[0.],[q2_true],[q3_true],[q4_true]]),(4,1,1))*np.random.randn(4,1,T)
v = np.array([[r1_true],[r2_true],[r3_true]])*np.random.randn(3,T)
y = np.zeros((Ny, T), dtype=float)

# initial state
z_sim[0,:,0] = z1_0
z_sim[1,:,0] = z2_0
z_sim[2,:,0] = z3_0
z_sim[3,:,0] = z4_0


### hmc parameters and set up the hmc model
warmup = 0
chains = 4
iter = warmup + int(Ns/chains)
model_name = 'pendulum_diag'
path = 'stan/'
if Path(path+model_name+'.pkl').is_file():
    model = pickle.load(open(path+model_name+'.pkl', 'rb'))
else:
    model = pystan.StanModel(file=path+model_name+'.stan')
    with open(path+model_name+'.pkl', 'wb') as file:
        pickle.dump(model, file)

# load hmc warmup
inv_metric = pickle.load(open('stan_traces/inv_metric.pkl','rb'))
stepsize = pickle.load(open('stan_traces/step_size.pkl','rb'))
last_pos = pickle.load(open('stan_traces/last_pos.pkl','rb'))


# define MPC cost, gradient and hessian function
cost = jit(log_barrier_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
gradient = jit(grad(log_barrier_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(log_barrier_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))

mu = 1e4
gamma = 1
delta = 0.05
max_iter = 10000

# declare some variables for storing the ongoing resutls
xt_est_save = np.zeros((Ns, Nx, T))
theta_est_save = np.zeros((Ns, 6, T))
q_est_save = np.zeros((Ns, 4, T))
r_est_save = np.zeros((Ns, 3, T))
uc_save = np.zeros((1, Nh, T))
mpc_result_save = []

### SIMULATE SYSTEM AND PERFORM MPC CONTROL
for t in tqdm(range(T),desc='Simulating system, running hmc, calculating control'):
    # simulate system
    # apply u_t
    # this takes x_t to x_{t+1}
    # measure y_t
    z_sim[:,:,[t+1]] = sim(z_sim[:,:,t],u[:,[t]],w_sim[:,:,[t]],theta_true)
    y[:2, t] = z_sim[0:2, 0, t] + v[:2, t]
    y[2, t] = (u[0, t] - theta_true['Km'] * z_sim[2, 0, t]) / theta_true['Rm'] + v[2, t]


    # estimate system (estimates up to x_t)
    stan_data ={'no_obs': t+1,
                'Ts':Ts,
                'y': y[:, :t+1],
                'u': u[0, :t+1],
                'Lr':Lr_true,
                'Mp':mp_true,
                'Lp':Lp_true,
                'g':grav,
                'theta_p_mu':np.array([Jr_true, Jp_true, Km_true, Rm_true, Dp_true, Dr_true]),
                'theta_p_std':0.1*np.array([Jr_true, Jp_true, Km_true, Rm_true, Dp_true, Dr_true]),
                'r_p_mu': np.array([r1_true, r2_true, r3_true]),
                'r_p_std': 0.5*np.array([r1_true, r2_true, r3_true]),
                'q_p_mu': np.array([q1_true, q2_true, q3_true, q4_true]),
                'q_p_std': np.array([q1_true, q2_true, 0.5*q3_true, 0.5*q3_true]),
                }


    this_init = list(last_pos)
    this_inv_metric = inv_metric.copy()
    for cc in range(4):
        this_inv_metric[cc] = np.hstack((inv_metric[cc][0:t+2],
                                         inv_metric[cc][101:101+t+2],
                                         inv_metric[cc][202:202+t+2],
                                         inv_metric[cc][303:303+t+2],
                                         inv_metric[cc][-13:]))
        if t == 0:
            # this_init[cc]['h'] = np.array([[z1_0, z1_0], [z2_0, z2_0], [z3_0, z3_0], [z4_0, z4_0]])
            this_init[cc]['h'] = np.hstack((np.array([[z1_0], [z2_0], [z3_0], [z4_0]]),
                                            z_sim[:,0,[t+1]]))
        else:
            # this_init[cc]['h'] = np.hstack((this_init[cc]['h'], this_init[cc]['mu'][:,[-1]])) # does not work (diverges)
            this_init[cc]['h'] = z_sim[:, 0, :t+2]     # TODO: dont use sim truth
        del this_init[cc]['mu']
        del this_init[cc]['yhat']

    # with suppress_stdout_stderr():
    # fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains,
    #                      control=control,
    #                      init=this_init)
    control = {"adapt_delta": 0.85,
               "max_treedepth": 13,
               "stepsize": stepsize,
               "inv_metric": this_inv_metric,
               "adapt_engaged": False}
    with suppress_stdout_stderr():
        fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains,
                             control=control,
                             init=this_init)
    traces = fit.extract()
    last_pos = fit.get_last_position()

    theta = traces['theta']
    h = traces['h']
    z = h[:,:,:t+1]
    r_mpc = traces['r']
    q_mpc = traces['q']

    # parameter samples
    Jr_samps = theta[:, 0].squeeze()
    Jp_samps = theta[:, 1].squeeze()
    Km_samps = theta[:, 2].squeeze()
    Rm_samps = theta[:, 3].squeeze()
    Dp_samps = theta[:, 4].squeeze()
    Dr_samps = theta[:, 5].squeeze()

    theta_mpc = {
        'Mp': mp_true,
        'Lp': Lp_true,
        'Lr': Lr_true,
        'Jr': Jr_samps,
        'Jp': Jp_samps,
        'Km': Km_samps,
        'Rm': Rm_samps,
        'Dp': Dp_samps,
        'Dr': Dr_samps,
        'g': grav,
        'h': Ts
    }
    theta_mpc = fill_theta(theta_mpc)

    # current state samples (reshaped to [o,M])
    xt = z[:,:,-1].T
    # we also need to sample noise
    # TODO: Only update one step of the noise and reuse previous
    w_mpc = np.zeros((Nx, Ns, Nh + 1), dtype=float)
    w_mpc[0, :, :] = np.expand_dims(col_vec(q_mpc[:, 0]) * np.random.randn(Ns, Nh + 1),
                                    0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
    w_mpc[1, :, :] = np.expand_dims(col_vec(q_mpc[:, 1]) * np.random.randn(Ns, Nh + 1), 0)
    w_mpc[2, :, :] = np.expand_dims(col_vec(q_mpc[:, 2]) * np.random.randn(Ns, Nh + 1), 0)
    w_mpc[3, :, :] = np.expand_dims(col_vec(q_mpc[:, 3]) * np.random.randn(Ns, Nh + 1), 0)

    ut = u[:,[t]]  # control action that was just applied


    # save some things for later plotting
    xt_est_save[:, :, t] = z[:, :, -1]
    theta_est_save[:, :, t] = theta
    q_est_save[:, :, t] = q_mpc
    r_est_save[:, :, t] = r_mpc

    # calculate next control action
    result = solve_chance_logbarrier(np.zeros((1, Nh)), cost, gradient, hessian, ut, xt, theta_mpc, w_mpc, z_star, sqc,
                                     src,
                                     delta, pend_simulate, state_constraints, input_constraints, verbose=False,
                                     max_iter=max_iter)
    mpc_result_save.append(result)

    uc = result['uc']
    u[:,t+1] = uc[0,0]

    uc_save[0, :, t] = uc[0,:]


# plt.subplot(2,1,1)
# plt.plot(x,label='True', color='k')
# plt.plot(xt_est_save[0,:,:].mean(axis=0), color='b',label='mean')
# plt.plot(np.percentile(xt_est_save[0,:,:],97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(xt_est_save[0,:,:],2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
# plt.ylabel('x')
# plt.axhline(x_ub,linestyle='--',color='r',linewidth=2.,label='constraint')
# plt.axhline(x_star,linestyle='--',color='g',linewidth=2.,label='target')
# plt.legend()
#
# plt.subplot(2,1,2)
# plt.plot(u)
# plt.axhline(u_ub,linestyle='--',color='r',linewidth=2.,label='constraint')
# plt.ylabel('u')
# plt.xlabel('t')
#
# plt.tight_layout()
# plt.show()
#
# plt.subplot(2,2,1)
# plt.plot(a_est_save.mean(axis=0),label='mean')
# plt.plot(np.percentile(a_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(a_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
# plt.ylabel('a')
# plt.axhline(0.9,linestyle='--',color='k',linewidth=2.,label='true')
# plt.legend()
#
# plt.subplot(2,2,2)
# plt.plot(b_est_save.mean(axis=0),label='mean')
# plt.plot(np.percentile(b_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(b_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
# plt.ylabel('b')
# plt.axhline(0.1,linestyle='--',color='k',linewidth=2.,label='true')
# plt.legend()
#
# plt.subplot(2,2,3)
# plt.plot(q_est_save.mean(axis=0),label='mean')
# plt.plot(np.percentile(q_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(q_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
# plt.ylabel('q')
# plt.axhline(q_true,linestyle='--',color='k',linewidth=2.,label='true')
# plt.legend()
# plt.xlabel('t')
#
# plt.subplot(2,2,4)
# plt.plot(r_est_save.mean(axis=0),label='mean')
# plt.plot(np.percentile(r_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(r_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
# plt.ylabel('r')
# plt.axhline(r_true,linestyle='--',color='k',linewidth=2.,label='true')
# plt.legend()
# plt.xlabel('t')
#
# plt.show()
