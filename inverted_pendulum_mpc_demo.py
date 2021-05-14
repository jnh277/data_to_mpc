###############################################################################
#    Data to Controller for Nonlinear Systems: An Approximate Solution
#    Copyright (C) 2021  Johannes Hendriks < johannes.hendriks@newcastle.edu.a >
#    and James Holdsworth < james.holdsworth@newcastle.edu.au >
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

""" This script runs a version of the rotary inverted pendulum example WITHOUT damping mismatch
and saves the results the results in rotary_inverted_pendulum_demo_results"""
""" This script will take a fair amount of time to run if you have not installed cuda enabled JAX,
    presaved results are included and can be plotted without running this script """
""" The results can then be plotted using the script 'plot_pendulum_results.py' after changing the file to be loaded """

# general imports
import pystan
import numpy as np
from helpers import col_vec, suppress_stdout_stderr
from pathlib import Path
import pickle
from tqdm import tqdm

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev
from jax.lax import scan
from jax.ops import index, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import solve_chance_logbarrier, log_barrier_cosine_cost

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy


# Control parameters
z_star = np.array([[0],[np.pi],[0.0],[0.0]],dtype=float)        # desired set point in z1
Ns = 200             # number of samples we will use for MC MPC
Nh = 25              # horizonline of MPC algorithm
sqc_v = np.array([1,30.,1e-5,1e-5],dtype=float) # cost on state error
sqc = np.diag(sqc_v)
src = np.array([[0.001]])

# define the state constraints, (these need to be tuples)
state_bound = 0.75*np.pi
input_bound = 18.0
state_constraints = (lambda z: state_bound - z[[0],:,:],lambda z: z[[0],:,:] + state_bound)
# define the input constraints
input_constraints = (lambda u: input_bound - u, lambda u: u + input_bound)


# simulation parameters
# WARNING DONT MAKE T > 100 due to size of saved inv_metric
T = 50             # number of time steps to simulate and record measurements for
Ts = 0.025
z1_0 = 0.0
z2_0 = 0.001
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
    x = jnp.zeros((Nx, Ns, Np1 + 1))
    x = index_update(x, index[:, :, 0], xt)
    iis = jnp.arange(Np1)
    dict = {
        'x':x,
        'u':u,
        'w':w,
        'theta':theta
    }
    dict,_ = scan(scan_func,dict,iis) # carry, ys = scan(scan_func,dict,iis). for each ii in iis, runs dict,y = scan_func(dict,ii) and ys = ys.append(y)
    return dict['x'][:, :, 1:]  # return everything except xt

def scan_func(carry,ii):
    carry['x'] = index_update(carry['x'], index[:, :, ii + 1], rk4(carry['x'][:, :, ii], carry['u'][:, ii], carry['theta']) + carry['w'][:, :, ii])
    return carry, []

sim = jit(pend_simulate)

# Pack true parameters
theta_true = fill_theta(theta_true)

# declare simulations variables
z_sim = np.zeros((Nx, 1, T+1), dtype=float) # state history
u = np.zeros((Nu, T+1), dtype=float)
# for i in range(int(T/5)):
#     u[0,i*5:] = np.random.uniform(-18,18,(1,))
w_sim = np.reshape(np.array([[q1_true],[q2_true],[q3_true],[q4_true]]),(4,1,1))*np.random.randn(4,1,T)
v = np.array([[r1_true],[r2_true],[r3_true]])*np.random.randn(3,T)
y = np.zeros((Ny, T), dtype=float)

# initial state
z_sim[0,:,0] = z1_0
z_sim[1,:,0] = z2_0
z_sim[2,:,0] = z3_0
z_sim[3,:,0] = z4_0


### hmc parameters and set up the hmc model
warmup = 300
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
cost = jit(log_barrier_cosine_cost, static_argnums=(11,12,13, 14, 15))  # static argnums means it will recompile if N changes
gradient = jit(grad(log_barrier_cosine_cost, argnums=0), static_argnums=(11, 12, 13, 14, 15))    # get compiled function to return gradients with respect to z (uc, s)
hessian = jit(jacfwd(jacrev(log_barrier_cosine_cost, argnums=0)), static_argnums=(11, 12, 13, 14, 15))


mu = 1e4
gamma = 1
delta = 0.05
max_iter = 5000

# declare some variables for storing the ongoing resutls
xt_est_save = np.zeros((Ns, Nx, T))
theta_est_save = np.zeros((Ns, 6, T))
q_est_save = np.zeros((Ns, 4, T))
r_est_save = np.zeros((Ns, 3, T))
uc_save = np.zeros((1, Nh, T))
mpc_result_save = []
hmc_traces_save = []

# predefine the mean of the prior so it doesnt change from run to run
theta_p_mu = np.array([1.1*Jr_true, 1.1*Jp_true, 1.1*Km_true,
                       1.1*Rm_true, 1.1*Dp_true, 1.1*Dr_true])
u[0,0] = 0.

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
                'theta_p_mu':theta_p_mu,
                'theta_p_std':0.2 * np.array([Jr_true, Jp_true, Km_true, Rm_true, Dp_true, Dr_true]),
                'r_p_mu': np.array([r1_true, r2_true, r3_true]),
                'r_p_std': 0.5*np.array([r1_true, r2_true, r3_true]),
                'q_p_mu': np.array([q1_true, q2_true, q3_true, q4_true]),
                'q_p_std': np.array([q1_true, q2_true, 0.5*q3_true, 0.5*q3_true]),
                }

    inv_metric = pickle.load(open('stan_traces/inv_metric.pkl', 'rb'))
    this_inv_metric = inv_metric.copy()
    for cc in range(4):
        # this_inv_metric[cc] = np.hstack((inv_metric[cc][0:t+2],
        #                                  inv_metric[cc][101:101+t+2],
        #                                  inv_metric[cc][202:202+t+2],
        #                                  inv_metric[cc][303:303+t+2],
        #                                  inv_metric[cc][-13:]))
        this_inv_metric[cc] = np.hstack((inv_metric[cc][0:((t+2)*4)],
                                         inv_metric[cc][-13:]))
    if t == 0:
        z_init = np.hstack((np.array([[z1_0], [z2_0], [z3_0], [z4_0]]),
                                        z_sim[:,0,[t+1]]))
    else:
        z_init = np.zeros((4, t + 2))
        z_init[0, :-1] = y[0, :t+1]
        z_init[1, :-1] = y[1, :t+1]
        z_init[0, -1] = y[0, t]  # repeat last entry
        z_init[1, -1] = y[1, t]  # repeat last entry
        z_init[2, :-1] = np.gradient(y[0, :t+1]) / Ts
        z_init[2, -1] = z_init[2, -2]
        z_init[3, :-1] = np.gradient(y[1, :t+1]) / Ts
        z_init[3, -1] = z_init[3, -2]

    # chain initialisation
    def init_function(ind):
        output = dict(theta=last_pos[ind]['theta'],
                      h=z_init,
                      q=last_pos[ind]['q'],
                      r=last_pos[ind]['r']
                      )
        return output
    init = [init_function(0),init_function(1),init_function(2),init_function(3)]

    control = {"adapt_delta": 0.85,
               "max_treedepth": 13,
               "stepsize": stepsize,
               "inv_metric": this_inv_metric,
               "adapt_engaged": True}
    with suppress_stdout_stderr():
        fit = model.sampling(data=stan_data, warmup=warmup, iter=iter, chains=chains,
                                 control=control,
                                 init=init)
    traces = fit.extract()
    hmc_traces_save.append(traces)

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
    if t > 0:
        uc0 = np.hstack((uc[[0],1:],np.reshape(uc[0,-1],(1,1))))
        mu = 200
        gamma = 0.2
        result = solve_chance_logbarrier(uc0, cost, gradient, hessian, ut, xt, theta_mpc, w_mpc, z_star, sqc,
                                         src,
                                         delta, pend_simulate, state_constraints, input_constraints, verbose=False,
                                         max_iter=max_iter,mu=mu,gamma=gamma)
    else:
        result = solve_chance_logbarrier(np.zeros((1, Nh)), cost, gradient, hessian, ut, xt, theta_mpc, w_mpc, z_star, sqc,
                                         src,
                                         delta, pend_simulate, state_constraints, input_constraints, verbose=False,
                                         max_iter=max_iter)
    mpc_result_save.append(result)

    uc = result['uc']
    u[:,t+1] = uc[0,0]

    uc_save[0, :, t] = uc[0,:]



# run = 'rotary_inverted_pendulum_demo_results'
run = 'run1'
with open('results/'+run+'/xt_est_save100.pkl','wb') as file:
    pickle.dump(xt_est_save, file)
with open('results/'+run+'/theta_est_save100.pkl','wb') as file:
    pickle.dump(theta_est_save, file)
with open('results/'+run+'/q_est_save100.pkl','wb') as file:
    pickle.dump(q_est_save, file)
with open('results/'+run+'/r_est_save100.pkl','wb') as file:
    pickle.dump(r_est_save, file)
with open('results/'+run+'/z_sim100.pkl','wb') as file:
    pickle.dump(z_sim, file)
with open('results/'+run+'/u100.pkl','wb') as file:
    pickle.dump(u, file)
with open('results/'+run+'/mpc_result_save100.pkl', 'wb') as file:
    pickle.dump(mpc_result_save, file)
with open('results/'+run+'/uc_save100.pkl', 'wb') as file:
    pickle.dump(uc_save, file)

