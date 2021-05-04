"""
Simulates a mass-spring-damper system in one dimension (2 state system) with random noise on the state transition. 
Generates measurements of the system corresponding to position and acceleration of the mass, with random noise on each measurement.
At time t, the system model changes from system1 --> system 2, in the simplest case this is a change in the parameters.
Given a window of data where the switch occurs at time t, pystan will sample the states, parameters for both systems, and switching time t jointly.
"""
import os
import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")
# general imports
import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace
from pathlib import Path
import pickle

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev
from jax.lax import scan
from jax.ops import index, index_update
from jax.config import config
config.update("jax_enable_x64", True)

plot_bool = True
#----------------- Parameters ---------------------------------------------------#

T = 50             # number of time steps to simulate and record measurements for
Ts = 0.004
# true (simulation) parameters
z1_0 = 0.009  # initial position
z2_0 = 0.0  # initial velocity
r1_true = 0.00002 # measurement noise standard deviation
q1_true = 0.0005 * Ts # process noise standard deviation
q2_true = 0.00005 * Ts 

# got these values from the data sheet
Mb_true = 0.06 # kg, mass of the steel ball
Ldiff_true = 0.04 # H, difference between coil L at zero and infinity (L(0) - L(inf))
x50_true = 0.002 # m, location in mode of L where L is 50% of the infinity and 0 L's
grav = 9.81
k0_true = 0.5*x50_true*Ldiff_true/Mb_true
I0_true = x50_true
theta_true = {
        'Mb': Mb_true,
        'Ldiff': Ldiff_true,
        'x50': x50_true,
        'I0': I0_true,
        'k0': k0_true,
        'g': grav,
        'h': Ts
}
Nx = 2
Ny = 1
Nu = 1

#----------------- Simulate the system-------------------------------------------#

def maglev_gradient(xt, u, t): # uses the lumped parametersation 
    dx = jnp.zeros_like(xt)
    dx = index_update(dx, index[0, :], xt[1, :])
    dx = index_update(dx, index[1, :], t['g'] - t['k0'] * u * u / (t['I0'] + xt[0,:]) / (t['I0'] + xt[0,:]))
    return dx

def current_current(xt,t): # expect a slice of x_hist to be xt, shape (2,)
    return np.sqrt(t['g']/t['k0'])*(xt[0] + t['I0'])

def rk4(xt, ut, theta):
    h = theta['h']
    k1 = maglev_gradient(xt, ut, theta)
    k2 = maglev_gradient(xt + k1 * h / 2, ut, theta)
    k3 = maglev_gradient(xt + k2 * h / 2, ut, theta)
    k4 = maglev_gradient(xt + k3 * h, ut, theta)
    return xt + (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6) * h  # should handle a 2D x just fine

def simulate(xt, u, w,
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


z_sim = np.zeros((Nx,1,T+1), dtype=float) # state history allocation

# load initial state
z_sim[0,0,0] = z1_0 
z_sim[1,0,0] = z2_0 

# noise is predrawn and independant
w_sim = np.zeros((Nx,1,T),dtype=float)
w_sim[0,0,:] = np.random.normal(0.0, q1_true, T)
w_sim[1,0,:] = np.random.normal(0.0, q2_true, T)

# create some inputs that are a step from one equilibrium to another
u = current_current(z_sim[:,0,0],theta_true) * np.ones((Nu,T), dtype=float)
u[:,25:] = u[:,25:] - 0.001

z_sim[:,:,1:] = simulate(z_sim[:,:,0],u,w_sim,theta_true)

# draw measurement noise
v = np.zeros((Ny,T), dtype=float)
v[0,:] = np.random.normal(0.0, r1_true, T)

# simulated measurements 
y = np.zeros((Ny,T), dtype=float)
y[0,:] = z_sim[0,0,:-1]
y = y + v; # add noise to measurements

plt.subplot(2,1,1)
plt.plot(u[0,:])
plt.title('Simulated inputs and measurement used for inference')
plt.subplot(2, 1, 2)
plt.plot(z_sim[0,:])
plt.plot(y[0,:],linestyle='None',color='r',marker='*')
plt.title('Simulated state 1 and measurements used for inferences')
plt.tight_layout()
plt.show()

# #----------- USE HMC TO PERFORM INFERENCE ---------------------------#
# # avoid recompiling
# script_path = os.path.dirname(os.path.realpath(__file__))
# model_name = 'maglev'
# path = '/stan/'
# if Path(script_path+path+model_name+'.pkl').is_file():
#     model = pickle.load(open(script_path+path+model_name+'.pkl', 'rb'))
# else:
#     model = pystan.StanModel(file=script_path+path+model_name+'.stan')
#     with open(script_path+path+model_name+'.pkl', 'wb') as file:
#         pickle.dump(model, file)

# stan_data = {
#     'N':T,
#     'y':y,
#     'u':u,
#     'g':grav,
    # 'theta_p_mu':theta_p_mu,
    # 'theta_p_std':0.0005*np.array([I0_true, k0_true]),
    # 'r_p_mu': np.array([r1_true]),
    # 'r_p_std': 0.0005*np.array([r1_true]),
    # 'q_p_mu': np.array([q1_true, q2_true]),
    # 'q_p_std': 0.0005*np.array([q1_true, q2_true]),
#     'Ts':Ts
# }

# fit = model.sampling(data=stan_data, warmup=1000, iter=2000)
# traces = fit.extract()

# # state samples
# z_samps = np.transpose(traces['z'],(1,0,2)) # Ns, Nx, T --> Nx, Ns, T
# theta_samps = traces['theta']

# # parameter samples
# I0_samps = theta_samps[:, 0].squeeze()
# k0_samps = theta_samps[:, 1].squeeze()
# r_samps = traces['r']
# q_samps = traces['q']

# # plot the initial parameter marginal estimates
# q1plt = q_samps[0,:].squeeze()
# q2plt = q_samps[1,:].squeeze()
# r1plt = r_samps[0,:].squeeze()

# plot_trace_grid(I0_samps,1,5,1,'I0')
# plt.title('HMC inferred parameters')
# plot_trace_grid(k0_samps,1,5,2,'k0')
# plot_trace_grid(q1plt,1,5,3,'q1')
# plot_trace_grid(q1plt,1,5,4,'q2')
# plot_trace_grid(r1plt,1,5,5,'r1')
# plt.show()

# # plot some of the initial marginal state estimates
# for i in range(4):
#     if i==1:
#         plt.title('HMC inferred position')
#     plt.subplot(2,2,i+1)
#     plt.hist(z_samps[0,:,i*20+1],bins=30, label='p(x_'+str(i+1)+'|y_{1:T})', density=True)
#     plt.axvline(z_sim[0,i*20+1], label='True', linestyle='--',color='k',linewidth=2)
#     plt.xlabel('x_'+str(i+1))
# plt.tight_layout()
# plt.legend()
# plt.show()






