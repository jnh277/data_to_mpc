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
from optimisation import log_barrier_cost, solve_chance_logbarrier, log_barrier_cosine_cost

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy


# Control parameters
z_star = np.array([[0],[np.pi],[0.0],[0.0]],dtype=float)        # desired set point in z1
Ns = 200             # number of samples we will use for MC MPC
Nh = 25              # horizonline of MPC algorithm
# sqc_v = np.array([1,10.0,1e-5,1e-5],dtype=float)            # cost on state error
sqc_v = np.array([1,30.,1e-5,1e-5],dtype=float)
sqc = np.diag(sqc_v)
src = np.array([[0.001]])

# define the state constraints, (these need to be tuples)
state_bound = 0.75*np.pi
input_bound = 18.0
# state_constraints = (lambda z: 0.75*np.pi - z[[0],:,:],lambda z: z[[0],:,:] + 0.75*np.pi)
# # define the input constraints
# input_constraints = (lambda u: 18. - u, lambda u: u + 18.)
state_constraints = (lambda z: state_bound - z[[0],:,:],lambda z: z[[0],:,:] + state_bound)
# define the input constraints
input_constraints = (lambda u: input_bound - u, lambda u: u + input_bound)

# run1: worked with 100, 2pi, and Nh=20, T=30
# run2: worked with 80, pi, and Nh=25, T=50
# run3: briefly stabilised with 60, pi, and NH=25, T=50
# run4: stabilised with 100, 1.5*pi, and Nh = 25, T=50
# run5: start fully down, same as above, stabilised
# run6: stabilised +/-18, no constraints on arm angle, Nh=25, T=50
# run7: stabilised +/-18 input, +/- pi on arm angle, Nh=25, T=50
# run8: static hange start: +/-18 input, +/- pi on angle, NH=25. T=50. Stabilised
# run9: as above but with +/- 0.75pi on angle, stabilised, start z1_0 at close to 0.75 pi
# run10: proper static swing up, starting all around 0, proper constraints

# start making the priors worse
# run11: 10% prior mean error, 20% standard deviation


# simulation parameters
# TODO: WARNING DONT MAKE T > 100 due to size of saved inv_metric
T = 50             # number of time steps to simulate and record measurements for
Ts = 0.025
# z1_0 = 0.7*np.pi            # initial states
# z1_0 = -0.7*np.pi            # initial states
# z1_0 = np.pi - 0.05
z1_0 = 0.0
# z1_0 = 0.75*np.pi - 0.1
# z2_0 = -np.pi/3
z2_0 = 0.001
# z2_0 = np.pi-0.1
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


## load results
run = 'run11'
with open('results/'+run+'/xt_est_save100.pkl','rb') as file:
    xt_est_save = pickle.load(file)
with open('results/'+run+'/theta_est_save100.pkl','rb') as file:
    theta_est_save = pickle.load(file)
with open('results/'+run+'/q_est_save100.pkl','rb') as file:
    q_est_save = pickle.load(file)
with open('results/'+run+'/r_est_save100.pkl','rb') as file:
    r_est_save = pickle.load(file)
with open('results/'+run+'/z_sim100.pkl','rb') as file:
    z_sim = pickle.load(file)
with open('results/'+run+'/u100.pkl','rb') as file:
    u = pickle.load(file)
with open('results/'+run+'/mpc_result_save100.pkl', 'rb') as file:
    mpc_result_save = pickle.load(file)
with open('results/'+run+'/uc_save100.pkl', 'rb') as file:
    uc_save = pickle.load(file)

## Plot results

plt.plot(u[0,:])
plt.title('MPC determined control action')
plt.axhline(input_bound, linestyle='--', color='r', linewidth=2, label='constraint')
plt.axhline(-input_bound, linestyle='--', color='r', linewidth=2, label='constraint')
plt.show()

plt.subplot(2, 1, 1)
plt.plot(z_sim[0,0,:],label='True',color='k')
plt.plot(xt_est_save[:,0,:].mean(axis=0), color='b',label='mean')
plt.plot(np.percentile(xt_est_save[:,0,:],97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
plt.plot(np.percentile(xt_est_save[:,0,:],2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.axhline(state_bound, linestyle='--', color='r', linewidth=2, label='constraint')
plt.axhline(-state_bound, linestyle='--', color='r', linewidth=2)
plt.ylabel('arm angle')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(z_sim[1,0,:],label='True',color='k')
plt.plot(xt_est_save[:,1,:].mean(axis=0), color='b',label='mean')
plt.plot(np.percentile(xt_est_save[:,1,:],97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
plt.plot(np.percentile(xt_est_save[:,1,:],2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.axhline(-z_star[1,0], linestyle='--', color='g', linewidth=2, label='target')
plt.ylabel('pendulum angle')
plt.legend()

plt.show()


ind = 0
plt.plot(theta_est_save[:,ind,:].mean(axis=0),color='b',label='mean')
plt.plot(np.percentile(theta_est_save[:,ind,:],97.5,axis=0),color='b',linestyle='--',label='95% CI')
plt.plot(np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color='b',linestyle='--')
plt.axhline(Jr_true,color='k',label='True')
plt.legend()
plt.xlabel('time step')
plt.title('Jr estimate over simulation')
plt.show()

ind = 2
plt.plot(theta_est_save[:,ind,:].mean(axis=0),color='b',label='mean')
plt.plot(np.percentile(theta_est_save[:,ind,:],97.5,axis=0),color='b',linestyle='--',label='95% CI')
plt.plot(np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color='b',linestyle='--')
plt.axhline(Km_true,color='k',label='True')
plt.legend()
plt.xlabel('time step')
plt.title('Km estimate over simulation')
plt.show()

ind = 3
plt.plot(theta_est_save[:,ind,:].mean(axis=0),color='b',label='mean')
plt.plot(np.percentile(theta_est_save[:,ind,:],97.5,axis=0),color='b',linestyle='--',label='95% CI')
plt.plot(np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color='b',linestyle='--')
plt.axhline(Rm_true,color='k',label='True')
plt.legend()
plt.xlabel('time step')
plt.title('Rm estimate over simulation')
plt.show()


# from matplotlib import animation
# pl = 0.5

# def animate(i):
#     arm_angle = z_sim[0,0,i]
#     pend_angle = z_sim[1,0,i]
#     vx = np.pi * pl * np.sin(pend_angle)
#     vy = pl * np.cos(pend_angle)
#     x = np.array([arm_angle, arm_angle+vx])
#     y = np.array([0, vy])
#     line.set_data(x,y)
#     return line,

import matplotlib.animation as manimation
FFMpegWriter = manimation.writers['ffmpeg']
writer = manimation.FFMpegFileWriter(fps=15)
# FFMpegFileWriter = manimation.writers['ffmpegfile']
# metadata = dict(title='pendulum_movie', artist='Matplotlib')
# writer = FFMpegWriter(fps=10, metadata=metadata)
# writer = FFMpegWriter(fps=10)
# writer = FFMpegFileWriter(fps=15
# fig,ax =  plt.subplots(2,2, gridspec_kw={
#                             'width_ratios':[2,1],
#                             'height_ratios':[2,1]})
#
# plt.show()
t = 10
pl = 0.5

fig, ax = plt.subplots(3,1,gridspec_kw={'width_ratios':[1],
                                        'height_ratios':[2,1,1]})

## set up first plot
l, = ax[0].plot([], [], 'k-o')
ax[0].axhline(0.0,color='k',linestyle='--',linewidth=0.5)
ax[0].axvline(-0.75*np.pi,color='r',linestyle='--',linewidth=1.0)
ax[0].axvline(0.75*np.pi,color='r',linestyle='--',linewidth=1.0)
ax[0].axis('equal')
ax[0].axis([-0.8*np.pi,0.8*np.pi,-100,100])

px = np.array([z_sim[0,0,t],z_sim[0,0,t]+pl*np.sin(z_sim[1,0,t])])
py = np.array([0.,-pl*np.cos(z_sim[1,0,t])])


l.set_data(px,py)
labels = [str(int(val/np.pi*180.)) for val in np.linspace(-0.8*np.pi,0.8*np.pi,7)]
ax[0].set_xticks(np.linspace(-0.8*np.pi,0.8*np.pi,7))
ax[0].set_xticklabels(labels)
ax[0].set_yticks([])
ax[0].set_xlabel('Arm angle (deg)')


# set up second plot
ts = np.arange(t+1)*0.025
#
ax[1].axhline(Jr_true,color='k',linestyle='--',linewidth=1.0,label='True')
ax[1].axis([0,49.*0.025,1.78e-4,3.6e-4])
ax[1].set_ylabel('Jr')
ax[1].set_xlabel('t (s)')
ind = 0
l2, = ax[1].plot(ts,theta_est_save[:,ind,0:t+1].mean(axis=0),color='b',label='mean')
l3, = ax[1].plot(ts,np.percentile(theta_est_save[:,ind,0:t+1],97.5,axis=0),color='b',linestyle='--',label='95% CI')
l4, = ax[1].plot(ts,np.percentile(theta_est_save[:,ind,0:t+1],2.5,axis=0),color='b',linestyle='--')
ax[1].legend()
#
# # set up third plot
# ax[2].axhline(Rm_true,color='k',linestyle='--',linewidth=1.0,label='True')
# ax[2].axis([0,49.*0.025,5.,12.5])
# ax[2].set_ylabel('Rm')
# ax[2].set_xlabel('t (s)')
# ind = 3
# l5, = ax[2].plot(ts,theta_est_save[:,ind,0:t+1].mean(axis=0),color='b',label='mean')
# l6, = ax[2].plot(ts,np.percentile(theta_est_save[:,ind,0:t+1],97.5,axis=0),color='b',linestyle='--',label='95% CI')
# l7, = ax[2].plot(ts,np.percentile(theta_est_save[:,ind,0:t+1],2.5,axis=0),color='b',linestyle='--')
# ax[2].legend()
# plt.show()

with writer.saving(fig, "inverted_pendulum.mp4", 100):
    for t in range(T):
        px = np.array([z_sim[0, 0, t], z_sim[0, 0, t] + pl * np.sin(z_sim[1, 0, t])])
        py = np.array([0., -pl * np.cos(z_sim[1, 0, t])])
        l.set_data(px,py)
        ind = 0
        l2.set_data(ts,theta_est_save[:,ind,0:t+1].mean(axis=0))
        l3.set_data(ts,np.percentile(theta_est_save[:,ind,0:t+1],97.5,axis=0))
        l4.set_data(ts,np.percentile(theta_est_save[:,ind,0:t+1],2.5,axis=0))
        # ind = 3
        # l5.set_data(ts,theta_est_save[:,ind,0:t+1].mean(axis=0))
        # l6.set_data(ts,np.percentile(theta_est_save[:,ind,0:t+1],97.5,axis=0))
        # l7.set_data(ts,np.percentile(theta_est_save[:,ind,0:t+1],2.5,axis=0))

        writer.grab_frame()
    # for i in range(10): # repeat last frame 10 times
    #     px = np.array([z_sim[0, 0, t], z_sim[0, 0, t] + pl * np.sin(z_sim[1, 0, t])])
    #     py = np.array([0., -pl * np.cos(z_sim[1, 0, t])])
    #     l.set_data(px,py)
    #     writer.grab_frame()

# for t in range(T):
#     plt.plot(np.arange(Nh)+t,uc_save[0,:,t])
# plt.title('Control over horizon from each MPC solve')
# plt.show()

# if len(state_constraints) > 0:
#     x_mpc = sim(xt, np.hstack([ut, uc]), w_mpc, theta_mpc)
#     hx = np.concatenate([state_constraint(x_mpc[:, :, 1:]) for state_constraint in state_constraints], axis=2)
#     cx = np.mean(hx > 0, axis=1)
#     print('State constraint satisfaction')
#     print(cx)
# if len(input_constraints) > 0:
#     cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints],axis=1)
#     print('Input constraint satisfaction')
#     print(cu >= 0)
# #
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.hist(x_mpc[1,:, i*4], label='MC forward sim')
#     if i==1:
#         plt.title('MPC solution over horizon')
#     plt.axvline(z_star[1,0], linestyle='--', color='g', linewidth=2, label='target')
#     plt.xlabel('t+'+str(i*4+1))
# plt.tight_layout()
# plt.legend()
# plt.show()



# plt.plot(x_mpc[0,:,:].mean(axis=0))
# plt.title('Predicted future arm angles')
# plt.axhline(state_bound, linestyle='--', color='r', linewidth=2, label='constraint')
# plt.axhline(-state_bound, linestyle='--', color='r', linewidth=2)
# plt.show()
#
# plt.plot(x_mpc[1,:,:].mean(axis=0))
# plt.axhline(z_star[1,0], linestyle='--', color='g', linewidth=2, label='target')
# plt.title('Predicted future pendulum angles')
# plt.show()