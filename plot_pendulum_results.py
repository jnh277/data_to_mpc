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
import seaborn as sns

# jax related imports
import jax.numpy as jnp
from jax import grad, jit, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.config import config

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier, log_barrier_cosine_cost

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy
plt.rcParams["font.family"] = "Times New Roman"
# lfont = {'fontname':''}
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
plotme1 = True
if plotme1:
    ts = np.arange(50)*0.025
    tsx = np.arange(51)*0.025
    plt.plot(u[0,:])
    plt.title('MPC determined control action')
    plt.axhline(input_bound, linestyle='--', color='r', linewidth=2, label='constraint')
    plt.axhline(-input_bound, linestyle='--', color='r', linewidth=2, label='constraint')
    plt.savefig('stills/plot_action'+'.png',format='png')
    plt.close()
    # plt.show()

    ## ! PLOT ANGLES AND CONTROL
    fig = plt.figure(figsize=(6.4,7.2),dpi=300)
    plt.subplot(3, 2, 1)
    # plt.plot(ts,z_sim[0,0,:-1],label='True',linewidth = 2.0,color='k')
    plt.plot(ts,xt_est_save[:,0,:].mean(axis=0), color=u'#1f77b4',linewidth = 1,label='True')
    # plt.fill_between(ts,np.percentile(xt_est_save[:,0,:],99.0,axis=0),np.percentile(xt_est_save[:,0,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(state_bound, linestyle='--', color='r', linewidth=1.0, label='Constraints')
    plt.axhline(-state_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm angle (rad)')
    # plt.legend()

    plt.subplot(3, 2, 2)
    # plt.plot(ts,z_sim[0,0,:-1],label='True',linewidth = 2.0,color='k')
    err = xt_est_save[:,0,:]-z_sim[0,0,:-1]
    # plt.axhline(0.0, linestyle='--', color='g', linewidth=1.0)
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1,label='Mean of error')
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    # plt.fill_between(ts,np.percentile(xt_est_save[:,0,:],99.0,axis=0),np.percentile(xt_est_save[:,0,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI') 
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm err. distr. (rad)')


    plt.subplot(3, 2, 3)
    # plt.plot(ts,z_im[1,0,:-1],label='True',linewidth = 2.0,color='k')
    plt.plot(ts,xt_est_save[:,1,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    # plt.fill_between(ts,np.percentile(xt_est_save[:,1,:],99.0,axis=0),np.percentile(xt_est_save[:,1,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(-z_star[1,0], linestyle='--', color='g', linewidth=1.0, label='Target')
    plt.xlim([0,49*0.025])
    plt.ylabel('Pendulum angle (rad)')
    # plt.xlabel('Time (s)')
    # plt.legend(loc = 'upper right')

    plt.subplot(3, 2, 4)
    # plt.plot(ts,z_sim[1,0,:-1],label='True',linewidth = 2.0,color='k')
    err = xt_est_save[:,1,:]-z_sim[1,0,:-1]
    # plt.axhline(0.0, linestyle='--', color='g', linewidth=1.0)
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.xlim([0,49*0.025])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel('Pendulum err. distr. (rad)')
    # plt.xlabel('Time (s)')
    # plt.legend(loc = 'upper right')

    plt.subplot(3, 2, 5)
    plt.plot(tsx,u[0,:],color=u'#1f77b4',linewidth = 1)
    # plt.title('MPC determined control action')
    plt.axhline(input_bound, linestyle='--', color='r', linewidth=1.0)
    plt.axhline(-input_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Control action (V)')
    plt.xlabel('Time (s)')

    plt.subplot(3, 2, 6)
    # plt.plot(ts,z_sim[1,0,:-1],label='True',linewidth = 2.0,color='k')
    kin_pend = 0.5*z_sim[2,0,:-1]*z_sim[2,0,:-1]*(mp_true*Lp_true*Lp_true + (mp_true*Lp_true*Lp_true+Jp_true)*np.sin(z_sim[1,0,:-1])*np.sin(z_sim[1,0,:-1]) + Lp_true)
    kin_pend = kin_pend + 0.5*z_sim[3,0,:-1]*z_sim[3,0,:-1]*(Jp_true+mp_true*Lp_true*Lp_true) + mp_true*Lr_true*Lp_true*np.cos(z_sim[1,0,:-1])*z_sim[3,0,:-1]*z_sim[2,0,:-1]
    kin_base = 0.5*z_sim[2,0,:-1]*z_sim[2,0,:-1]*(mr_true*Lr_true*Lr_true + Jr_true)
    kin = kin_base + kin_pend
    pot = grav*mp_true*Lp_true*(1 - np.cos(z_sim[1,0,:-1]))
    plt.plot(ts,kin_pend + pot, color=u'#1f77b4',linewidth = 1)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.ylabel('Total pendulum energy (J)')
    plt.xlabel('Time (s)')
    # plt.legend(loc = 'upper right')


    plt.figlegend(loc='upper center',bbox_to_anchor=[0.5, 0.0666666], ncol=5)
    plt.tight_layout(rect=[0,0.04666666,1,1])
    plt.savefig('stills/plot_angles_and_control'+'.png',format='png')
    plt.close()

    ## ! PLOT ANGLES
    fig = plt.figure(figsize=(6.4,4.8),dpi=300)
    plt.subplot(2, 2, 1)
    # plt.plot(ts,z_sim[0,0,:-1],label='True',linewidth = 2.0,color='k')
    plt.plot(ts,xt_est_save[:,0,:].mean(axis=0), color=u'#1f77b4',linewidth = 1,label='True')
    # plt.fill_between(ts,np.percentile(xt_est_save[:,0,:],99.0,axis=0),np.percentile(xt_est_save[:,0,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(state_bound, linestyle='--', color='r', linewidth=1.0, label='Constraints')
    plt.axhline(-state_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm angle (rad)')
    # plt.legend()

    plt.subplot(2, 2, 2)
    # plt.plot(ts,z_sim[0,0,:-1],label='True',linewidth = 2.0,color='k')
    err = xt_est_save[:,0,:]-z_sim[0,0,:-1]
    # plt.axhline(0.0, linestyle='--', color='g', linewidth=1.0)
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1,label='Mean of error')
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    # plt.fill_between(ts,np.percentile(xt_est_save[:,0,:],99.0,axis=0),np.percentile(xt_est_save[:,0,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI') 
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm err. distr. (rad)')


    plt.subplot(2, 2, 3)
    # plt.plot(ts,z_im[1,0,:-1],label='True',linewidth = 2.0,color='k')
    plt.plot(ts,xt_est_save[:,1,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    # plt.fill_between(ts,np.percentile(xt_est_save[:,1,:],99.0,axis=0),np.percentile(xt_est_save[:,1,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(-z_star[1,0], linestyle='--', color='g', linewidth=1.0, label='Target')
    plt.xlim([0,49*0.025])
    plt.ylabel('Pendulum angle (rad)')
    plt.xlabel('Time (s)')
    # plt.legend(loc = 'upper right')

    plt.subplot(2, 2, 4)
    # plt.plot(ts,z_sim[1,0,:-1],label='True',linewidth = 2.0,color='k')
    err = xt_est_save[:,1,:]-z_sim[1,0,:-1]
    # plt.axhline(0.0, linestyle='--', color='g', linewidth=1.0)
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.xlim([0,49*0.025])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel('Pendulum err. distr. (rad)')
    plt.xlabel('Time (s)')
    # plt.legend(loc = 'upper right')
    plt.figlegend(loc='upper center',bbox_to_anchor=[0.5, 0.1], ncol=5)
    plt.tight_layout(rect=[0,0.07,1,1])
    plt.savefig('stills/plot_angles'+'.png',format='png')
    plt.close()

    ## ! FULL STATE
    fig = plt.figure(figsize=(6.4,9.6),dpi=300)
    plt.subplot(4, 2, 1)
    # plt.plot(ts,z_sim[0,0,:-1],label='True',linewidth = 2.0,color='k')
    plt.plot(ts,xt_est_save[:,0,:].mean(axis=0), color=u'#1f77b4',linewidth = 1,label='State')
    # plt.fill_between(ts,np.percentile(xt_est_save[:,0,:],99.0,axis=0),np.percentile(xt_est_save[:,0,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(state_bound, linestyle='--', color='r', linewidth=1.0, label='Constraints')
    plt.axhline(-state_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm angle (rad)')
    # plt.legend()

    plt.subplot(4, 2, 2)
    # plt.plot(ts,z_sim[0,0,:-1],label='True',linewidth = 2.0,color='k')
    err = xt_est_save[:,0,:]-z_sim[0,0,:-1]
    # plt.axhline(0.0, linestyle='--', color='g', linewidth=1.0)
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1,label='Mean of error')
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    # plt.fill_between(ts,np.percentile(xt_est_save[:,0,:],99.0,axis=0),np.percentile(xt_est_save[:,0,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI') 
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm err. distr. (rad)')

    plt.subplot(4, 2, 3)
    # plt.plot(ts,z_sim[0,0,:-1],label='True',linewidth = 2.0,color='k')
    plt.plot(ts,xt_est_save[:,2,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    # plt.fill_between(ts,np.percentile(xt_est_save[:,0,:],99.0,axis=0),np.percentile(xt_est_save[:,0,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    # plt.axhline(state_bound, linestyle='--', color='r', linewidth=1.0, label='Constraints')
    # plt.axhline(-state_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm velocity (rad/s)')
    # plt.legend()

    plt.subplot(4, 2, 4)
    # plt.plot(ts,z_sim[0,0,:-1],label='True',linewidth = 2.0,color='k')
    err = xt_est_save[:,2,1:]-z_sim[2,0,1:-1]
    # plt.axhline(0.0, linestyle='--', color='g', linewidth=1.0)
    plt.plot(ts[1:],err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts[1:],np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    # plt.fill_between(ts,np.percentile(xt_est_save[:,0,:],99.0,axis=0),np.percentile(xt_est_save[:,0,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI') 
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm velocity err. distr. (rad/s)')


    plt.subplot(4, 2, 5)
    # plt.plot(ts,z_im[1,0,:-1],label='True',linewidth = 2.0,color='k')
    plt.plot(ts,xt_est_save[:,1,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    # plt.fill_between(ts,np.percentile(xt_est_save[:,1,:],99.0,axis=0),np.percentile(xt_est_save[:,1,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(-z_star[1,0], linestyle='--', color='g', linewidth=1.0, label='Target')
    plt.xlim([0,49*0.025])
    plt.ylabel('Pendulum angle (rad)')
    plt.xlabel('Time (s)')
    # plt.legend(loc = 'upper right')

    plt.subplot(4, 2, 6)
    # plt.plot(ts,z_sim[1,0,:-1],label='True',linewidth = 2.0,color='k')
    err = xt_est_save[:,1,:]-z_sim[1,0,:-1]
    # plt.axhline(0.0, linestyle='--', color='g', linewidth=1.0)
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.xlim([0,49*0.025])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel('Pendulum err. distr. (rad)')
    # plt.legend(loc = 'upper right')

    plt.subplot(4, 2, 7)
    # plt.plot(ts,z_im[1,0,:-1],label='True',linewidth = 2.0,color='k')
    plt.plot(ts,xt_est_save[:,3,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    # plt.fill_between(ts,np.percentile(xt_est_save[:,1,:],99.0,axis=0),np.percentile(xt_est_save[:,1,:],1.0,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    # plt.axhline(-z_star[1,0], linestyle='--', color='g', linewidth=1.0, label='Target')
    plt.xlim([0,49*0.025])
    plt.ylabel('Pendulum velocity (rad/s)')
    # plt.legend(loc = 'upper right')

    plt.subplot(4, 2, 8)
    # plt.plot(ts,z_sim[1,0,:-1],label='True',linewidth = 2.0,color='k')
    err = xt_est_save[:,3,1:]-z_sim[3,0,1:-1]
    # plt.axhline(0.0, linestyle='--', color='g', linewidth=1.0)
    plt.plot(ts[1:],err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts[1:],np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.xlim([0,49*0.025])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel('Pendulum velocity err. distr. (rad/s)')
    plt.xlabel('Time (s)')
    # plt.legend(loc = 'upper right')
    lgd = plt.figlegend(loc='upper center',bbox_to_anchor=[0.5, 0.05], ncol=5)
    plt.tight_layout(rect=[0,0.035,1,1])
    plt.savefig('stills/plot_states'+'.png',format='png')
    fig.set_figheight(4.8)
    lgd.set_bbox_to_anchor([0.5, 0.1])
    # plt.figlegend(loc='upper center',bbox_to_anchor=[0.5, 0.1], ncol=5)
    plt.tight_layout(rect=[0,0.07,1,1])
    plt.savefig('stills/plot_states_short'+'.png',format='png')
    plt.close()

    ## ! PARAMS
    ind = 0
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Jr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.title(r'$J_r$ estimate over simulation')
    plt.savefig('stills/plot_'+str(ind)+'.png',format='png')
    plt.close()

    ind = 1
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.axhline(Jp_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlabel(r'Time (s)')
    plt.xlim([0,49*0.025])
    plt.title(r'$J_p$ estimate over simulation')
    plt.savefig('stills/plot_'+str(ind)+'.png',format='png')
    plt.close()
    # plt.show()

    ind = 2
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Km_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.xlim([0,49*0.025])
    plt.title(r'$K_m$ estimate over simulation')
    plt.savefig('stills/plot_'+str(ind)+'.png',format='png')
    plt.close()

    ind = 3
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Rm_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.xlim([0,49*0.025])
    plt.xlabel('Time (s)')
    plt.title(r'$R_m$ estimate over simulation')
    plt.savefig('stills/plot_'+str(ind)+'.png',format='png')
    plt.close()
    # plt.show()

    ind = 4
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Dp_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.xlabel('Time (s)')
    plt.title(r'$D_p$ estimate over simulation')
    plt.savefig('stills/plot_'+str(ind)+'.png',format='png')
    plt.close()

    ind = 5
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Dr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.xlabel('Time (s)')
    plt.title(r'$D_r$ estimate over simulation')
    plt.savefig('stills/plot_'+str(ind)+'.png',format='png')
    plt.close()
    # plt.show()

    # ! PARAM 
    fig = plt.figure(figsize=(6.4,4.8),dpi=300)
    plt.subplot(3, 1, 1)
    ind = 1
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Jp_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    # plt.xlabel(r'Time (s)')
    plt.ylabel(r'$J_p$ ($kg/m^2$)')
    plt.xlim([0,49*0.025])
    plt.title(r'Parameter estimates over simulation')
    plt.subplot(3, 1, 2)
    ind = 3
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Rm_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    # plt.legend()
    plt.xlim([0,49*0.025])
    plt.ylabel(r'$R_m$ ($\Omega$)')
    # plt.xlabel('Time (s)')
    # plt.title(r'$R_m$ estimate over simulation')
    plt.subplot(3, 1, 3)
    ind = 5
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Dr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    # plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.ylabel(r'$D_r$ ($Nms/rad$)')
    plt.xlabel('Time (s)')
    # plt.title(r'$D_r$ estimate over simulation')
    plt.tight_layout()
    plt.savefig('stills/subplot_params.png',format='png')
    plt.close()

    # ! SIX FIGURE SUBPLOT
    fig = plt.figure(figsize=(6.4,4.8),dpi=300)
    plt.subplot(3, 2, 2)
    ind = 1
    l3 = plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    l2 = plt.axhline(Jp_true,color='k',label='True value',linewidth=1,linestyle='--')
    l1 = plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    # plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    # plt.xlabel(r'Time (s)')
    plt.ylabel(r'$J_p$ ($kg/m^2$)')
    plt.xlim([0,25*0.025])
    # plt.suptitle(r'Parameter estimates over simulation')

    plt.subplot(3, 2, 1)
    ind = 0
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Jr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    # plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel(r'$J_r$ ($kg/m^2$)')
    # plt.xlabel('Time (s)')
    plt.xlim([0,25*0.025])

    plt.subplot(3, 2, 3)
    ind = 3
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Rm_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    # plt.legend()
    plt.xlim([0,25*0.025])
    plt.ylabel(r'$R_m$ ($\Omega$)')
    # plt.xlabel('Time (s)')
    # plt.title(r'$R_m$ estimate over simulation')

    plt.subplot(3, 2, 4)
    ind = 2
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Km_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    # plt.legend()
    # plt.xlabel('Time (s)')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel(r'$K_m$ ($Nm/A$)')
    plt.xlim([0,25*0.025])
    # plt.title(r'$K_m$ estimate over simulation')

    plt.subplot(3, 2, 5)
    ind = 5
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Dr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    # plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,25*0.025])
    plt.ylabel(r'$D_r$ ($Nms/rad$)')
    plt.xlabel('Time (s)')
    # plt.title(r'$D_r$ estimate over simulation')
    plt.subplot(3, 2, 6)
    ind = 4
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Dp_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    # plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,25*0.025])
    plt.ylabel(r'$D_p$ ($Nms/rad$)')
    plt.xlabel('Time (s)')

    plt.tight_layout(rect=[0,0.07,1,1])
    plt.figlegend(('True value','Sample mean','95% CI'),loc='upper center',bbox_to_anchor=[0.5, 0.1], ncol=3)
    plt.savefig('stills/subplot_params_six.png',format='png')
    plt.close()

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

# import matplotlib.animation as manimation
# FFMpegWriter = manimation.writers['ffmpeg']
# # writer = manimation.FFMpegFileWriter(fps=15)
# # FFMpegFileWriter = manimation.writers['ffmpegfile']
# # metadata = dict(title='pendulum_movie', artist='Matplotlib')
# # writer = FFMpegWriter(fps=10, metadata=metadata)
# writer = FFMpegWriter(fps=10)
# writer = FFMpegFileWriter(fps=15
# fig,ax =  plt.subplots(2,2, gridspec_kw={
#                             'width_ratios':[2,1],
#                             'height_ratios':[2,1]})
#
# plt.show()

# t = 10
# pl = 0.5

# # TODO: data interpolation (linear)
# # super_fig = plt.figure()
# # axe = fig.gca(projection='3d')
# # plt.hist(x_mpc[1,:, 0], label='MC forward sim')
# mpc_n = 5
# q_mpc = q_est_save[:,:,mpc_n].T
# w_mpc2 = np.zeros((Nx,Ns,Nh+1),dtype=float)
# w_mpc2[0,:,:] = np.expand_dims(col_vec(q_mpc[0,:]) * np.random.randn(Ns, Nh+1), 0)  # uses the sampled stds, need to sample for x_t to x_{t+N+1}
# w_mpc2[1,:,:] = np.expand_dims(col_vec(q_mpc[1,:]) * np.random.randn(Ns, Nh+1), 0)
# w_mpc2[2,:,:] = np.expand_dims(col_vec(q_mpc[2,:]) * np.random.randn(Ns, Nh+1), 0)
# w_mpc2[3,:,:] = np.expand_dims(col_vec(q_mpc[3,:]) * np.random.randn(Ns, Nh+1), 0)
# uc = uc_save[[0],:,mpc_n]
# ut = u[[0],[mpc_n]]
# ut = np.expand_dims(ut,axis=1)
# uc = jnp.hstack([ut, uc])
# xt = xt_est_save[:,:,mpc_n].T
# theta = theta_est_save[:,:,mpc_n]
# # theta = theta.T
# Jr_samps = theta[:,0].squeeze()
# Jp_samps = theta[:,1].squeeze()
# Km_samps = theta[:,2].squeeze()
# Rm_samps = theta[:,3].squeeze()
# Dp_samps = theta[:,4].squeeze()
# Dr_samps = theta[:,5].squeeze()

# theta_mpc = {
#         'Mp': mp_true,
#         'Lp': Lp_true,
#         'Lr': Lr_true,
#         'Jr': Jr_samps,
#         'Jp': Jp_samps,
#         'Km': Km_samps,
#         'Rm': Rm_samps,
#         'Dp': Dp_samps,
#         'Dr': Dr_samps,
#         'g': grav,
#         'h': Ts
# }
# theta_mpc = fill_theta(theta_mpc)
# xtraj = sim(xt,uc,w_mpc2,theta_mpc)

# # add "actual" copnstraint violation 

# # compute as future



# axe = sns.kdeplot(data=xtraj[0,:,:], fill=True,alpha=.5,linewidth=0.2)
# axe.set_xlabel(r'Base arm angled (rad)')
# axe.axvline(-0.75*np.pi,color='r',linestyle='--',linewidth=0.75)
# plt.show()


for t in range(T+15):
    fig, ax = plt.subplots(3,1,gridspec_kw={'width_ratios':[1],
                                            'height_ratios':[2,1,1]})
    t = min(t,T-1)
    ## set up first plot
    l, = ax[0].plot([], [], 'k-o')
    ax[0].axhline(0.0,color='k',linestyle='--',linewidth=0.5)
    ax[0].axvline(-0.75*np.pi,color='r',linestyle='--',linewidth=0.75)
    ax[0].axvline(0.75*np.pi,color='r',linestyle='--',linewidth=0.75)
    ax[0].axis('equal')
    ax[0].axis([-0.8*np.pi,0.8*np.pi,-100,100])

    px = np.array([z_sim[0,0,t],z_sim[0,0,t]+pl*np.sin(z_sim[1,0,t])])
    py = np.array([0.,-pl*np.cos(z_sim[1,0,t])])


    l.set_data(px,py)
    labels = [str(int(val/np.pi*180.)) for val in np.linspace(-0.8*np.pi,0.8*np.pi,7)]
    ax[0].set_xticks(np.linspace(-0.8*np.pi,0.8*np.pi,7))
    ax[0].set_xticklabels(labels)
    ax[0].set_yticks([])
    ax[0].set_xlabel(r'Base arm angle ($^{\circ}$)')


    # set up second plot
    ts = np.arange(t+1)*0.025
    #
    ax[1].axhline(Jr_true,color='k',linestyle='--',linewidth=0.75,label='True')
    ax[1].axis([0,49.*0.025,1.78e-4,3.6e-4])
    ax[1].set_ylabel(r'$J_r$')
    ax[1].set_xlabel('t (s)')
    ind = 0
    l2, = ax[1].plot(ts,theta_est_save[:,ind,0:t+1].mean(axis=0),color=u'#1f77b4',linewidth=1,label='mean')
    ax[1].fill_between(ts,np.percentile(theta_est_save[:,ind,0:t+1],97.5,axis=0),np.percentile(theta_est_save[:,ind,0:t+1],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    ax[1].legend(loc='center right')
    ax[1].ticklabel_format(style='sci',scilimits=(-1,1))
    #
    # set up third plot
    ax[2].axhline(Rm_true,color='k',linestyle='--',linewidth=0.75,label='True')
    ax[2].axis([0,49.*0.025,5.,12.5])
    ax[2].set_ylabel(r'$R_m$')
    ax[2].set_xlabel('t (s)')
    ind = 3
    l5, = ax[2].plot(ts,theta_est_save[:,ind,0:t+1].mean(axis=0),color=u'#1f77b4',linewidth=1,label='mean')
    ax[2].fill_between(ts,np.percentile(theta_est_save[:,ind,0:t+1],97.5,axis=0),np.percentile(theta_est_save[:,ind,0:t+1],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    # l6, = ax[2].plot(ts,np.percentile(theta_est_save[:,ind,0:t+1],97.5,axis=0),color='b',linestyle='--',label='95% CI')
    # l7, = ax[2].plot(ts,np.percentile(theta_est_save[:,ind,0:t+1],2.5,axis=0),color='b',linestyle='--')
    ax[2].legend(loc='center right')
    plt.tight_layout()
    plt.savefig('movie_frames/frame'+str(t)+'.png',format='png')
    plt.close(fig)

# compile frames into movie using ffmpeg -i "frame%d.png"  test.m4v
