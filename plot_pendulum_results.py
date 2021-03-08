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

""" This script plots the results from simulation B) Rotary Inverted Pendulum """
""" You can plot the results from the paper by running this script, otherwise
    if you run 'inverted_pendulum_mpc_demo.py' then you will generate new results which
    will be plotted instead """

# general imports
import numpy as np
import matplotlib.pyplot as plt
import pickle

# jax related imports
import jax.numpy as jnp
from jax import jit
from jax.ops import index, index_update
from jax.config import config



config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy
plt.rcParams["font.family"] = "Times New Roman"
# Control parameters
z_star = np.array([[0],[np.pi],[0.0],[0.0]],dtype=float)        # desired set point in z1
Ns = 200             # number of samples we will use for MC MPC
Nh = 25              # horizonline of MPC algorithm
sqc_v = np.array([1,30.,1e-5,1e-5],dtype=float)
sqc = np.diag(sqc_v)
src = np.array([[0.001]])

# define the state constraints, (these need to be tuples)
state_bound = 0.75*np.pi
input_bound = 18.0
state_constraints = (lambda z: state_bound - z[[0],:,:],lambda z: z[[0],:,:] + state_bound)
# define the input constraints
input_constraints = (lambda u: input_bound - u, lambda u: u + input_bound)


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
run = 'rotary_inverted_pendulum_demo_results'
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
    # plt.savefig('stills/'+run+'_plot_action'+'.png',format='png')
    # plt.close()
    plt.show()

    ## ! PLOT ANGLES AND CONTROL
    fig = plt.figure(figsize=(6.4,7.2),dpi=300)
    plt.subplot(3, 2, 1)
    plt.plot(ts,xt_est_save[:,0,:].mean(axis=0), color=u'#1f77b4',linewidth = 1,label='True')
    plt.axhline(state_bound, linestyle='--', color='r', linewidth=1.0, label='Constraints')
    plt.axhline(-state_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm angle (rad)')

    plt.subplot(3, 2, 2)
    err = xt_est_save[:,0,:]-z_sim[0,0,:-1]
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1,label='Mean of error')
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm err. distr. (rad)')


    plt.subplot(3, 2, 3)
    plt.plot(ts,xt_est_save[:,1,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    plt.axhline(-z_star[1,0], linestyle='--', color='g', linewidth=1.0, label='Target')
    plt.xlim([0,49*0.025])
    plt.ylabel('Pendulum angle (rad)')

    plt.subplot(3, 2, 4)
    err = xt_est_save[:,1,:]-z_sim[1,0,:-1]
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.xlim([0,49*0.025])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel('Pendulum err. distr. (rad)')


    plt.subplot(3, 2, 5)
    plt.plot(tsx,u[0,:],color=u'#1f77b4',linewidth = 1)
    plt.axhline(input_bound, linestyle='--', color='r', linewidth=1.0)
    plt.axhline(-input_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Control action (V)')
    plt.xlabel('Time (s)')

    plt.subplot(3, 2, 6)
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


    plt.figlegend(loc='upper center',bbox_to_anchor=[0.5, 0.0666666], ncol=5)
    plt.tight_layout(rect=[0,0.04666666,1,1])
    # plt.savefig('stills/'+run+'_plot_angles_and_control'+'.png',format='png')
    # plt.close()
    plt.show()

    ## ! PLOT ANGLES
    fig = plt.figure(figsize=(6.4,4.8),dpi=300)
    plt.subplot(2, 2, 1)
    plt.plot(ts,xt_est_save[:,0,:].mean(axis=0), color=u'#1f77b4',linewidth = 1,label='True')
    plt.axhline(state_bound, linestyle='--', color='r', linewidth=1.0, label='Constraints')
    plt.axhline(-state_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm angle (rad)')


    plt.subplot(2, 2, 2)
    err = xt_est_save[:,0,:]-z_sim[0,0,:-1]
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1,label='Mean of error')
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm err. distr. (rad)')


    plt.subplot(2, 2, 3)
    plt.plot(ts,xt_est_save[:,1,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    plt.axhline(-z_star[1,0], linestyle='--', color='g', linewidth=1.0, label='Target')
    plt.xlim([0,49*0.025])
    plt.ylabel('Pendulum angle (rad)')
    plt.xlabel('Time (s)')


    plt.subplot(2, 2, 4)
    err = xt_est_save[:,1,:]-z_sim[1,0,:-1]
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.xlim([0,49*0.025])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel('Pendulum err. distr. (rad)')
    plt.xlabel('Time (s)')
    plt.figlegend(loc='upper center',bbox_to_anchor=[0.5, 0.1], ncol=5)
    plt.tight_layout(rect=[0,0.07,1,1])
    # plt.savefig('stills/'+run+'_plot_angles'+'.png',format='png')
    # plt.close()
    plt.show()

    ## ! FULL STATE
    fig = plt.figure(figsize=(6.4,9.6),dpi=300)
    plt.subplot(4, 2, 1)
    plt.plot(ts,xt_est_save[:,0,:].mean(axis=0), color=u'#1f77b4',linewidth = 1,label='State')
    plt.axhline(state_bound, linestyle='--', color='r', linewidth=1.0, label='Constraints')
    plt.axhline(-state_bound, linestyle='--', color='r', linewidth=1.0)
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm angle (rad)')

    plt.subplot(4, 2, 2)
    err = xt_est_save[:,0,:]-z_sim[0,0,:-1]
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1,label='Mean of error')
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm err. distr. (rad)')

    plt.subplot(4, 2, 3)
    plt.plot(ts,xt_est_save[:,2,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm velocity (rad/s)')

    plt.subplot(4, 2, 4)
    err = xt_est_save[:,2,1:]-z_sim[2,0,1:-1]
    plt.plot(ts[1:],err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts[1:],np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.ylabel('Base arm velocity err. distr. (rad/s)')


    plt.subplot(4, 2, 5)
    plt.plot(ts,xt_est_save[:,1,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    plt.axhline(-z_star[1,0], linestyle='--', color='g', linewidth=1.0, label='Target')
    plt.xlim([0,49*0.025])
    plt.ylabel('Pendulum angle (rad)')
    plt.xlabel('Time (s)')


    plt.subplot(4, 2, 6)
    err = xt_est_save[:,1,:]-z_sim[1,0,:-1]
    plt.plot(ts,err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts,np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.xlim([0,49*0.025])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel('Pendulum err. distr. (rad)')

    plt.subplot(4, 2, 7)
    plt.plot(ts,xt_est_save[:,3,:].mean(axis=0), color=u'#1f77b4',linewidth = 1)
    plt.xlim([0,49*0.025])
    plt.ylabel('Pendulum velocity (rad/s)')

    plt.subplot(4, 2, 8)
    err = xt_est_save[:,3,1:]-z_sim[3,0,1:-1]
    plt.plot(ts[1:],err.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1)
    plt.fill_between(ts[1:],np.percentile(err,97.5,axis=0),np.percentile(err,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.xlim([0,49*0.025])
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel('Pendulum velocity err. distr. (rad/s)')
    plt.xlabel('Time (s)')
    lgd = plt.figlegend(loc='upper center',bbox_to_anchor=[0.5, 0.05], ncol=5)
    plt.tight_layout(rect=[0,0.035,1,1])
    # plt.savefig('stills/'+run+'_plot_states'+'.png',format='png')
    fig.set_figheight(4.8)
    lgd.set_bbox_to_anchor([0.5, 0.1])
    plt.tight_layout(rect=[0,0.07,1,1])
    # plt.savefig('stills/'+run+'_plot_states_short'+'.png',format='png')
    # plt.close()
    plt.show()

    ## ! PARAMS
    ind = 0
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Jr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.title(r'$J_r$ estimate over simulation')
    # plt.savefig('stills/'+run+'_plot_'+str(ind)+'.png',format='png')
    # plt.close()
    plt.show()

    ind = 1
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15)
    plt.axhline(Jp_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlabel(r'Time (s)')
    plt.xlim([0,49*0.025])
    plt.title(r'$J_p$ estimate over simulation')
    # plt.savefig('stills/'+run+'_plot_'+str(ind)+'.png',format='png')
    # plt.close()
    plt.show()

    ind = 2
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Km_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.xlim([0,49*0.025])
    plt.title(r'$K_m$ estimate over simulation')
    # plt.savefig('stills/'+run+'_plot_'+str(ind)+'.png',format='png')
    # plt.close()
    plt.show()

    ind = 3
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Rm_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.xlim([0,49*0.025])
    plt.xlabel('Time (s)')
    plt.title(r'$R_m$ estimate over simulation')
    # plt.savefig('stills/'+run+'_plot_'+str(ind)+'.png',format='png')
    # plt.close()
    plt.show()

    ind = 4
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Dp_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.xlabel('Time (s)')
    plt.title(r'$D_p$ estimate over simulation')
    # plt.savefig('stills/'+run+'_plot_'+str(ind)+'.png',format='png')
    # plt.close()
    plt.show()

    ind = 5
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
    plt.axhline(Dr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.xlabel('Time (s)')
    plt.title(r'$D_r$ estimate over simulation')
    # plt.savefig('stills/'+run+'_plot_'+str(ind)+'.png',format='png')
    # plt.close()
    plt.show()

    # ! PARAM 
    fig = plt.figure(figsize=(6.4,4.8),dpi=300)
    plt.subplot(3, 1, 1)
    ind = 1
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Jp_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.legend()
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel(r'$J_p$ ($kg/m^2$)')
    plt.xlim([0,49*0.025])
    plt.title(r'Parameter estimates over simulation')
    plt.subplot(3, 1, 2)
    ind = 3
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Rm_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.xlim([0,49*0.025])
    plt.ylabel(r'$R_m$ ($\Omega$)')
    plt.subplot(3, 1, 3)
    ind = 5
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Dr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,49*0.025])
    plt.ylabel(r'$D_r$ ($Nms/rad$)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    # plt.savefig('stills/'+run+'_subplot_params.png',format='png')
    # plt.close()

    # ! SIX FIGURE SUBPLOT
    fig = plt.figure(figsize=(6.4,4.8),dpi=300)
    plt.subplot(3, 2, 2)
    ind = 1
    l3 = plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    l2 = plt.axhline(Jp_true,color='k',label='True value',linewidth=1,linestyle='--')
    l1 = plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel(r'$J_p$ ($kg/m^2$)')
    plt.xlim([0,25*0.025])

    plt.subplot(3, 2, 1)
    ind = 0
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Jr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel(r'$J_r$ ($kg/m^2$)')
    plt.xlim([0,25*0.025])

    plt.subplot(3, 2, 3)
    ind = 3
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Rm_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.xlim([0,25*0.025])
    plt.ylabel(r'$R_m$ ($\Omega$)')

    plt.subplot(3, 2, 4)
    ind = 2
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Km_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.ylabel(r'$K_m$ ($Nm/A$)')
    plt.xlim([0,25*0.025])

    plt.subplot(3, 2, 5)
    ind = 5
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Dr_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,25*0.025])
    plt.ylabel(r'$D_r$ ($Nms/rad$)')
    plt.xlabel('Time (s)')
    plt.subplot(3, 2, 6)
    ind = 4
    plt.fill_between(ts,np.percentile(theta_est_save[:,ind,:],97.5,axis=0),np.percentile(theta_est_save[:,ind,:],2.5,axis=0),color=u'#DDEBF4',label='95% CI')
    plt.axhline(Dp_true,color='k',label='True value',linewidth=1,linestyle='--')
    plt.plot(ts,theta_est_save[:,ind,:].mean(axis=0),color=u'#1f77b4',label='Sample mean',linewidth=1)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,25*0.025])
    plt.ylabel(r'$D_p$ ($Nms/rad$)')
    plt.xlabel('Time (s)')

    plt.tight_layout(rect=[0,0.07,1,1])
    plt.figlegend(('True value','Sample mean','95% CI'),loc='upper center',bbox_to_anchor=[0.5, 0.1], ncol=3)
    # plt.savefig('stills/'+run+'_subplot_params_six.png',format='png')
    # plt.close()
    plt.show()


