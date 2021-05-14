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

""" This script plots the results from simulation A) Pedagogical Example """
""" You can plot the results from the paper by running this script, otherwise
    if you run 'single_state_mpc_demo.py' then you will generate new results which
    will be plotted instead """

# general imports
import numpy as np
import matplotlib.pyplot as plt
from helpers import col_vec
import pickle


# jax related imports
import jax.numpy as jnp
from jax.ops import index, index_update
from jax.config import config
import seaborn as sns

config.update("jax_enable_x64", True)           # run jax in 64 bit mode for accuracy


# Control parameters
x_star = np.array([1.0])        # desired set point
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
T = 30              # number of time steps to simulate and record measurements for
x0 = 0.5            # initial time step
r_true = 0.05       # measurement noise standard deviation
q_true = 0.05       # process noise standard deviation

#----------------- Simulate the system-------------------------------------------#
def ssm(x, u, a=0.9, b=0.2):
    return a*x + b*np.sin(u)

## define jax friendly function for simulating the system during mpc
def simulate(xt, u, w, theta):
    a = theta['a']
    b = theta['b']
    [o, M, N] = w.shape
    x = jnp.zeros((o, M, N+1))
    x = index_update(x, index[:, :,0], xt)
    for k in range(N):
        x = index_update(x, index[:, :, k+1], a * x[:, :, k] + b * jnp.sin(u[:, k]) + w[:, :, k])
    return x[:, :, 1:]




run = 'non_affine_results'
with open('results/'+run+'/xt_est_save.pkl','rb') as file:
    xt_est_save = pickle.load(file)
with open('results/'+run+'/a_est_save.pkl','rb') as file:
    a_est_save = pickle.load(file)
with open('results/'+run+'/b_est_save.pkl','rb') as file:
    b_est_save = pickle.load(file)
with open('results/'+run+'/q_est_save.pkl','rb') as file:
    q_est_save = pickle.load(file)
with open('results/'+run+'/r_est_save.pkl','rb') as file:
    r_est_save = pickle.load(file)
with open('results/'+run+'/x.pkl','rb') as file:
    x = pickle.load(file)
with open('results/'+run+'/u.pkl','rb') as file:
    u = pickle.load(file)
with open('results/'+run+'/mpc_result_save.pkl', 'rb') as file:
    mpc_result_save = pickle.load(file)
with open('results/' + run + '/accept_rates.pkl', 'rb') as file:
    accept_rates = pickle.load(file)


fontsize=12
fig = plt.figure(figsize=(6,4))
plt.subplot(2,1,1)
plt.rcParams["font.family"] = "Times New Roman"
#print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.fill_between(np.arange(T),np.percentile(xt_est_save[0,:,:],97.5,axis=0),np.percentile(xt_est_save[0,:,:],2.5,axis=0),alpha=0.2,label='95% CI',color=u'#1f77b4')
plt.plot(x,label='True', color='k',linewidth=2.)
plt.plot(xt_est_save[0,:,:].mean(axis=0),linewidth=2.,label='mean',color=u'#1f77b4',linestyle='--')
plt.ylabel('x', fontsize=fontsize)
plt.xticks([])
plt.axhline(x_ub,linestyle='--',color='r',linewidth=2.,label='constraint')
plt.axhline(x_star,linestyle='--',color='g',linewidth=2.,label='target')

plt.subplot(2,1,2)
plt.plot(u,linewidth=2., color='k')
# plt.axhline(u_ub,linestyle='--',color='r',linewidth=2.)
plt.ylabel('u', fontsize=fontsize)
plt.xlabel(r'$t$', fontsize=fontsize)
plt.figlegend(loc='upper center',bbox_to_anchor=[0.55, 0.063], ncol=5)
plt.tight_layout(rect=[0.0,0.03,1,1])
plt.savefig('stills/order1_x_u.png', format='png')
plt.close()
# plt.show()

fig = plt.figure(figsize=(6.4,3.5))
plt.subplot(2,2,1)
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(a_est_save.mean(axis=0),linewidth=2)
plt.fill_between(np.arange(T),np.percentile(a_est_save,97.5,axis=0),np.percentile(a_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
plt.ylabel(r'$a$', fontsize=fontsize)
plt.axhline(0.9,linestyle='--',color='k',linewidth=2.)
plt.xticks([])

plt.subplot(2,2,2)
plt.plot(b_est_save.mean(axis=0),linewidth=2,label='Mean')
plt.fill_between(np.arange(T),np.percentile(b_est_save,97.5,axis=0),np.percentile(b_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
plt.ylabel(r'$b$',fontsize=fontsize)
plt.axhline(0.2,linestyle='--',color='k',linewidth=2.,label='True')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(q_est_save.mean(axis=0),linewidth=2)
plt.fill_between(np.arange(T),np.percentile(q_est_save,97.5,axis=0),np.percentile(q_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
plt.ylabel(r'$q$',fontsize=fontsize)
plt.axhline(q_true,linestyle='--',color='k',linewidth=2.)
plt.xlabel(r'$t$',fontsize=fontsize)

plt.subplot(2,2,4)
plt.plot(r_est_save.mean(axis=0),linewidth=2)
plt.fill_between(np.arange(T),np.percentile(r_est_save,97.5,axis=0),np.percentile(r_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
plt.ylabel(r'$r$',fontsize=fontsize)
plt.axhline(r_true,linestyle='--',color='k',linewidth=2.)
plt.xlabel(r'$t$',fontsize=fontsize)
plt.figlegend(loc='upper center',bbox_to_anchor=[0.54, 0.1], ncol=5)
plt.tight_layout(rect=[0,0.03,1,1])
# plt.savefig('stills/order1_params.png', format='png')
# plt.close()
plt.show()

tt = 13
result = mpc_result_save[tt]
xt = np.reshape(x[tt],(1,1))
ut = np.array([[u[tt]]])
a = a_est_save[:,tt]
b = b_est_save[:,tt]
q = q_est_save[:,tt]
w = np.expand_dims(col_vec(q) * np.random.randn(M, N+1), 0)
theta = {'a':a,
         'b':b}
uc = result['uc']

if len(state_constraints) > 0:
    x_mpc = simulate(xt, np.hstack([ut, uc]), w, theta)
    hx = np.concatenate([state_constraint(x_mpc[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = np.mean(hx > 0, axis=1)
    print('State constraint satisfaction  over forecast horizon')
    print(cx)
if len(input_constraints) > 0:
    cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints],axis=1)
    print('Input constraint satisfaction over forecast horizon')
    print(cu >= 0)
#

fig = plt.figure(figsize=(6,3.7))
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.rcParams["font.family"] = "Times New Roman"
    sns.kdeplot(data=x_mpc[0, :, i*3], fill=True, alpha=.5, linewidth=0.2,label='density')
    plt.ylabel('')
    if i==0:
        plt.ylabel(r'$p(x_{t+k} | y_{1:t},u_{1:t})$', fontsize=fontsize)
    plt.axvline(x_star, linestyle='--', color='g', linewidth=2, label='target')
    plt.axvline(x_ub, linestyle='--', color='r', linewidth=2, label='constraint')
    plt.xlabel(r'$x_{t+k}$ for $k='+str(i*3+1)+'$', fontsize=fontsize)
    plt.xlim([-0.7,2.1])
    plt.yticks([])
plt.tight_layout()

plt.subplot(2,1,2)
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(np.arange(1,N+1),uc[0,:],color='k',linewidth=2.0)
# plt.axhline(u_ub, linestyle='--', color='r', linewidth=2, label='constraint')
plt.xlabel(r'$u_{t+k}$ for $k \in [1,N]$', fontsize=fontsize)
plt.ylabel(r'u', fontsize=fontsize)
plt.xlim([1,10])
plt.savefig('stills/order1_mpc_horizon.png', format='png')
# plt.close()
plt.show()




