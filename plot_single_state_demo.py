"""
Simulate a first order state space model given by
x_{t+1} = a*x_t + b*u_t + w_t
y_t = x_t + e_t
with q and r the standard deviations of w_t and q_t respectively

Jointly estimate the smoothed state trajectories and parameters theta = {a, b, q, r}
to give p(x_{1:T}, theta | y_{1:T})

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
import seaborn as sns

# optimisation module imports (needs to be done before the jax confix update)
from optimisation import log_barrier_cost, solve_chance_logbarrier

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
    # x[:, 0] = xt
    x = index_update(x, index[:, :,0], xt)
    for k in range(N):
        # x[:, k+1] = a * x[:, k] + b * u[k] + w[:, k]
        x = index_update(x, index[:, :, k+1], a * x[:, :, k] + b * u[:, k] + w[:, :, k])
    return x[:, :, 1:]




run = 'single_state_run2'
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
with open('results/'+run+'/mpc_result_save100.pkl', 'rb') as file:
    mpc_result_save = pickle.load(file)


fontsize=12
plt.subplot(2,1,1)
plt.rcParams["font.family"] = "Times New Roman"
#print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.fill_between(np.arange(T),np.percentile(xt_est_save[0,:,:],97.5,axis=0),np.percentile(xt_est_save[0,:,:],2.5,axis=0),alpha=0.2,label='95% CI',color=u'#1f77b4')
plt.plot(x,label='True', color='k',linewidth=2.)
plt.plot(xt_est_save[0,:,:].mean(axis=0),linewidth=2.,label='mean',color=u'#1f77b4',linestyle='--')
# plt.plot(np.percentile(xt_est_save[0,:,:],97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(xt_est_save[0,:,:],2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel('x', fontsize=fontsize)
plt.xticks([])
plt.axhline(x_ub,linestyle='--',color='r',linewidth=2.,label='constraint')
plt.axhline(x_star,linestyle='--',color='g',linewidth=2.,label='target')
# plt.legend(fontsize=12)

plt.subplot(2,1,2)
plt.plot(u,linewidth=2., color='k')
plt.axhline(u_ub,linestyle='--',color='r',linewidth=2.)
plt.ylabel('u', fontsize=fontsize)
plt.xlabel(r'$t$', fontsize=fontsize)
plt.figlegend(loc='upper center',bbox_to_anchor=[0.55, 0.07], ncol=5)
plt.tight_layout(rect=[0.0,0.03,1,1])
plt.savefig('stills/order1_x_u.png', format='png')
plt.close()
# plt.show()

plt.subplot(2,2,1)
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(a_est_save.mean(axis=0),linewidth=2)
plt.fill_between(np.arange(T),np.percentile(a_est_save,97.5,axis=0),np.percentile(a_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
# plt.plot(np.percentile(a_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(a_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel(r'$a$', fontsize=fontsize)
plt.axhline(0.9,linestyle='--',color='k',linewidth=2.)

# plt.xlabel(r'$t$')
plt.xticks([])

plt.subplot(2,2,2)
plt.plot(b_est_save.mean(axis=0),linewidth=2,label='Mean')
plt.fill_between(np.arange(T),np.percentile(b_est_save,97.5,axis=0),np.percentile(b_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% CI')
# plt.plot(np.percentile(b_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(b_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel(r'$b$',fontsize=fontsize)
plt.axhline(0.1,linestyle='--',color='k',linewidth=2.,label='True')
# plt.legend(fontsize=12,bbox_to_anchor=(1.15, 1.15))

# plt.xlabel(r'$t$')
plt.xticks([])

plt.subplot(2,2,3)
plt.plot(q_est_save.mean(axis=0),linewidth=2)
plt.fill_between(np.arange(T),np.percentile(q_est_save,97.5,axis=0),np.percentile(q_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
# plt.plot(np.percentile(q_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(q_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel(r'$q$',fontsize=fontsize)
plt.axhline(q_true,linestyle='--',color='k',linewidth=2.)
# plt.legend()
plt.xlabel(r'$t$',fontsize=fontsize)

plt.subplot(2,2,4)
plt.plot(r_est_save.mean(axis=0),linewidth=2)
plt.fill_between(np.arange(T),np.percentile(r_est_save,97.5,axis=0),np.percentile(r_est_save,2.5,axis=0),color=u'#1f77b4',alpha=0.15)
# plt.plot(np.percentile(r_est_save,97.5,axis=0), color='b',linestyle='--',linewidth=0.5,label='95% CI')
# plt.plot(np.percentile(r_est_save,2.5,axis=0), color='b',linestyle='--',linewidth=0.5)
plt.ylabel(r'$r$',fontsize=fontsize)
plt.axhline(r_true,linestyle='--',color='k',linewidth=2.)
# plt.legend()
plt.xlabel(r'$t$',fontsize=fontsize)


# plt.figlegend(loc='upper center',bbox_to_anchor=[0.5, 0.1],ncol=3)
# plt.tight_layout(rect=[0,0.1,1,1])
plt.figlegend(loc='upper center',bbox_to_anchor=[0.54, 0.0666666], ncol=5)
plt.tight_layout(rect=[0,0.03,1,1])
plt.savefig('stills/order1_params.png', format='png')
plt.close()
# plt.show()


result = mpc_result_save[5]
xt = np.reshape(x[5],(1,1))
ut = np.array([[u[5]]])
a = a_est_save[:,5]
b = b_est_save[:,5]
q = q_est_save[:,5]
w = np.expand_dims(col_vec(q) * np.random.randn(M, N+1), 0)
theta = {'a':a,
         'b':b}
uc = result['uc']

if len(state_constraints) > 0:
    x_mpc = simulate(xt, np.hstack([ut, uc]), w, theta)
    hx = np.concatenate([state_constraint(x_mpc[:, :, 1:]) for state_constraint in state_constraints], axis=2)
    cx = np.mean(hx > 0, axis=1)
    print('State constraint satisfaction')
    print(cx)
if len(input_constraints) > 0:
    cu = jnp.concatenate([input_constraint(uc) for input_constraint in input_constraints],axis=1)
    print('Input constraint satisfaction')
    print(cu >= 0)
#
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.rcParams["font.family"] = "Times New Roman"
    sns.kdeplot(data=x_mpc[0, :, i*3], fill=True, alpha=.5, linewidth=0.2,label='density')
    # plt.hist(x_mpc[0,:, i*3], label='MC forward sim',density=True)
    plt.ylabel('')
    if i==0:
        plt.ylabel(r'$p(x_{t+k} | y_{1:t},u_{1:t})$', fontsize=fontsize)
        # plt.title(r'State prediction over horizon for $t=6$')
    plt.axvline(x_star, linestyle='--', color='g', linewidth=2, label='target')
    plt.axvline(x_ub, linestyle='--', color='r', linewidth=2, label='constraint')
    # if i==2:
        # plt.legend(bbox_to_anchor=(0.85, 0.85))
    plt.xlabel(r'$x_{t+k}$ for $k='+str(i*3+1)+'$', fontsize=fontsize)
    plt.xlim([-0.7,2.3])
    plt.yticks([])
plt.tight_layout()

plt.subplot(2,1,2)
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(np.arange(1,N+1),uc[0,:],color='k',linewidth=2.0)
plt.axhline(u_ub, linestyle='--', color='r', linewidth=2, label='constraint')
# plt.title(r'Control action over horizon for $t=6$')
plt.xlabel(r'$u_{t+k}$ for $k \in [1,N]$', fontsize=fontsize)
plt.ylabel(r'u', fontsize=fontsize)
plt.xlim([1,10])
plt.show()


# axe = sns.kdeplot(data=x_mpc[0,:,:], fill=True,alpha=.5,linewidth=0.2)
# axe.set_xlabel(r'Base arm angled (rad)')
# axe.axvline(-0.75*np.pi,color='r',linestyle='--',linewidth=0.75)
# plt.show()

