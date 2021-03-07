import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# from matplotlib.colors import LightSource
import numpy as np
import pickle
from numpy.core.numeric import zeros_like
from scipy.interpolate import interp1d


def generate_cylinder(p0,p1,R):
    ''' Generates the X, Y, Z meshgrid stuff for a cylinder, starting p0, ending p1, with radius R
        Thank you stack overflow: https://stackoverflow.com/questions/39822480/plotting-a-solid-cylinder-centered-on-a-plane-in-matplotlib'''
    v = p1 - p0 # these gotta be tuples or np arrays, 1d
    mag = np.linalg.norm(v)
    v = v/mag
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v,not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)
    n2 /= np.linalg.norm(n2)
    t = np.linspace(0, mag, 2)
    theta = np.linspace(0, 2 * np.pi, 100)
    rsample = np.linspace(0, R, 2)
    t, theta2 = np.meshgrid(t, theta)
    rsample,theta = np.meshgrid(rsample, theta)
    #generate coordinates for surface
    # "Tube"
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta2) * n1[i] + R * np.cos(theta2) *       n2[i] for i in [0, 1, 2]]
    # "Bottom"
    X2, Y2, Z2 = [p0[i] + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    # "Top"
    X3, Y3, Z3 = [p0[i] + v[i]*mag + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    return X,Y,Z,X2,Y2,Z2,X3,Y3,Z3

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

run = 'run12'
with open('results/'+run+'/z_sim100.pkl','rb') as file:
    z_sim = pickle.load(file)
with open('results/'+run+'/theta_est_save100.pkl','rb') as file:
    theta_est_save = pickle.load(file)
with open('results/'+run+'/u100.pkl','rb') as file:
    u = pickle.load(file)
with open('results/'+run+'/xt_est_save100.pkl','rb') as file:
    xt_est_save = pickle.load(file)

Ns,Nx,Tt = xt_est_save.shape
theta_est_old = xt_est_save[:,0,:]
alpha_est_old = xt_est_save[:,1,:]

means_old = zeros_like(theta_est_save[:,:,:].mean(axis=0))
tops_old = zeros_like(theta_est_save[:,:,:].mean(axis=0))
bots_old = zeros_like(theta_est_save[:,:,:].mean(axis=0))
for ind in range(6):
    means_old[ind,:] = theta_est_save[:,ind,:].mean(axis=0)
    tops_old[ind,:] = np.percentile(theta_est_save[:,ind,:],97.5,axis=0)
    bots_old[ind,:] = np.percentile(theta_est_save[:,ind,:],2.5,axis=0)

theta = z_sim[0,0,:-1].flatten()
alpha = z_sim[1,0,:-1].flatten()

ctrl_old = u[0,:-1]

old_lin = np.linspace(0,1,num = len(theta))
new_lin = np.linspace(0,1,num = 4*len(theta)) # 4 times stretch factor (woah)

# interpolate the sample estimates (by doing a zoh)
theta_est = np.zeros((Ns,len(new_lin)))
alpha_est = np.zeros((Ns,len(new_lin)))
for ii in range(Ns):
    theta_est_itpl = interp1d(old_lin,theta_est_old[ii,:],kind='zero')
    theta_est[ii,:] = theta_est_itpl(new_lin)
    alpha_est_itpl = interp1d(old_lin,alpha_est_old[ii,:],kind='zero')
    alpha_est[ii,:] = alpha_est_itpl(new_lin)


# interpolate the theta estimate parameters
param_itpl = interp1d(old_lin, means_old, kind='zero')
means = param_itpl(new_lin)
param_itpl = interp1d(old_lin, tops_old, kind='zero')
tops = param_itpl(new_lin)
param_itpl = interp1d(old_lin, bots_old, kind='zero')
bots = param_itpl(new_lin)

# zoh on control signal
ctrl_itpl = interp1d(old_lin, ctrl_old,kind = 'zero')
ctrl = ctrl_itpl(new_lin)

# interp true states truthfully (polynomially)
theta_itpl = interp1d(old_lin, theta, kind='cubic')
alpha_itpl = interp1d(old_lin, alpha, kind='cubic')
alpha= alpha_itpl(new_lin)
theta= theta_itpl(new_lin)

scale = 2/0.1 # animation to world scale

x_qube, y_qube, z_qube = np.indices((8, 8, 8))
qube1 = (x_qube > 1) & (y_qube > 2) & (z_qube > -1)
qube2 = (x_qube < 4) & (y_qube < 5) & (z_qube < 2)
qube = qube1 & qube2
colours = np.empty(qube.shape, dtype=object)
colours[qube] = u'#333333'

origin_xy = np.array([3.0,4.0,0.0])
origin = np.array([3.0,4.0,2+0.0254*scale])

## FIGURE
lwidth = 2.0
fig = plt.figure(figsize=(9.6*1.7777777,9.6),dpi=225)
# ax = p3.Axes3D(fig,azim=45,elev=30) #for on it's own
ax = fig.add_subplot(4,7,(1,10), projection='3d',azim=45,elev=30)
ctr_ax = fig.add_subplot(4,7,(4,7))
Jp_ax = fig.add_subplot(4,7,(11,12))
Jp_ax.collections = []
Jr_ax = fig.add_subplot(4,7,(13,14))
Jr_ax.collections = []
Zpkde_ax = fig.add_subplot(4,7,15)
Zp_ax = fig.add_subplot(4,7,(16,17))
Km_ax = fig.add_subplot(4,7,(18,19))
Rm_ax = fig.add_subplot(4,7,(20,21))
Zrkde_ax = fig.add_subplot(4,7,22)
Zr_ax = fig.add_subplot(4,7,(23,24))
Dp_ax = fig.add_subplot(4,7,(25,26))
Dr_ax = fig.add_subplot(4,7,(27,28))
plt.tight_layout(pad = 2.0)
ax.voxels(qube,facecolors=colours,edgecolors=None)
ax.margins( x=None, y=None)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

ax.set_xlim(1, 5) # front-back
ax.set_ylim(2, 6) # left-right
ax.set_zlim3d(0,3)

X,Y,Z,X2,Y2,Z2,X3,Y3,Z3 = generate_cylinder(origin_xy,origin,0.0254/2*scale)
ax.plot_surface(X, Y, Z, color=u'#777777')

# naughty ways of doing stuff and things
X,Y,Z,X2,Y2,Z2,X3,Y3,Z3 = generate_cylinder(origin,origin,0.1)
X4,Y4,Z4,X5,Y5,Z5,X6,Y6,Z6 = generate_cylinder(origin,origin,0.1)
X7,Y7,Z7,X8,Y8,Z8,X9,Y9,Z9 = generate_cylinder(origin,origin,0.1)
plot = [ax.plot_surface(X, Y, Z, color=u'#777777'),ax.plot_surface(X2, Y2, Z2, color=u'#777777'),ax.plot_surface(X3, Y3, Z3, color=u'#777777'), ax.plot_surface(X4, Y4, Z4, color=u'#777777'),ax.plot_surface(X5, Y5, Z5, color=u'#777777'),ax.plot_surface(X6, Y6, Z6, color=u'#777777'),ax.plot_surface(X7, Y7, Z7, color=u'#777777'),ax.plot_surface(X8, Y8, Z8, color=u'#777777'),ax.plot_surface(X9, Y9, Z9, color=u'#777777')]

ts = np.arange(50)*0.025
ts = np.arange(200)*0.025/4
tsx = np.arange(51)*0.025
param_ax = [Jr_ax,Jp_ax,Km_ax,Rm_ax,Dp_ax,Dr_ax]


param_ln = []
param_ci = []

for ind in range(6):
    axed = param_ax[ind]
    plt.axes(axed)
    plt.ticklabel_format(style='sci',axis='y',scilimits=(1,-1))
    param_ci.append(plt.fill_between(ts[[0]],tops[ind,[0]],bots[ind,[0]],color=u'#1f77b4',alpha=0.15))
    param_ln.append( plt.plot(ts[0],means[ind,0],color=u'#1f77b4',linewidth=lwidth))
# param_ax = [Jr_ax,Jp_ax,Km_ax,Rm_ax,Dp_ax,Dr_ax]
state_ax = [Zr_ax,Zp_ax,ctr_ax]
state_ln = []
axed = state_ax[0]
plt.axes(axed)
state_ln.append(plt.plot(ts[0],theta[0],linewidth =lwidth,color='k'))
axed = state_ax[1]
plt.axes(axed)
state_ln.append(plt.plot(ts[0],alpha[0],linewidth =lwidth,color='k'))
axed = state_ax[2]
plt.axes(axed)
state_ln.append(plt.plot(ts[0],ctrl[0],linewidth =lwidth,color='k'))

hist_ax = [Zpkde_ax,Zrkde_ax]
hist_pl = []
axed = hist_ax[0]
plt.axes(axed)
_,_,barrs = axed.hist(alpha_est[:, 0],bins=20,color = u'#1f77b4')
hist_pl.append(barrs)
axed = hist_ax[1]
plt.axes(axed)
_,_,barrs = axed.hist(theta_est[:, 0],bins=20,color = u'#1f77b4')
hist_pl.append(barrs)


# static elements (hopefully these are drawn on top as they are done last)
Zp_ax.axhline(-np.pi, linestyle='--', color='g', linewidth=lwidth, label='Target')
Zp_ax.set_xlim([0,49*0.025])
Zp_ax.set_ylabel(r'Pend. angle (rad)')
Zr_ax.axhline(-0.75*np.pi,color='r',linestyle='--',linewidth=lwidth)
Zr_ax.axhline(0.75*np.pi,color='r',linestyle='--',linewidth=lwidth)
Zr_ax.set_xlim([0,49*0.025])
Zr_ax.set_ylabel(r'Base angle (rad)')
Zrkde_ax.axvline(-0.75*np.pi,color='r',linestyle='--',linewidth=lwidth)
Zrkde_ax.axvline(0.75*np.pi,color='r',linestyle='--',linewidth=lwidth)
Zrkde_ax.set_ylabel(r'Base ang. hist.')
Zrkde_ax.set_yticks([])
Zpkde_ax.axvline(np.pi,color='g',linestyle='--',linewidth=lwidth)
Zpkde_ax.set_ylabel(r'Pend. ang. hist.')
Zpkde_ax.set_yticks([])
ctr_ax.axhline(18.0, linestyle='--', color='r', linewidth=lwidth)
ctr_ax.axhline(-18.0, linestyle='--', color='r', linewidth=lwidth,label='Constraint')
ctr_ax.set_xlim([0,49*0.025])
ctr_ax.set_ylabel(r'Control action (V)')
ctr_ax.set_yticks(np.arange(-18.0, 19.0, 6.0))
# ctr_ax.set_ylim([-18.0,18.0])
Jp_ax.axhline(Jp_true,color='k',linewidth=lwidth,linestyle='--')
Jp_ax.set_xlim([0,49*0.025])
Jp_ax.set_ylabel(r'$J_p$ ($kg/m^2$)')
Jr_ax.axhline(Jr_true,color='k',linewidth=lwidth,linestyle='--')
Jr_ax.set_xlim([0,49*0.025])
Jr_ax.set_ylabel(r'$J_r$ ($kg/m^2$)')
Km_ax.axhline(Km_true,color='k',linewidth=lwidth,linestyle='--')
Km_ax.set_xlim([0,49*0.025])
Km_ax.set_ylabel(r'$K_m$ ($Nm/A$)')
Rm_ax.axhline(Rm_true,color='k',linewidth=lwidth,linestyle='--')
Rm_ax.set_xlim([0,49*0.025])
Rm_ax.set_ylabel(r'$R_m$ ($\Omega$)')
Dp_ax.axhline(Dp_true,color='k',linewidth=lwidth,linestyle='--')
Dp_ax.set_xlim([0,49*0.025])
Dp_ax.set_ylabel(r'$D_p$ ($Nms/rad$)')
Dr_ax.axhline(Dr_true,color='k',linewidth=lwidth,linestyle='--',label='True value')
Dr_ax.set_xlim([0,49*0.025])
Dr_ax.set_ylabel(r'$D_r$ ($Nms/rad$)')


#frame update function
def animate(i,theta,alpha,origin,ts,plot,param_ax,param_ci,param_ln,means,tops,bots,state_ax,state_ln,ctrl,hist_ax,hist_pl,theta_est,alpha_est):
    # ls = LightSource(azdeg =0, altdeg = 65)
    ## ! 3d PLOTS
    plot[0].remove()
    plot[1].remove()
    plot[2].remove()
    plot[3].remove()
    plot[4].remove()
    plot[5].remove()
    plot[6].remove()
    plot[7].remove()
    plot[8].remove()
    scale = 2/0.1
    Lp = 0.129*scale
    Lr = 0.085*scale
    R = 0.1
    r_10 = Lr*np.array([np.cos(theta[i]),np.sin(theta[i]),0.0])
    start_link_1 = origin
    end_link_1 = origin + r_10
    start_motor = origin - 0.02*scale*r_10/np.linalg.norm(r_10)
    end_motor = origin + 0.03*scale*r_10/np.linalg.norm(r_10)
    phi = np.arctan(Lp*np.sin(alpha[i])/Lr) + theta[i]

    r_20 = np.sqrt(Lr**2 + (Lp*np.sin(alpha[i]))**2)*np.array([np.cos(phi),np.sin(phi),0.0]) + np.array([0.0,0.0,-Lp*np.cos(alpha[i])])
    r_21 = r_20 - r_10
    start_link_2 = origin + r_10 - 0.05*r_21
    end_link_2 = origin + r_10 + r_21
    X,Y,Z,X2,Y2,Z2,X3,Y3,Z3 = generate_cylinder(end_motor,end_link_1,0.007/2*scale)
    # rgb1 = ls.shade(Z)
    # rgb2 = ls.shade(Z2)
    # rgb3 = ls.shade(Z3)
    plot[0] = ax.plot_surface(X, Y, Z, color=u'#EEEEEE')
    plot[1] = ax.plot_surface(X2, Y2, Z2, color=u'#EEEEEE')
    plot[2] = ax.plot_surface(X3, Y3, Z3, color=u'#EEEEEE')
    X,Y,Z,X2,Y2,Z2,X3,Y3,Z3 = generate_cylinder(start_link_2,end_link_2,0.01/2*scale)
    plot[3] = ax.plot_surface(X, Y, Z, color=u'#EE1111')
    plot[4] = ax.plot_surface(X2, Y2, Z2, color=u'#EE1111')
    plot[5] = ax.plot_surface(X3, Y3, Z3, color=u'#EE1111')
    X,Y,Z,X2,Y2,Z2,X3,Y3,Z3 = generate_cylinder(start_motor,end_motor,0.0254/2*scale)  
    plot[6] = ax.plot_surface(X, Y, Z, color=u'#EE1111')
    plot[7] = ax.plot_surface(X2, Y2, Z2, color=u'#EE1111')
    plot[8] = ax.plot_surface(X3, Y3, Z3, color=u'#EE1111')

    ## PARAMS
    for ind in range(6):
        axed = param_ax[ind]
        plt.axes(axed)
        param_ci[ind].remove()
        param_ci[ind] = plt.fill_between(ts[:i],tops[ind,:i],bots[ind,:i],color=u'#1f77b4',alpha=0.15)
        ln = param_ln[ind]
        ln[0].set_data(ts[:i],means[ind,:i])
        plt.ticklabel_format(style='sci',axis='y',scilimits=(1,-1))

    # STATES
    axed = state_ax[0]
    plt.axes(axed)
    ln = state_ln[0]
    ln[0].set_data(ts[:i],theta[:i])
    axed.relim()
    axed.autoscale()
    axed.set_xlim([0,49*0.025])
    axed = state_ax[1]
    plt.axes(axed)
    ln = state_ln[1]
    ln[0].set_data(ts[:i],alpha[:i])
    axed.relim()
    axed.autoscale()
    axed.set_xlim([0,49*0.025])
    axed = state_ax[2]
    plt.axes(axed)
    ln = state_ln[2]
    ln[0].set_data(ts[:i],ctrl[:i])
    axed.relim()
    axed.autoscale()
    axed.set_xlim([0,49*0.025])

    # HISTOGRAMS

    axed = hist_ax[0]
    plt.axes(axed)
    barrs = hist_pl[0]
    _ = [b.remove() for b in barrs] # wowza
    _,_,hist_pl[0] = axed.hist(alpha_est[:, i],bins=20,color = u'#1f77b4')
    ulim = np.percentile(alpha_est[:,i],97.5)
    llim = np.percentile(alpha_est[:,i],2.5)
    clim = alpha_est[:,i].mean()
    axed.axvline(np.pi,color='g',linestyle='--',linewidth=lwidth)
    axed.set_xlim([clim+5*(llim-clim),clim+5*(ulim-clim)])
   

    axed = hist_ax[1]
    plt.axes(axed)
    barrs = hist_pl[1]
    _ = [b.remove() for b in barrs] # wowza
    _,_,hist_pl[1] = axed.hist(theta_est[:, i],bins=20,color = u'#1f77b4')
    ulim = np.percentile(theta_est[:,i],97.5)
    llim = np.percentile(theta_est[:,i],2.5)
    clim = theta_est[:,i].mean()
    axed.axvline(-0.75*np.pi,color='r',linestyle='--',linewidth=lwidth)
    axed.axvline(0.75*np.pi,color='r',linestyle='--',linewidth=lwidth)
    axed.set_xlim([clim+5*(llim-clim),clim+5*(ulim-clim)])
    

ani = animation.FuncAnimation(fig, animate, frames=len(theta), fargs=(theta,alpha,origin,ts,plot,param_ax,param_ci,param_ln,means,tops,bots,state_ax,state_ln,ctrl,hist_ax,hist_pl,theta_est,alpha_est),
                              interval=1/30, blit=False)

ani.save('videos/pendulum_'+run+'_subplots.mp4',writer='ffmpeg',fps=30)
# plt.show()