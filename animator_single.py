import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# from matplotlib.colors import LightSource
import numpy as np
import pickle
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


run = 'run12'
with open('results/'+run+'/z_sim100.pkl','rb') as file:
    z_sim = pickle.load(file)

theta = z_sim[0,0,:].flatten()
alpha = z_sim[1,0,:].flatten()
old_lin = np.linspace(0,1,num = len(theta))
new_lin = np.linspace(0,1,num = 4*len(theta))
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

fig = plt.figure(figsize=(6.4,6.4),dpi=300)
ax = Axes3D(fig,azim=45,elev=30) #for on it's own

ax.voxels(qube,facecolors=colours,edgecolors=None)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

ax.set_xlim(2, 6)
ax.set_ylim(2, 6)
ax.set_zlim3d(0,3)

X,Y,Z,X2,Y2,Z2,X3,Y3,Z3 = generate_cylinder(origin_xy,origin,0.0254/2*scale)
ax.plot_surface(X, Y, Z, color=u'#777777')

# naughty ways of doing stuff and things
X,Y,Z,X2,Y2,Z2,X3,Y3,Z3 = generate_cylinder(origin,origin,0.1)
X4,Y4,Z4,X5,Y5,Z5,X6,Y6,Z6 = generate_cylinder(origin,origin,0.1)
X7,Y7,Z7,X8,Y8,Z8,X9,Y9,Z9 = generate_cylinder(origin,origin,0.1)
plot = [ax.plot_surface(X, Y, Z, color=u'#777777'),ax.plot_surface(X2, Y2, Z2, color=u'#777777'),ax.plot_surface(X3, Y3, Z3, color=u'#777777'), ax.plot_surface(X4, Y4, Z4, color=u'#777777'),ax.plot_surface(X5, Y5, Z5, color=u'#777777'),ax.plot_surface(X6, Y6, Z6, color=u'#777777'),ax.plot_surface(X7, Y7, Z7, color=u'#777777'),ax.plot_surface(X8, Y8, Z8, color=u'#777777'),ax.plot_surface(X9, Y9, Z9, color=u'#777777')]
#frame update function
def animate(i,theta,alpha,origin,plot):
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

ani = animation.FuncAnimation(fig, animate, frames=len(theta), fargs=(theta,alpha,origin,plot),
                              interval=100, blit=False)

ani.save('pendulum_'+run+'.mp4',writer='ffmpeg',fps=30)
# plt.show()