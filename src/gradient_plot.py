# Python-matplotlib Commands
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from utils import h
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(0, 1, .01)
Y = np.arange(0, 1, .01)
X, Y = np.meshgrid(X, Y)
Z = 1 - X + Y
Col = -(X*np.log(X) + Y*np.log(Y) + (1-X-Y)*np.log(1-X-Y))
Gx, Gy = np.gradient(Col) # gradients with respect to x and y
G = (Gx**2+Gy**2)**.5  # gradient magnitude
#N = G/G.max()  # normalize 0..1
N = G/0.5  # normalize 0..1
surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    facecolors=cm.jet(Col),
    linewidth=0, antialiased=False, shade=False)
plt.show()
