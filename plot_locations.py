import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pylab

data = pylab.loadtxt('plotdata.txt', delimiter=',')

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(data[:, 2], data[:, 3], data[:, 1], label='sensor')
ax.plot(data[:, 4], data[:, 5], data[:, 1], label='target')
ax.legend()

plt.show()
