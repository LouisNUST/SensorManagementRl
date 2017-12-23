import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

filename = "out/locations_TFNeuralNetStochasticPolicyOTPSensor_8_1e-06_1e-10_10000_True_1.txt"
episode = 3651


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

data = []
with open(filename) as f:
    for line in f:
        if line.startswith(str(episode) + ','):
            vals = line.split(',')
            entry = []
            for n in vals:
                if is_int(n):
                    entry.append(int(n))
                else:
                    entry.append(float(n))
            data.append(np.array(entry, dtype=object))
data = np.array(data)

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot(data[:, 2], data[:, 3], data[:, 1], label='sensor')
ax.plot(data[:, 4], data[:, 5], data[:, 1], label='target truth')
ax.plot(data[:, 6], data[:, 7], data[:, 1], label='target est.', linestyle=':')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('step')
ax.legend()

fig = plt.figure()
ax = fig.gca()
ax.plot(data[:, 1], data[:, 8])
ax.grid(True)
ax.set_xlabel('step')
ax.set_ylabel('bearing')

fig = plt.figure()
ax = fig.gca()
ax.plot(data[:, 2], data[:, 3], label='sensor')
ax.plot(data[:, 4], data[:, 5], label='target truth')
ax.plot(data[:, 6], data[:, 7], label='target est.', linestyle=':')
ax.grid(True)
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')

for d in data:
    print(','.join(str(i) for i in d))

plt.show()
