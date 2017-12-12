import matplotlib.pyplot as plt
import pylab

data = pylab.loadtxt('plotdata.txt', delimiter=',')

pylab.plot(data[:, 0], data[:, 1], 'ko-', linewidth=3)

plt.xlabel("iteration", size=15)
plt.ylabel("avg. reward", size=15)
plt.grid(True)

plt.show()
