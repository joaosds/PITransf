import numpy as np
from matplotlib import pyplot as plt

n_axis = np.arange(3, 30, 0.01)
plt.plot(n_axis, [2**n for n in n_axis], label="$2^N$")
plt.plot(n_axis, [n*n*1.5+n+n*1.5 for n in n_axis], label="Fully connected")
plt.xlabel("N")
plt.legend()
plt.yscale('log')
# plt.xticks(np.arange(0, 110, 10))
plt.show()

plt.plot(n_axis, [(n*n*1.5+n+n*1.5)/(2**n) for n in n_axis], label="$2^N/\#\Lambda$")
plt.xlabel("N")
plt.legend()
plt.grid()
plt.yscale('log')
# plt.xticks(np.arange(0, 110, 10))
plt.show()
