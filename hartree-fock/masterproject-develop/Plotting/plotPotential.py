import inspect

import numpy as np
from matplotlib import pyplot as plt

from RBM.FermionModel import FermionModel


# def potentialInsights(potential_function, N):
#     model = FermionModel(potential_function, lambda k, q: 1,
#                          lambda k, q: np.sin(q) * (np.sin(k) + np.sin(k + q)), lambda k, q: 0, lambda k, q: 0, 1.0,
#                          length=N)
#     potential_sum = 0
#     integral = 0
#     delta_q = abs(model.q[0][1]-model.q[1][1])
#     for q in model.q:
#         potential_sum += model.potential(q[1])
#         integral += model.potential(q[1]) * delta_q
#     return potential_sum, integral


def __construct_q(N, numberBZ, potential):
    q_pot = []
    q_index_blank = []
    for i in range(-numberBZ * N,
                   numberBZ * N + 1):
        if i % N:  # q must not be elem RL
            potential_q = potential(i * 2 * np.pi / N)
            q_pot.append(potential_q)
            q_index_blank.append(i)
    return np.array((q_index_blank, np.array([i * 2 * np.pi / N for i in q_index_blank]), q_pot)).T

N = 20
numberBZ = 3
potential_function = lambda q: 1/(q*q+1)/N
q_arr = __construct_q(N, numberBZ, potential_function)

plt.plot([q[1] for q in q_arr], [q[2] for q in q_arr], linestyle="None", marker=".", label="$V(q)$")
plt.legend()
xticks = [np.pi * a for a in range(-6,6+1)]
plt.xticks(xticks, [f"{round(round(q, 10) / round(np.pi, 10), 2)}$\pi$" for q in xticks])
plt.xlabel("$q$")
# plt.ylabel("$V(q)$")
# plt.title("$V(q)$")
plt.tight_layout()
plt.savefig("potential.pdf")
plt.show()

# print(potentialInsights(potential_function, N))

# potential_function = lambda q: 1/(1 + q * q * (2 * np.pi / N)**2)/(2*N)
# q_arr = __construct_q(N, numberBZ, potential_function)
# plt.plot([q[1] for q in q_arr], [q[2] for q in q_arr], label="potential")
# plt.legend()
# plt.xticks([q[1] for q in q_arr[::10]], [f"{round(round(q[1], 10) / round(np.pi, 10), 2)}$\pi$" for q in q_arr[::10]])
# plt.title(inspect.getsource(potential_function))
# plt.show()
#
# print(potentialInsights(potential_function, N))
#
# potential_function = lambda q: 1/(1+q*q)
# q_arr = __construct_q(N, numberBZ, potential_function)
# potential_function_norm = sum(potential_function(q[1]) for q in q_arr)
# potential_function = lambda q: 1/(1+q*q)/(potential_function_norm)
# q_arr = __construct_q(N, numberBZ, potential_function)
# plt.plot([q[1] for q in q_arr], [q[2] for q in q_arr], label="potential", linestyle="None", marker="x")
# plt.legend()
# plt.xticks([q[1] for q in q_arr[::10]], [f"{round(round(q[1], 10) / round(np.pi, 10), 2)}$\pi$" for q in q_arr[::10]])
# plt.title(inspect.getsource(potential_function))
# plt.show()
#
# print(potentialInsights(potential_function, N))
#
# potential_function = lambda q: 1/(1+q*q)
# q_arr = __construct_q(N, numberBZ, potential_function)
# plt.plot([q[1] for q in q_arr], [q[2] for q in q_arr], label="potential", linestyle="None", marker="x")
# plt.legend()
# plt.xticks([q[1] for q in q_arr[::10]], [f"{round(round(q[1], 10) / round(np.pi, 10), 2)}$\pi$" for q in q_arr[::10]])
# plt.title(inspect.getsource(potential_function))
# plt.show()
#
# print(potentialInsights(potential_function, N))
