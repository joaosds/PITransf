import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as tck
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from RBM import FermionModel

t=1
N=5000
ff1 = lambda k, q: randomVar1
ff2 = lambda k, q: np.round(randomVar2 * np.sin(q) * (np.sin(k) + np.sin((k + q))) + randomVar3 * np.sin(randomFreq1 * q) * (np.sin(randomFreq2 * k) + np.sin(randomFreq2 * (k + q))) + randomVar4 * np.sin(randomFreq3 * q) * (np.sin(randomFreq4 * k) + np.sin(randomFreq4 * (k + q))),8)
ff3 = lambda k, q: 0
ff4 = lambda k, q: 0
potential_function = lambda q: 1/(q*q+1)/N
randomVar1 = 1
randomVar2 = 1
randomVar3 = 0.
randomVar4 = 0.
randomVar5 = 0.
randomVar6 = 0.
randomVar7 = 0.
randomFreq1 = 0.
randomFreq2 = 0.
randomFreq3 = 0.
randomFreq4 = 0.
randomFreq5 = 0.

NBZ=2
chain = FermionModel(potential_function=potential_function,
                     ff1=ff1, ff2=ff2,
                     ff3=ff3, ff4=ff4,
                     h=float(t), length=N, potential_over_brillouin_zones=NBZ)



k_values = [k[1] for k in chain.k]
q_values = [q[1] for q in chain.q]
K, Q = np.meshgrid(k_values, q_values)

function_dict = {
    "ff1": ff1,
    "ff2": ff2,
    "ff3": ff3,
    "ff4": ff4
}
fig, ax = plt.subplots(1, 1)
for function_name in function_dict.keys():
    F = function_dict[function_name](K,Q)
    try:
        cp = ax.contour(K, Q, F, sorted([0,0.4,0.8,1.2]+[-0.4,-0.8,-1.2]))
    except TypeError as e:
        if "Input z must be 2D" in str(e):
            print(f"0D and 1D plots are not possible in this setting, skip {function_name}")
            continue
        else:
            raise TypeError
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title("$f_2$")
    ax.set_xlabel('$k$')
    ax.set_ylabel('$q$')
    ax.grid()
    fig.tight_layout()
    fig.savefig(function_name + f"NBZ={NBZ}-contour.pdf")
    # fig.show()
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax1.plot_surface(K, Q, F, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.set_ylim(-4.01, 4.01)
    ax1.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax1.zaxis.set_major_formatter('{x:.02f}')
    ax1.set_title("$f_2$")
    ax1.set_xlabel('$k$')
    ax1.set_ylabel('$q$')
    # Add a color bar which maps values to colors.
    fig1.colorbar(surf, shrink=0.5, aspect=5)
    fig1.tight_layout()
    fig1.savefig(function_name + f"NBZ={NBZ}-surface.pdf")
    fig1.show()




