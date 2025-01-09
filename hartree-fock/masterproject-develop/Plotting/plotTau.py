import os
import numpy as np
from matplotlib import pyplot as plt

getH = lambda matrix: np.conj(matrix).T
pi = np.pi

path = "C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\HF_Results\\ff2 neq 0\\"
N = 10
h_list = [0,1,9]
k_values = np.linspace(start=-pi, stop=pi * (1 - 2 / N), num=N)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
sigma_z = np.array([[1, 0], [0, -1]])

U_files = sorted([file_name for file_name in os.listdir(path=path) if f"Uk_N={N}" in file_name], key= lambda file: float(file[file.find("t=") + len("t="):file.find(".npy")]))


for U_file in U_files:
    t = U_file[U_file.find("t=") + len("t="):U_file.find(".npy")]
    U = np.load(path + U_file)
    if float(t) not in h_list:
        continue
    for rotated in [0,1]:
        if rotated:
            phi_k = pi/2
            rot_matrix = np.diag([np.exp(complex(0, phi_k/2)), np.exp(complex(0, -phi_k/2))])
            U = [U_k @ rot_matrix for U_k in U]
        Tau = [getH(U_k) @ sigma_x @ U_k for U_k in U]
        plt.plot(k_values, [np.real(Tau_k[1][0]) for Tau_k in Tau], label="$Re\{\mathcal{T}_{-+}\}$", linestyle="None", marker="o", fillstyle="none")
        plt.plot(k_values, [np.real(Tau_k[0][1]) for Tau_k in Tau], label="$Re\{\mathcal{T}_{+-}\}$", linestyle="None", marker="x")
        plt.plot(k_values, [np.imag(Tau_k[1][0]) for Tau_k in Tau], label="$Im\{\mathcal{T}_{-+}\}$", linestyle="None", marker="o")
        plt.plot(k_values, [np.imag(Tau_k[0][1]) for Tau_k in Tau], label="$Im\{\mathcal{T}_{+-}\}$", linestyle="None", marker="o")
        plt.plot(k_values, np.sign(np.array([np.real(Tau_k[1][0]) for Tau_k in Tau])*np.cos(k_values)), label="$sign(Re\{\mathcal{T}_{-+}(k)\}\cos(k))$", linestyle="None", marker="s")
        plt.plot(k_values, np.cos(k_values), label="$\cos(k)$", linestyle="None", marker="s")
        plt.title(f"t={t}, rotated: {bool(rotated) if not bool(rotated) else f'by {round(phi_k/pi,2)}' + 'pi'}")
        plt.xlabel("k")
        plt.legend(loc=0)
        plt.grid()
        plt.xticks(k_values, [f"{round(round(k, 10) / round(np.pi, 10), 2)}$\pi$" for k in k_values])
        plt.show()
        plt.close()



