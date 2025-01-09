import os

import numpy as np
from matplotlib import pyplot as plt

from HartreeFock.HFNumerics import HartreeFockCalculations
from RBM.FermionModel import FermionModel

hf_path = "C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\HF_Results"
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def get_mm_list(list_of_matrices, elem1, elem2):
    return np.array([matrix[elem1][elem2] for matrix in list_of_matrices])


def plot_projector_and_untiray(identifier, t_list):
    hf_result_folders = list(filter(lambda x: "zzz" not in x and identifier in x, os.listdir(hf_path)))
    N = int(identifier[identifier.find("N=")+len("N="):identifier.find("-N")])
    for hf_result in hf_result_folders:
        for t in t_list:
            try:
                print("getting unitaries and k points")
                k_points, projector_list_of_matrices, unitary_list_of_matrices = get_projector_unitaries_kpoints(hf_result, t)
                print("getting hf functional list")
                hf_dict = get_hartree_fock_functional(identifier, N, t)
                hf_functional_list_of_matrices = hf_dict["hf_functional_list"]
                fock_list_of_matrices = hf_dict["fock_list"]
                print("calculating eigenvalues")
                Dk_list_of_matrices = [[[np.linalg.eigh(hf_functional)[0][0] if np.linalg.eigh(hf_functional)[0][0] > np.linalg.eigh(hf_functional)[0][1] else np.linalg.eigh(hf_functional)[0][1], 0], [0, np.linalg.eigh(hf_functional)[0][0] if np.linalg.eigh(hf_functional)[0][0] < np.linalg.eigh(hf_functional)[0][1] else np.linalg.eigh(hf_functional)[0][1]]] for hf_functional in hf_functional_list_of_matrices]
                for ki, k in enumerate(k_points):
                    offset = 0.5 * np.trace(projector_list_of_matrices[ki].T @ fock_list_of_matrices[ki])
                    Dk_list_of_matrices[ki][0][0] -= offset
                    Dk_list_of_matrices[ki][1][1] -= offset
            except ValueError as e:
                print(e)
                continue
            """
            U_mm = get_mm_list(unitary_list_of_matrices, 1, 1)
            U_mp = get_mm_list(unitary_list_of_matrices, 1, 0)
            U_pm = get_mm_list(unitary_list_of_matrices, 0, 1)
            U_pp = get_mm_list(unitary_list_of_matrices, 0, 0)

            P_mm = get_mm_list(projector_list_of_matrices, 1, 1)
            P_mp = get_mm_list(projector_list_of_matrices, 1, 0)
            P_pm = get_mm_list(projector_list_of_matrices, 0, 1)
            P_pp = get_mm_list(projector_list_of_matrices, 0, 0)
            """

            print("making lists")
            """
            H_mm = get_mm_list(hf_functional_list_of_matrices, 1, 1)
            H_mp = get_mm_list(hf_functional_list_of_matrices, 1, 0)
            H_pm = get_mm_list(hf_functional_list_of_matrices, 0, 1)
            H_pp = get_mm_list(hf_functional_list_of_matrices, 0, 0)
            """
            print("...")

            Dk_mm = get_mm_list(Dk_list_of_matrices, 1, 1)
            Dk_mp = get_mm_list(Dk_list_of_matrices, 1, 0)
            Dk_pm = get_mm_list(Dk_list_of_matrices, 0, 1)
            Dk_pp = get_mm_list(Dk_list_of_matrices, 0, 0)


            """
            plt.plot(k_points, np.abs(U_pp), label="$|U_{++}|$", linestyle="-", alpha=0.7)
            plt.plot(k_points, np.abs(U_mp), label="$|U_{-+}|$", linestyle="--")
            plt.plot(k_points, np.abs(U_pm), label="$|U_{+-}|$", linestyle=":")
            plt.plot(k_points, np.abs(U_mm), label="$|U_{--}|$", linestyle="-.", alpha=0.7)
            plt.title(hf_result + f" t={t}")
            plt.legend()
            plt.show()

            plt.plot(k_points, np.real(U_pp), label="$Re\{U_{++}\}$", linestyle="-", alpha=0.7)
            plt.plot(k_points, np.real(U_mp), label="$Re\{U_{-+}\}$", linestyle="--")
            plt.plot(k_points, np.real(U_pm), label="$Re\{U_{+-}\}$", linestyle=":")
            plt.plot(k_points, np.real(U_mm), label="$Re\{U_{--}\}$", linestyle="-.", alpha=0.7)
            plt.title(hf_result + f" t={t}")
            plt.legend()
            plt.show()

            plt.plot(k_points, np.imag(U_mp), label="$Im\{U_{-+}\}$", linestyle="-", alpha=0.7)
            plt.plot(k_points, np.imag(U_pm), label="$Im\{U_{+-}\}$", linestyle="--")
            plt.plot(k_points, np.imag(U_pp), label="$Im\{U_{++}\}$", linestyle=":")
            plt.plot(k_points, np.imag(U_mm), label="$Im\{U_{--}\}$", linestyle="-.", alpha=0.7)
            plt.title(hf_result + f" t={t}")
            plt.legend()
            plt.show()

            plt.plot(k_points, np.real(P_mp), label="$Re\{P_{-+}\}$", linestyle="-", alpha=0.7)
            plt.plot(k_points, np.real(P_pm), label="$Re\{P_{+-}\}$", linestyle="--")
            plt.plot(k_points, np.real(P_pp), label="$Re\{P_{++}\}$", linestyle=":")
            plt.plot(k_points, np.real(P_mm), label="$Re\{P_{--}\}$", linestyle="-.", alpha=0.7)
            plt.title(hf_result + f" t={t}")
            plt.legend()
            plt.xticks([k for k in k_points[::2]], [f"{round(round(k, 10) / round(np.pi, 10), 2)}$\pi$" for k in k_points[::2]])
            plt.show()

            plt.plot(k_points, np.imag(P_mp), label="$Im\{P_{-+}\}$", linestyle="-", alpha=0.7)
            plt.plot(k_points, np.imag(P_pm), label="$Im\{P_{+-}\}$", linestyle="--")
            plt.plot(k_points, np.imag(P_pp), label="$Im\{P_{++}\}$", linestyle=":")
            plt.plot(k_points, np.imag(P_mm), label="$Im\{P_{--}\}$", linestyle="-.", alpha=0.7)
            plt.title(hf_result + f" t={t}")
            plt.legend()
            plt.xticks([k for k in k_points[::2]], [f"{round(round(k, 10) / round(np.pi, 10), 2)}$\pi$" for k in k_points[::2]])
            plt.show()
            """

            print("prepare plot")
            """
            plt.plot(k_points, np.real(H_mp), label="$Re\{H_{-+}\}$", linestyle="-", alpha=0.7)
            plt.plot(k_points, np.real(H_pm), label="$Re\{H_{+-}\}$", linestyle="--")
            plt.plot(k_points, np.real(H_pp), label="$Re\{H_{++}\}$", linestyle=":")
            plt.plot(k_points, np.real(H_mm), label="$Re\{H_{--}\}$", linestyle="-.", alpha=0.7)
            plt.title(hf_result + f" t={t}")
            plt.legend()
            plt.xticks([k for k in k_points[::2]], [f"{round(round(k, 10) / round(np.pi, 10), 2)}$\pi$" for k in k_points[::2]])
            plt.show()

            plt.plot(k_points, np.imag(H_mp), label="$Im\{H_{-+}\}$", linestyle="-", alpha=0.7)
            plt.plot(k_points, np.imag(H_pm), label="$Im\{H_{+-}\}$", linestyle="--")
            plt.plot(k_points, np.imag(H_pp), label="$Im\{H_{++}\}$", linestyle=":")
            plt.plot(k_points, np.imag(H_mm), label="$Im\{H_{--}\}$", linestyle="-.", alpha=0.7)
            plt.title(hf_result + f" t={t}")
            plt.legend()
            plt.xticks([k for k in k_points[::2]], [f"{round(round(k, 10) / round(np.pi, 10), 2)}$\pi$" for k in k_points[::2]])
            plt.show()
            """


            # plt.plot(k_points, np.real(Dk_mp), label="$Re\{Dk_{-+}\}$", linestyle="-", alpha=0.7)
            # plt.plot(k_points, np.real(Dk_pm), label="$Re\{Dk_{+-}\}$", linestyle="--")
            plt.plot(k_points, np.real(Dk_pp), label="$E^+_{k}$", linestyle="none", marker=".")
            plt.plot(k_points, np.real(Dk_mm), label="$E^-_{k}$", linestyle="none", marker=".")
            plt.plot(np.linspace(-np.pi, np.pi, 100), -t * np.cos(np.linspace(-np.pi, np.pi, 100)), color='black', label='$\pm t\cos(k)$', linestyle="--")
            plt.plot(np.linspace(-np.pi, np.pi, 100), t * np.cos(np.linspace(-np.pi, np.pi, 100)), color='black', linestyle="--")
            current_ylim = plt.ylim()
            # plt.vlines([-np.pi/2, np.pi/2], current_ylim[0], current_ylim[1], linestyles="--", colors="black", alpha=0.4)
            plt.ylim(current_ylim)
            # plt.title(hf_result + f" t={t}")
            plt.title(f"t={t}")
            plt.legend()
            plt.xlabel("$k$")
            # plt.xticks([k for k in k_points[::2]], [f"{round(round(k, 10) / round(np.pi, 10), 2)}$\pi$" for k in k_points[::2]])
            k_points_for_ticks = np.linspace(start=-np.pi, stop=np.pi, num=5)
            plt.grid()
            plt.xticks([k for k in k_points_for_ticks], [f"{round(round(k, 10) / round(np.pi, 10), 2)}$\pi$" for k in k_points_for_ticks])
            plt.tight_layout()
            # plt.savefig("C:\\Users\\Hester\\PycharmProjects\\masterproject\\Text\\" + identifier + f"_t={t}" + "_band_structure_cos" + ".eps")
            plt.show()


            """
            plt.plot(k_points, np.imag(Dk_mp), label="$Im\{Dk_{-+}\}$", linestyle="-", alpha=0.7)
            plt.plot(k_points, np.imag(Dk_pm), label="$Im\{Dk_{+-}\}$", linestyle="--")
            plt.plot(k_points, np.imag(Dk_pp), label="$Im\{Dk_{++}\}$", linestyle=":")
            plt.plot(k_points, np.imag(Dk_mm), label="$Im\{Dk_{--}\}$", linestyle="-.", alpha=0.7)
            plt.title(hf_result + f" t={t}")
            plt.legend()
            plt.xticks([k for k in k_points[::2]], [f"{round(round(k, 10) / round(np.pi, 10), 2)}$\pi$" for k in k_points[::2]])
            plt.show()
            """
            print("last plot finished")
            # return Dk_pp, Dk_mm


def get_projector_unitaries_kpoints(hf_result, t, hf_path=hf_path):
    unitaries = list(filter(lambda x: "Uk_N" in x and f"t={t:.5e}" in x, os.listdir(os.path.join(hf_path, hf_result))))
    projectors = list(
        filter(lambda x: "bin_projector_N" in x and f"t={t:.5e}" in x, os.listdir(os.path.join(hf_path, hf_result))))
    if len(unitaries) != 1 or len(projectors) != 1:
        raise ValueError(f"unable to plot {hf_result} with t={t:.5e}")
    N = int(unitaries[0][unitaries[0].find("_N=") + len("_N="):unitaries[0].find("_t=")])
    k_points = np.linspace(start=-np.pi, stop=np.pi * (1 - 2 / N), num=N)
    unitary_list_of_matrices = np.load(os.path.join(hf_path, hf_result, unitaries[0]))
    projector_list_of_matrices = np.load(os.path.join(hf_path, hf_result, projectors[0]))
    return k_points, projector_list_of_matrices, unitary_list_of_matrices



def get_hartree_fock_functional(identifier, N, t, p0=None, hf_path=hf_path):
    hf_result_folders = list(filter(lambda x: "zzz" not in x and identifier in x, os.listdir(hf_path)))
    if len(hf_result_folders) != 1:
        raise ValueError
    hf_result = hf_result_folders[0]
    k_points, projector_list_of_matrices, unitary_list_of_matrices = get_projector_unitaries_kpoints(hf_result, t)
    ff1 = lambda k, q: 1
    ff2 = lambda k, q: 1 * np.sin(q) * (np.sin(k) + np.sin((k + q)))
    ff3 = lambda k, q: 0
    ff4 = lambda k, q: 0
    potential_function = lambda q: 1 / (1 + q * q) / (2 * N)
    model = FermionModel(potential_function=potential_function,
                         ff1=ff1, ff2=ff2,
                         ff3=ff3, ff4=ff4,
                         h=float(t), length=N, sumOverG=False)
    HFClass = HartreeFockCalculations(model=model, p0=projector_list_of_matrices if p0 is None else p0)
    hf_functional_fock_list = [HFClass.get_hf_functional(k) for k in model.k]
    hf_functional_list = [functional_fock_tuple[0] for functional_fock_tuple in hf_functional_fock_list]
    fock_list = [functional_fock_tuple[1] for functional_fock_tuple in hf_functional_fock_list]
    return dict(p0_list=projector_list_of_matrices, hf_functional_list=hf_functional_list, unitary_list_of_matrices=unitary_list_of_matrices, fock_list=fock_list)


def get_interaction_edge_case_projector():
    return np.array([[0, 0],[0, 1]])

def get_kinetic_edge_case_projector():
    return np.array([[1, -1], [-1, 1]]) * 0.5

def compare_projector_to_edge_case(identifier, t_list):
    hf_result_folders = list(filter(lambda x: "zzz" not in x and identifier == x, os.listdir(hf_path)))
    fig, axes = plt.subplots(len(t_list), 2, figsize=(3*11.69, 3*8.27))
    fig_sum, axes_sum = plt.subplots(1, 1)
    diff_to_kinetic_edge_sum_list = []
    diff_to_interaction_edge_sum_list = []
    diff_to_interaction_edge_sum_list_2 = []
    # axes_iter = np.nditer(axes, flags=["refs_ok"])
    plot_counter = 0
    for hf_result in hf_result_folders:
        for t in t_list:
            # try:
            k_points, projector_list_of_matrices, unitary_list_of_matrices = get_projector_unitaries_kpoints(hf_result, t)
            # except ValueError as e:
            #     print(e)
            #     if t == t_list[-1] or t == t_list[0]:
            #         raise ValueError("Not able to lable correctly. First and last t value must be plottable")
            #     continue
            diff_to_kinetic_edge = [np.linalg.norm(np.abs(projector_k)-np.abs(get_kinetic_edge_case_projector())) for projector_k in projector_list_of_matrices]
            diff_to_interaction_edge = [np.linalg.norm(np.abs(projector_k) - get_interaction_edge_case_projector()) for projector_k in projector_list_of_matrices]
            diff_to_interaction_edge_2 = [np.linalg.norm(np.abs(projector_k) - sigma_x@get_interaction_edge_case_projector()@sigma_x) for projector_k in projector_list_of_matrices]
            axes[plot_counter][0].plot(k_points, diff_to_kinetic_edge)
            axes[plot_counter][0].set_ylabel("$||abs(P_k)-abs(P_0)||$")
            axes[plot_counter][1].set_ylabel(f"t={t:.5e}", rotation=0, labelpad=25, bbox = dict(facecolor='None', alpha=0.5))
            axes[plot_counter][1].plot(k_points, diff_to_interaction_edge)

            # for ax in axes[plot_counter]:
            #     ax.annotate(f't={t:.5e}', xy=(-24, 0), xycoords='axes points',
            #                 size=14, ha='right', va='top',
            #                 bbox=dict(boxstyle='round', fc='w'))
            if plot_counter == 0:
                axes[plot_counter][0].set_title("$P_0 = P_0^{kin}$")
                axes[plot_counter][1].set_title("$P_0 = P_0^{pot}$")
            if t == t_list[-1]:
                axes[plot_counter][0].set_xlabel("$k$")
                axes[plot_counter][1].set_xlabel("$k$")
            diff_to_kinetic_edge_sum_list.append(sum(diff_to_kinetic_edge)/len(k_points))
            diff_to_interaction_edge_sum_list.append(sum(diff_to_interaction_edge)/len(k_points))
            diff_to_interaction_edge_sum_list_2.append(sum(diff_to_interaction_edge_2) / len(k_points))
            plot_counter += 1
    fig.suptitle(f"Deviation of converged HF projector $P_k$ from kinetic and potential edge case formfactor label: {identifier}")
    fig.tight_layout()
    axes_sum.plot(t_list, diff_to_kinetic_edge_sum_list, label="$P_0 = P_0^{kin}$", marker="x")
    axes_sum.plot(t_list, diff_to_interaction_edge_sum_list, label="$P_0 = P_0^{pot}$", marker="o", fillstyle="none", linestyle="none")
    axes_sum.plot(t_list, diff_to_interaction_edge_sum_list_2, label="$P_0 = \sigma_x P_0^{pot} \sigma_x$", marker="o", fillstyle="none", linestyle="none")
    final_line_list = [min(diff_to_interaction_edge_sum_list[0],diff_to_interaction_edge_sum_list_2[0])]
    for i in range(1, len(diff_to_interaction_edge_sum_list)):
        final_line_list.append(diff_to_interaction_edge_sum_list[i] if abs(final_line_list[i-1]-diff_to_interaction_edge_sum_list[i]) < abs(final_line_list[i-1]-diff_to_interaction_edge_sum_list_2[i]) else diff_to_interaction_edge_sum_list_2[i])
    axes_sum.plot(t_list, final_line_list, color="black")
    axes_sum.set_xlabel("$t$")
    axes_sum.set_ylabel("$\sum_k||abs(P_k)-abs(P_0)||/N$")
    axes_sum.legend()
    axes_sum.grid()
    # axes_sum.set_xlim(0.08, 0.11)
    axes_sum.set_xscale("log")
    # fig_sum.suptitle(f"Deviation of converged HF projector $P_k$ from kinetic and potential edge case \n formfactor label: {identifier}")
    # fig_sum.suptitle(f"$N={N}$")
    fig_sum.tight_layout()
    # fig_sum.show()
    fig_sum.savefig(identifier+"_hf_projector.jpg")
    # fig.show()
    return fig, fig_sum

def matrix_to_latex(B, decimals):
    A = np.round(B, decimals)
    latex_str = r"\begin{pmatrix} " + f"{A[0][0]} & {A[0][1]} \\\\ {A[1][0]} & {A[1][1]}" + r" \end{pmatrix}"
    return latex_str

# print(matrix_to_latex(np.array([[1,2],[3,4]])))

# N_list = [8]
# for N in N_list:
#     identifier = f"original0_p0rand_longConv_final_N={N}-N={N}"
#     t_list = sorted([float(s[s.find("t=") + len("t="):s.find(".npy")]) for s in list(filter(lambda x: "bin_projector" in x, os.listdir(f"C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\HF_Results\\{identifier}")))])
#     fig, fig_sum = compare_projector_to_edge_case(identifier, t_list)
#     # fig_sum.savefig("C:\\Users\\Hester\\PycharmProjects\\masterproject\\Text\\" + "projector_convergence" + identifier + ".eps")
#     if N != N_list[-1]:
#         plt.close(fig)
#         plt.close(fig_sum)
# print("finished")


def print_matrix_list_to_latex(list_of_matrices, label, decimals=10):
    latex_str = "\\begin{align*}"
    for i, matrix in enumerate(list_of_matrices[:-1]):
        latex_str += f"{label}_{i} &= " + matrix_to_latex(matrix, decimals) + "\\\\"
    latex_str += f"{label}_{len(list_of_matrices) - 1} &= " + matrix_to_latex(list_of_matrices[-1], decimals) + "\end{align*}"
    print(latex_str.replace("+0j)", ")").replace("-0j)", ")").replace("(-0)", "0j").replace("(+0)", "0j").replace("-0j","0j"))
    

# N = 8
# t = 0.16
# identifier = f"original0_p0rand_longConv_final_N={N}-N={N}"
# k_points, projector_list_of_matrices, unitary_list_of_matrices = get_projector_unitaries_kpoints(identifier, t)
# print_matrix_list_to_latex(unitary_list_of_matrices, "U")
# print_matrix_list_to_latex(projector_list_of_matrices, "P")
# transformation_list_of_matrices = [np.conj(unitary).T@(np.eye(2)+1j*sigma_z)@unitary for unitary in unitary_list_of_matrices]
# print_matrix_list_to_latex(transformation_list_of_matrices, "S")


# N = 20
# identifier = f"original0_p0rand_longConv_final_N={N}-N={N}"
# # # # t_list = sorted([float(s[s.find("t=") + len("t="):s.find(".npy")]) for s in list(filter(lambda x: "bin_projector" in x, os.listdir(f"C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\HF_Results\\{identifier}")))])
# t_list = [0]
# plot_projector_and_untiray(identifier, t_list)
# print("finished")
