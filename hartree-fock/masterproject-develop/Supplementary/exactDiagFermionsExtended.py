import inspect
import os
import sys

from RBM.FermionModel import FermionModel
import numpy as np
from scipy.linalg import eigh
import scipy
import time
from datetime import datetime
import matplotlib.pyplot as plt

np.set_printoptions(precision=16)
# from FermionModel import FermionModel

"""
fJ implementation as in (6.47)
"""

ff1 = lambda k, q: 1
ff2 = lambda k, q: 0.9 * np.sin(q) * (np.sin(k) + np.sin((k + q)))
ff3 = lambda k, q: 0
ff4 = lambda k, q: 0


def calculate_wf_div_1(k_index, vector2, k_value, kmq_index=None, l_value=None):
    boolean = 1
    if (
        l_value is None
        and kmq_index is None
        and k_index is not None
        and k_value is not None
    ):
        boolean *= vector2[k_index] == k_value
    elif (
        k_index is not None
        and kmq_index is not None
        and k_value is not None
        and l_value is not None
    ):
        boolean *= vector2[k_index] == k_value and vector2[kmq_index] == l_value
    else:
        raise ValueError
    return boolean


def get_index(a):
    """
    the notation is at follows:
    ++ -> 1,1 -> [0][0]
    -- -> -1,-1 -> [1][1]
    +- -> 1,-1 -> [0][1]
    -+ -> -1,1 -> [1][0]
    :param a: corresponds to +/-
    :return: matrix index 0,1
    :raises ValueError: if unexpected value is given
    """
    if a == 1:
        return 0
    if a == -1:
        return 1
    raise ValueError("expected 'spin value' but received number that is not in [-1,1]")


def minus(index):
    """
    :param index: + -> [0] or - -> [1]
    :return: - if + is given and vice versa
    :raises ValueError: if unexpected value is given
    """
    if index not in [0, 1]:
        raise ValueError("expected index, no index was given")
    return int(not index)


def getMatrixElement(binary1, binary2, chain):
    # computes the matrix elements given in section 6.2.1
    # print(f"getMatrixElement({binary1}, {binary2}) = ", end="")
    vector1 = getConfiguration(binary1, chain.length)
    vector2 = getConfiguration(binary2, chain.length)
    resultKinetic = 0.0
    for k in chain.k:
        if not vector1[int(k[0])] == -vector2[int(k[0])]:
            continue
        equalIndicesMask = np.array(
            [0 if k1[0] == k[0] else 1 for k1 in chain.k], dtype=bool
        )
        if not np.all(vector1[equalIndicesMask] == vector2[equalIndicesMask]):
            continue
        else:
            resultKinetic += np.cos(k[1])
    resultKinetic *= chain.h

    resultInteraction = 0.0
    for q in chain.q:
        cp_k = 0.0
        for k in chain.k:
            # print(k, q)
            k_index = int(k[0])
            kmq_index = chain.pbc(int(k[0]) - int(q[0]))
            # print(k[0], kmq_index)
            # print(vector2)
            # print(vector1)
            equalIndicesMask = np.array(
                [0 if (k1[0] == k[0] or k1[0] == kmq_index) else 1 for k1 in chain.k],
                dtype=bool,
            )
            # print(equalIndicesMask)
            if not np.all(vector1[equalIndicesMask] == vector2[equalIndicesMask]):
                continue
            bigF_k_mq = chain.bigF(k[1], -q[1])
            bigF_kmq_q = chain.bigF(k[1] - q[1], q[1])
            beta = -vector1[kmq_index]
            alpha = vector1[k_index]
            delta = vector2[k_index]
            a_index = get_index(alpha)
            b_index = get_index(beta)
            d_index = get_index(delta)
            if vector1[kmq_index] == vector2[kmq_index]:
                addTo = bigF_k_mq[b_index][d_index]
            else:  # vector1[kmq_index] == -vector2[kmq_index]
                addTo = -bigF_k_mq[minus(b_index)][d_index]
            addTo *= bigF_kmq_q[a_index][b_index]
            cp_k += addTo

        resultInteraction += q[2] * cp_k

    # print(resultInteraction + resultKinetic)
    # print(np.shape(resultInteraction))
    return resultInteraction + resultKinetic


def getBinaryBasis(N):
    # these are the 2^N basis states in the hilbert space
    return np.arange(2**N)


# def getBinary(chain):
#     list1 = [1 if x == 1 else 0 for x in chain.configuration]
#     return int("".join(map(str, list1)), 2)


def getConfiguration(binary, N):
    # translates configuration identifier to configuration
    stringRepresentation = format(binary, f"0{N}b")
    arrayRepresentation = np.array(
        [1 if string == "1" else -1 for string in stringRepresentation]
    )
    return arrayRepresentation


def constructMatrix(N, chain):
    # does not even use the fact that hamiltonian is hermitian and thus takes twice as long as constructMatrixHermitian
    configurationsVector = getBinaryBasis(N)
    return np.array(
        [
            [
                getMatrixElement(configuration1, configuration2, chain)
                for configuration2 in configurationsVector
            ]
            for configuration1 in configurationsVector
        ]
    )


def constructMatrixHermitian(N, chain):
    # provides the same output as constructMatrix
    # uses the fact that H is hermitian
    configurationsVector = getBinaryBasis(N)
    # print(configurationsVector)

    configurations_matrix = np.array(
        [
            [
                (configuration1, configuration2)
                for configuration2 in configurationsVector
            ]
            for configuration1 in configurationsVector
        ]
    )

    operator_matrix = np.array(
        [
            [
                (
                    getMatrixElement(configuration[0], configuration[1], chain)
                    if configuration[0] <= configuration[1]
                    else 0
                )
                for configuration in row
            ]
            for row in configurations_matrix
        ]
    )

    return operator_matrix + np.conj(np.triu(operator_matrix, 1).T)


def main(path0, ff1, ff2, ff3, ff4, potential_function, t, N, identifier, info_string):
    chain = FermionModel(
        potential_function=potential_function,
        ff1=ff1,
        ff2=ff2,
        ff3=ff3,
        ff4=ff4,
        h=float(t),
        length=N,
    )

    # print("f1")
    # for k in chain.k:
    #     for q in chain.q:
    #         print(ff1(k[1], -q[1]))
    #
    # print("f2")
    # for k in chain.k:
    #     for q in chain.q:
    #         print(ff2(k[1], -q[1]), k[1], q[1])
    # print(chain.potential)

    # result_path = (
    #     os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/RawResults/ED_Results/"
    # )
    # # result_path = ""
    # result_name = identifier + f"-N={N}_t={t:.5e}"
    start = datetime.now()
    operatorMatrixHermitian = constructMatrixHermitian(N, chain)
    # np.save(result_path + f"{result_name}.npy", operatorMatrixHermitian)
    # with open(result_path + f"{result_name}.dat", "a") as config_file:
    #     formfactor_string = [
    #         inspect.getsource(ff1),
    #         inspect.getsource(ff2), inspect.getsource(ff3),
    #         inspect.getsource(ff4),
    #         inspect.getsource(potential_function),
    #         f"t={t}\n",
    #         f"N={N}\n",
    #     ]
    #     config_file.writelines(formfactor_string + [info_string])
    d = np.real(operatorMatrixHermitian)
    # plt.imshow(d, interpolation="none", cmap="binary")
    # plt.colorbar()
    # plt.show()
    matrix = operatorMatrixHermitian 
    sp = np.count_nonzero(matrix)
    # If matrix element is 0 = > True = 1 
    # If matrix element is not zero = > False = 0
    matrixsize = matrix.size
    print(sp, sp/matrixsize, "sparsity")
    wh, vh = eigh(operatorMatrixHermitian)
    print(matrix)
    print(isinstance(matrix, scipy.sparse.sparray))
    print(scipy.sparse.issparse(matrix))
    # fp.write(f"{numberBZ}\t{wh[0]}\n")
    fp.write(f"\n{t}\t{wh[0]/N}")
    print("Ground state energy: " + str(wh[0] / (N)))

    print(path0 + "ed.npy")
    np.save(
        path0 + "ed.npy",
        np.array([wh[0] / (N)]),
    )
    return wh[0]


ti = float(sys.argv[1])
N = int(sys.argv[2])
path0 = str(sys.argv[3])

identifier = f"original{0}_N={N}"
# ff1 = lambda k, q: randomVar1
# ff2 = (
#     lambda k, q: randomVar2 * np.sin(q) * (np.sin(k) + np.sin((k + q)))
#     + randomVar3
#     * np.sin(randomFreq1 * q)
#     * (np.sin(randomFreq2 * k) + np.sin(randomFreq2 * (k + q)))
#     + randomVar4
#     * np.sin(randomFreq3 * q)
#     * (np.sin(randomFreq4 * k) + np.sin(randomFreq4 * (k + q)))
# )
# ff3 = lambda k, q: 0
# ff4 = lambda k, q: 0
potential_function = lambda q, N: 1 / (1 + q * q) / (2 * N)

randomVar1 = 1
randomVar2 = 1
randomVar3 = 0
randomVar4 = 0
randomVar5 = 0
randomVar6 = 0
randomVar7 = 0
randomFreq1 = 0
randomFreq2 = 0
randomFreq3 = 0
randomFreq4 = 0
randomFreq5 = 0

info_string = f"randomVar1 = {randomVar1}\nrandomVar2 = {randomVar2}\nrandomVar3 = {randomVar3}\nrandomVar4 = {randomVar4}\nrandomVar5 = {randomVar5}\nrandomVar6 = {randomVar6}\nrandomVar7 = {randomVar7}\n"
info_string += f"randomFreq1 = {randomFreq1}\nrandomFreq2 = {randomFreq2}\nrandomFreq3 = {randomFreq3}\nrandomFreq4 = {randomFreq4}\nrandomFreq5 = {randomFreq5}\n"

fp = open("energyed.dat", "a")
for t in [ti]:
    E_ED = main(
        path0, ff1, ff2, ff3, ff4, potential_function, t, N, identifier, info_string
    )
fp.close()
