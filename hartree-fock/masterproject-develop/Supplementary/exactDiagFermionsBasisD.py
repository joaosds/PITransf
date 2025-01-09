from RBM.FermionModel import FermionModel
import sys

# import FermionModel
import numpy as np
from scipy.linalg import eigh
from datetime import datetime
import matplotlib.pyplot as plt

"""
!!! Deprecated !!!
fJ Even tough I think that it provides correct output if (6.15) is fulfilled and thus provides equivalent results to exactDiagFermionsBasisD.py
"""

ff1 = lambda k, q: 1 
ff2 = lambda k, q: 0.9 * np.sin(q) * (np.sin(k) + np.sin(k + q))
ff3 = lambda k, q: 0
ff4 = lambda k, q: 0
potential_function = lambda q: 1 / (1 + q * q) / (2 * N)
# potential_function = lambda q: 1 / (1 + q * q)

t = 1.0
N = 6

"""
ff3 neq 0 or ff4 neq zero is not allowed here and will lead to wrong results.
Use exactDiagFermionsExtended.py for ff3 ff4 neq 0 (implements generalised matrix elements)
"""


def getMatrixElement(binary1, binary2, model):
    # print(f"getMatrixElement({binary1}, {binary2}) = ", end="")
    vector1 = getConfiguration(binary1)
    vector2 = getConfiguration(binary2)
    resultKinetic = 0.0
    for k in model.k:
        if not vector1[int(k[0])] == -vector2[int(k[0])]:
            continue
        equalIndicesMask = np.array(
            [0 if k1[0] == k[0] else 1 for k1 in model.k], dtype=bool
        )
        if not np.all(vector1[equalIndicesMask] == vector2[equalIndicesMask]):
            continue
        else:
            resultKinetic += np.cos(k[1])
    resultKinetic *= model.h

    resultInteraction = complex(0.0)
    for q in model.q:
        potential = q[2]
        for k in model.k:
            kmq_index = model.pbc(int(k[0]) - int(q[0]))
            if not (
                vector1[kmq_index] == -vector1[int(k[0])]
                and vector2[kmq_index] == -vector2[int(k[0])]
            ):
                continue
            equalIndicesMask = np.array(
                [0 if (k1[0] == k[0] or k1[0] == kmq_index) else 1 for k1 in model.k],
                dtype=bool,
            )
            if not np.all(vector1[equalIndicesMask] == vector2[equalIndicesMask]): continue
            if (
                vector1[int(k[0])] == vector2[int(k[0])]
                and vector1[kmq_index] == vector2[kmq_index]
            ):
                resultInteraction += (
                    model.ff1(k[1], -q[1]) ** 2 + model.ff2(k[1], -q[1]) ** 2
                ) * potential
            elif (
                vector1[int(k[0])] == -vector2[int(k[0])]
                and vector1[kmq_index] == -vector2[kmq_index]
            ):
                p = vector1[int(k[0])]
                resultInteraction += (
                    - model.ff1(k[1], -q[1]) ** 2
                    + model.ff2(k[1], -q[1]) ** 2
                    + complex(
                        0, 2 * p * model.ff2(k[1], -q[1]) * model.ff1(k[1], -q[1])
                    )
                ) * potential
            else:
                raise ValueError

    # h1aloc is also proven to be zero when using complex parameter
    h1aloc = 0.0
    if np.all(vector1 == vector2):
        for G in model.G:
            for k in model.k:
                for ks in model.k:
                    p = model.configuration[int(k[0])]
                    ps = model.configuration[int(ks[0])]
                    h1aloc += complex(
                        p * ps * model.ff2(k[1], G[1]) * model.ff2(ks[1], G[1]),
                        p * model.ff2(k[1], G[1]) * model.ff1(ks[1], G[1])
                        - ps * model.ff2(ks[1], G[1]) * model.ff1(k[1], G[1]),
                    )

    if abs(h1aloc) > 1e-13:
        raise ValueError

    # print(resultInteraction + resultKinetic)
    return resultInteraction + resultKinetic


def getBinaryBasis(N):
    return np.arange(2**N)


def getConfiguration(binary):
    stringRepresentation = format(binary, f"0{N}b")
    arrayRepresentation = np.array(
        [1 if string == "1" else -1 for string in stringRepresentation]
    )
    return arrayRepresentation


def constructMatrixHermitian(N, model):
    configurationsVector = getBinaryBasis(N)
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
                getMatrixElement(configuration[0], configuration[1], model)
                if configuration[0] <= configuration[1]
                else 0
                for configuration in row
            ]
            for row in configurations_matrix
        ]
    )

    return operator_matrix + np.conj(np.triu(operator_matrix, 1).T)


fp = open("energyOverNumberBZ.dat", "a")
fp.write("#BZ\tE\n")
for numberBZ in range(15, 16):
    print(numberBZ)
    model_obj = FermionModel(
        potential_function=potential_function,
        ff1=ff1,
        ff2=ff2,
        ff3=ff3,
        ff4=ff4,
        h=float(t),
        length=N,
        # potential_over_brillouin_zones=numberBZ,
    )
    start = datetime.now()
    operatorMatrixHermitian = constructMatrixHermitian(N, model_obj)
    print(np.shape(operatorMatrixHermitian))
    d = np.real(operatorMatrixHermitian)
    plt.imshow(d, interpolation="none", cmap="binary")
    plt.colorbar()
    plt.show()
    wh, vh = eigh(operatorMatrixHermitian)
    fp.write(f"{numberBZ}\t{wh[0]}\n")
    print("Ground state energy: " + str(wh[0]/N))
    print(f"success after {datetime.now() - start}")
    fp.flush()
fp.close()
