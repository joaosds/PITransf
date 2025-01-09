import json
import os

from HartreeFock.HF_main import main as hf_main
from HartreeFock.HF_main import scanForInterestingMagnitude as scan
from Supplementary.exactDiagFermionsExtended import main as ed_main
import testFormFactors
import random
import numpy as np
sin = np.sin
cos = np.cos


"""
fJ a fun little script where I wanted to find formfactors that yield highest difference between ED and HF energy. see outlook of my thesis:
fJ "In the context of the sampling algorithm we used within this thesis, we further could try
fJ to use form factors and a potential function which increase the relative error of the HFapproximation.
fJ Simply put, the idea is that a stronger deviance of the HF approximation
fJ to the exact solution could improve the efficiency of the Metropolis algorithm using the
fJ presented local update scheme."
fJ this is also the reason why the constants in the formfactors are called "rnadomVar" and "randomFreq". This did not enter my thesis and at some point I stopped working with the file
fJ ofc you can revive it but you have to adapt calculations if (6.15) is violated
"""

N_arr = [10]
result_summery={}

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

ff1 = lambda k, q: randomVar1
ff2 = lambda k, q: randomVar2 * np.sin(q) * (np.sin(k) + np.sin((k + q))) + randomVar3 * np.sin(randomFreq1 * q) * (
            np.sin(randomFreq2 * k) + np.sin(randomFreq2 * (k + q))) + randomVar4 * np.sin(randomFreq3 * q) * (
                               np.sin(randomFreq4 * k) + np.sin(randomFreq4 * (k + q)))
ff3 = lambda k, q: randomVar5 * (1 - np.cos(q)) + randomVar6 * (1 - np.cos(q)) * (
            np.cos((k + q)) + np.cos(k)) + randomVar7 * (1 - np.cos(randomFreq5 * q)) * (
                               np.cos(randomFreq5 * (k + q)) + np.cos(randomFreq5 * k))
# ff3 = lambda k, q: 0
ff4 = lambda k, q: 0
potential_function = lambda q: 1 / (1 + q * q) / (2 * N)

def given_list_of_t(t_arr):
    for t in t_arr:
        testClass.testProperties(ff1, ff2, ff3, ff4, potential_function, t, N, identifier)
        ed_energy = ed_main(ff1, ff2, ff3, ff4, potential_function, t, N, identifier, info_string)
        hf_energy = hf_main(ff1, ff2, ff3, ff4, potential_function, t, N, identifier, info_string)
        result_summery[identifier].append([t, str((ed_energy - hf_energy) / ed_energy) if ed_energy != 0 else "divBy0", str(ed_energy), str(hf_energy)])


def scan_for_relevant_range():
    scan(ff1=ff1, ff2=ff2, ff3=ff3, ff4=ff4, potential_function=potential_function, N=N, identifier=identifier, info_string=info_string)
    t_list = sorted([float(s[s.find("t=") + len("t="):s.find(".npy")]) for s in list(filter(lambda x: "bin_projector" in x, os.listdir(f"C:\\Users\\Hester\\PycharmProjects\\masterproject\\RawResults\\HF_Results\\{identifier}")))])
    for t in t_list:
        ed_main(ff1, ff2, ff3, ff4, potential_function, t, N, identifier, info_string)


for N in N_arr:


    inRange = range(0,1)

    for i in inRange:
        identifier = f"original0_N_p0_kin={N}"



        testClass = testFormFactors.Test()
        result_summery[identifier] = []
        # scan_for_relevant_range()
        given_list_of_t([0.11])

# json.dump(result_summery, open("result_summery_"+str(inRange), "a"))
# print("MF-results from being the least accurate to most accurate:")
# print(sorted(result_summery, key = lambda key: max([abs(complex(t_e_tuple[1].replace("(", "").replace(")", ""))) for t_e_tuple in result_summery.get(key)]), reverse = True))

