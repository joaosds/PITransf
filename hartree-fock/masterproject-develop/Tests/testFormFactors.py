import unittest

import numpy as np

from RBM.FermionModel import FermionModel

"""
fJ can be used that form factors fulfill conditions from section 2.2
fJ and can be used to ensure that (6.15) holds (not directly but with f2(k,G)=f3(k,G)=f4(k,G)=0 it should hold 6.15)
"""


class Test(unittest.TestCase):

    def testProperties(self, ff1, ff2, ff3, ff4, potential_function, t, N, identifier):

        identifier_msg = "in " + identifier

        chain = FermionModel(potential_function=potential_function,
                             ff1=ff1, ff2=ff2,
                             ff3=ff3, ff4=ff4,
                             h=float(t), length=N)

        for k in chain.k:
            for q in np.append(chain.q, chain.G, axis=0):
                self.assertAlmostEqual(ff1(k[1], q[1]), ff1(-k[1], -q[1]), delta=1e-12,
                                       msg="Error in ff1(-k,-q)==ff1(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff2(k[1], q[1]), ff2(-k[1], -q[1]), delta=1e-12,
                                       msg="Error in ff2(-k,-q)==ff2(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff3(k[1], q[1]), ff3(-k[1], -q[1]), delta=1e-12,
                                       msg="Error in ff3(-k,-q)==ff3(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff4(k[1], q[1]), ff4(-k[1], -q[1]), delta=1e-12,
                                       msg="Error in ff4(-k,-q)==ff4(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff1(k[1], q[1]), ff1(k[1] + q[1], -q[1]), delta=1e-12,
                                       msg="Error in ff1(k+q,-q)==ff1(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff2(k[1], q[1]), -ff2(k[1] + q[1], -q[1]), delta=1e-12,
                                       msg="Error in -ff2(k+q,-q)==ff2(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff3(k[1], q[1]), ff3(k[1] + q[1], -q[1]), delta=1e-12,
                                       msg="Error in ff3(k+q,-q)==ff3(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff4(k[1], q[1]), ff4(k[1] + q[1], -q[1]), delta=1e-12,
                                       msg="Error in ff4(k+q,-q)==ff4(k,q)" + identifier_msg)
                for G in chain.G:
                    # 2pi periodicity
                    self.assertAlmostEqual(ff1(k[1], q[1]+G[1]), ff1(k[1], q[1]), delta=1e-12,
                                           msg="Error in 2pi periodicity of ff1" + identifier_msg)
                    self.assertAlmostEqual(ff1(k[1]+G[1], q[1]), ff1(k[1], q[1]), delta=1e-12,
                                           msg="Error in 2pi periodicity of ff1" + identifier_msg)
                    self.assertAlmostEqual(ff2(k[1], q[1]+G[1]), ff2(k[1], q[1]), delta=1e-12,
                                           msg="Error in 2pi periodicity of ff2" + identifier_msg)
                    self.assertAlmostEqual(ff2(k[1]+G[1], q[1]), ff2(k[1], q[1]), delta=1e-12,
                                           msg="Error in 2pi periodicity of ff2" + identifier_msg)
                    self.assertAlmostEqual(ff3(k[1], q[1]+G[1]), ff3(k[1], q[1]), delta=1e-12,
                                           msg="Error in 2pi periodicity of ff3" + identifier_msg)
                    self.assertAlmostEqual(ff3(k[1]+G[1], q[1]), ff3(k[1], q[1]), delta=1e-12,
                                           msg="Error in 2pi periodicity of ff3" + identifier_msg)
                    self.assertAlmostEqual(ff4(k[1], q[1]+G[1]), ff4(k[1], q[1]), delta=1e-12,
                                           msg="Error in 2pi periodicity of ff4" + identifier_msg)
                    self.assertAlmostEqual(ff4(k[1]+G[1], q[1]), ff4(k[1], q[1]), delta=1e-12,
                                           msg="Error in 2pi periodicity of ff4" + identifier_msg)
            for G in chain.G:
                self.assertAlmostEqual(ff1(k[1], G[1]), ff1(-k[1], G[1]), delta=1e-12,
                                       msg="Error in ff1(-k,-q)==ff1(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff2(k[1], G[1]), -ff2(-k[1], G[1]), delta=1e-12,
                                       msg="Error in ff2(-k,-q)==ff2(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff3(k[1], G[1]), ff3(-k[1], G[1]), delta=1e-12,
                                       msg="Error in ff3(-k,-q)==ff3(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff4(k[1], G[1]), ff4(-k[1], G[1]), delta=1e-12,
                                       msg="Error in ff4(-k,-q)==ff4(k,q)" + identifier_msg)
                self.assertAlmostEqual(ff2(k[1], G[1]), 0, delta=1e-14,
                                       msg="Error in ff2(k,G)==0" + identifier_msg)
                self.assertAlmostEqual(ff3(k[1], G[1]), 0, delta=1e-14,
                                       msg="Error in ff3(k,G)==0" + identifier_msg)
                self.assertAlmostEqual(ff4(k[1], G[1]), 0, delta=1e-14,
                                       msg="Error in ff4(k,G)==0" + identifier_msg)

        return True

    if __name__ == '__main__':
        unittest.main()
