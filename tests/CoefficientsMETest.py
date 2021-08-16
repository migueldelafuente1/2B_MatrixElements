'''
Created on Mar 12, 2021

@author: Miguel
'''
import unittest
import numpy as np
import time
from matrix_elements.transformations import TalmiTransformation
from helpers.Helpers import fact, gamma_half_int, shellSHO_Notation
from helpers.integrals import _RadialTwoBodyDecoupled
from helpers.WaveFunctions import QN_1body_radial


class BCoefficientTest(unittest.TestCase):

    def setUp(self):
        self._tik = time.time()
        self._numberBs = 0


    def tearDown(self):

        total_time = time.time() - self._tik

        args = (round(1e+6 * total_time/(self._numberBs + 1), 3),
                round(self._numberBs/total_time, 1),
                self._numberBs,
                round(total_time, 6))
        print()
        print("[{}]us/Bs [{}] Bs/s -> [{}] B Coeffs calculated:: [{}] seconds"
              .format(*args))


    def test_B_diagonal_lqEqualsl_withBookValues(self):
        """ B(n, l, n', l'=l, p) """

        benchmarks = [
            # n'=n, l'=l
            ((0,0,0,0, 0), 1.00000),
            ((1,0,1,0, 0), 1.5), 
            ((2,0,2,0, 0), 1.875),
            ((4,1,4,1, 2), -72.187499),
            ((5,1,5,1, 6), -11133.892),
            ((3,6,3,6, 7), -605.62498),
            ((1,10,1,10, 12), 12.5),
            # n'=n-1, l'=l+2
            ((2,0,1,2, 3), 11.244443),
            ((2,0,1,2, 4), -5.9529404),
            ((5,1,4,3, 5), -3163.8604),
            ((1,6,0,8, 7), 2.5724789),
            ((3,6,2,8, 10), -879.75715),
            ((1,10,0,12, 12), -3.5355338)
        ]

        fails = []
        tmp_f = "B(nln'l', p): B{} = [{}], but got [{}]"
        for qqnns, value_bench in benchmarks:

            val = TalmiTransformation.BCoefficient(*qqnns)
            if abs(val - value_bench)/value_bench > 1e-9:
                fails.append(tmp_f.format(qqnns, value_bench, val))

        # Print the book B Coeffs to compare
        self._print_Book_BCoefficients()

        self.assertTrue(len(fails) == 0, '\n'+'\n'.join(fails))


    def _print_Book_BCoefficients(self):

        rho = 12
        templ = "{p}  {n}  {l}      {value}"
        line = "#==========================================================================="

        diag_list = [line+"\n# n'=n && l'=l     B COEFFICIENTS\n"+line]
        offd_list = [line+"\n# n'=n-1 && l'=l+2 B COEFFICIENTS\n"+line]

        for p in range(rho +1):
            print_p, print_poff = True, True

            diag_list.append('')
            offd_list.append('')
            for l in range(p +1):
                for n in range(rho//2 +1):
                    if 2*n + l > rho:
                        continue

                    p_str = ' '
                    diag = TalmiTransformation.BCoefficient(n, l, n, l, p)

                    if abs(diag) > 1e-8:
                        if print_p:
                            p_str = p
                            print_p = False
                        diag_list.append(templ.format(p=p_str, n=n, l=l, 
                                                      value= round(diag,7)))

                    p_str = ' '
                    offd = TalmiTransformation.BCoefficient(n, l, n-1, l+2, p)
                    if abs(offd) > 1e-8:
                        if print_poff:
                            p_str = p
                            print_poff = False
                        offd_list.append(templ.format(p=p_str, n=n, l=l, 
                                                      value= round(offd,7)))

                    self._numberBs += 2

        with open("TablesOfB(nln'l',p)_BMbook.txt", "+w") as f:
            f.write('\n'.join(diag_list) + '\n\n' + '\n'.join(offd_list))


class BDecoupledCoefficientsTest(unittest.TestCase):
    
    def _phiWF(self, n, l, r):
        
        aux = 0.0 * r
        for p in range(0, n +1):
            den = fact(p) + fact(n - p) + gamma_half_int(2*(p + l) + 3)
            aux += ((-1)**p) * (r**(2*p + l)) / np.exp(den)
        N = np.exp(0.5 * (fact(n) + gamma_half_int(2*(n + l) + 3) + np.log(2)))
        return aux * N
    
    def test_additionWF(self):
        n1, l1 = 1, 5
        n2, l2 = 2, 3
        
        dr = 0.001
        r = np.arange(dr, 6+dr, dr)
        
        f1 = self._phiWF(n1, l1, r) / np.exp(0.5 * r**2)
        f2 = self._phiWF(n2, l2, r) / np.exp(0.5 * r**2)
        f1f2 = f1 * f2
        
        Integrals_ = _RadialTwoBodyDecoupled()
        aux = 0.0 * r
        print("\nSeries of B(n{} l{},n{} l{}, p)::".format(n1,l1,n2,l2))
        for p in range(0, n1 + n2 +1):
            Bp = Integrals_._B_coeff(n1, l1, n2, l2, p)
            print("  B(nl(1){},{} nl(2){},{}, p{})={}".format(n1,l1,n2,l2,p,Bp))
            aux += Bp * (r**(2*p + l1 + l2))
        
        aux /= np.exp(r**2)
        import matplotlib.pyplot as plt
        
        plt.plot(r, f1, 'r--', label='f1({})'.format((n1, l1)))
        plt.plot(r, f2, 'b--', label='f2({})'.format((n2, l2)))
        plt.plot(r, f1f2, label='Analytical f1*f2')
        plt.plot(r, aux, label='f1*f2 from sum')
        plt.title("Checking the product of two radial wave functions [{}, {}]"
                  .format(shellSHO_Notation(n1,l1),
                          shellSHO_Notation(n2,l2)))
        plt.xlabel("r")
        plt.legend()
        plt.show()
        
        diff = sum((aux - f1f2)) * dr
        self.assertAlmostEqual(diff, 0.0, msg="Not equal", delta=1e-8)
        # if abs(diff) > 1e-7:
    
    def test_printBCoeffsAndExchanged(self):
        """ Checking the permutation properties of the B coefficients """
        
        self._printExchangedBCoefficients(5)
        
    def _printExchangedBCoefficients(self, Nmax):
        print("\n==== TEST PRINTING DECOUPLED B COEFFS AND EXCHANGED ==== CHECK SYMMETRY ====\n")
        Integrals_ = _RadialTwoBodyDecoupled()
        for n1 in range(Nmax//2 +1):
            l1 = Nmax - 2*n1
            
            for n2 in range((Nmax + 3)//2 +1):
                l2 = (Nmax + 3) - 2*n2
                print("Test n1,l1,n2,l2 =", n1, l1, n2, l2)
                for p in range(n1+n2 +1):
                    B0 = Integrals_._B_coeff(n1, l1, n2, l2, p)
                    Bexch = Integrals_._B_coeff(n2, l2, n1, l1, p)
                    print("  p={} B[n1l1({},{})n2l2({},{}):{:7.5}  -> B[n1l1({},{})n2l2({},{}):{:7.5}"
                          .format(p, n1,l1,n2,l2, B0, n2,l2,n1,l1, Bexch))
                print()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    