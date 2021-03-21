'''
Created on Mar 12, 2021

@author: Miguel
'''
import unittest
import time
from matrix_elements.transformations import TalmiTransformation


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
        
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    