'''
Created on Mar 11, 2021

@author: Miguel
'''
import unittest
from matrix_elements.BM_brackets import BM_Bracket, _BMB_Memo
from helpers.Helpers import angular_condition, fact

import numpy as np
import time
from sys import getsizeof

class FactorialTestCase(unittest.TestCase):
    
    TOLERANCE = 1.0e-9
    
    def test_basefactorials(self):
        
        examples = [i for i in range(100)]
        
        tmp_fail = "[{}]! = [{}]. Got [{}]"
        
        checks = map(lambda x: (x, np.math.factorial(x), np.exp(fact(x))), examples)
        checks = filter(lambda f: abs(f[1] - f[2])/f[1] > self.TOLERANCE, checks)
        failures = [tmp_fail.format(*f) for f in checks]
                
        self.assertTrue(len(failures)==0, "\n"+"\n".join(failures))
        
        
    

class BrodyMoshinskyBracketsTestCase(unittest.TestCase):
    
    TOLERANCE = 1.0e-9
    TOLERANCE_BMB_BOOK = 1.0e-7
    
    def setUp(self):
        self._tik = time.time()
        self._numberBMBs = 0
    
    def _allPossibleQNforARho(self, rho, lambda_):
        
        for n1 in range(rho//2 +1):
            for n2 in range(rho//2 - n1 +1):
                for l1 in range(rho - 2*(n1 + n2) +1):
                    
                    l2 = rho - 2*(n1 + n2) - l1
                    
                    if not angular_condition(l1, l2, lambda_) or (l2 < 0):
                        continue
                    
                    yield (n1, l1, n2, l2)
        
    def _checkCompletnessConditionForBMB(self, rho):
        """ 
        For all n1, n2, l1, l2 compatible with a certain rho, evaluate 
        same/different COM quantum numbers for two states and verify orthogonality
        """
        max_lambda = 8
        
        fails = []
        _fail_msg_template = "[!= {}] (lambda={}) nlNL{} nlNL_prima{}, got: [{}]"
        for lambda_ in range(max_lambda):
            for nlNL in self._allPossibleQNforARho(rho, lambda_):
                for nlNL_prima in self._allPossibleQNforARho(rho, lambda_):
                    
                    orthogonality = []
                    for n1l1n2l2 in self._allPossibleQNforARho(rho, lambda_):

                        aux = BM_Bracket(*nlNL, *n1l1n2l2, lambda_) \
                            * BM_Bracket(*nlNL_prima, *n1l1n2l2, lambda_)
                        
                        self._numberBMBs += 2
                        
                        orthogonality.append(aux)
                    
                    _angular_cond_1 = angular_condition(nlNL[1], nlNL[3], lambda_)
                    _angular_cond_2 = angular_condition(nlNL_prima[1], 
                                                        nlNL_prima[3], lambda_)
                    _same_qns = nlNL == nlNL_prima
                    _result   = sum(orthogonality)
                    
                    if _angular_cond_1 and _angular_cond_2:
                        if _same_qns:
                            if abs(_result - 1)  > self.TOLERANCE:
                                fails.append(_fail_msg_template.format(
                                    1, lambda_, nlNL, nlNL_prima, _result))
                        else:
                            if abs(_result)  > self.TOLERANCE:
                                fails.append(_fail_msg_template.format(
                                    0, lambda_, nlNL, nlNL_prima, _result))
                    else:
                        if abs(_result) != 0:
                            fails.append(_fail_msg_template.format(
                                0, lambda_, nlNL, nlNL_prima, _result))
                        
        
        self.assertTrue(len(fails) == 0, "\n"+"\n".join(fails))
                            
                        
                        
    def test_checkCompletnessConditionForBMB_rho1(self):
        self._checkCompletnessConditionForBMB(1)
         
    def test_checkCompletnessConditionForBMB_rho2(self):
        self._checkCompletnessConditionForBMB(2)
     
    def test_checkCompletnessConditionForBMB_rho3(self):
        self._checkCompletnessConditionForBMB(3)
         
    def test_checkCompletnessConditionForBMB_rho4(self):
        self._checkCompletnessConditionForBMB(4)
         
    def test_checkCompletnessConditionForBMB_rho5(self):
        self._checkCompletnessConditionForBMB(5)
#           
#     def test_checkCompletnessConditionForBMB_rho6(self):
#         self._checkCompletnessConditionForBMB(6)
    
        
        
    def _check_InverseCompletnessConditionForBMB(self, rho):
        """ 
            Inverse 
        For all n,  l, N, L compatible with a certain rho, evaluate 
        same/different COM quantum numbers for two states and verify orthogonality
        """
        max_lambda = 6
        
        fails = []
        _fail_msg_template = "[!= {}] (lambda={}) nlNL{} nlNL_prima{}, got: [{}]"
        for lambda_ in range(max_lambda):
            for n1l1n2l2 in self._allPossibleQNforARho(rho, lambda_):
                for n1l1n2l2_prima in self._allPossibleQNforARho(rho, lambda_):
                    
                    orthogonality = []
                    for nlNL in self._allPossibleQNforARho(rho, lambda_):

                        aux = BM_Bracket(*nlNL, *n1l1n2l2, lambda_) \
                            * BM_Bracket(*nlNL, *n1l1n2l2_prima, lambda_)
                        
                        self._numberBMBs += 2
                        
                        orthogonality.append(aux)
                    
                    _angular_cond_1 = angular_condition(n1l1n2l2[1], 
                                                        n1l1n2l2[3], lambda_)
                    _angular_cond_2 = angular_condition(n1l1n2l2_prima[1], 
                                                        n1l1n2l2_prima[3], lambda_)
                    _same_qns = n1l1n2l2 == n1l1n2l2_prima
                    _result   = sum(orthogonality)
                    
                    if _angular_cond_1 and _angular_cond_2:
                        if _same_qns:
                            if abs(_result - 1)  > self.TOLERANCE:
                                fails.append(_fail_msg_template.format(
                                    1, lambda_, n1l1n2l2, n1l1n2l2_prima, _result))
                        else:
                            if abs(_result)  > self.TOLERANCE:
                                fails.append(_fail_msg_template.format(
                                    0, lambda_, n1l1n2l2, n1l1n2l2_prima, _result))
                    else:
                        if abs(_result) != 0:
                            fails.append(_fail_msg_template.format(
                                0, lambda_, n1l1n2l2, n1l1n2l2_prima, _result))
                            
        self.assertTrue(len(fails) == 0, "\n"+"\n".join(fails))
    
    def test_check_InverseCompletnessConditionForBMB_rho1(self):
        self._check_InverseCompletnessConditionForBMB(1)
     
    def test_check_InverseCompletnessConditionForBMB_rho2(self):
        self._check_InverseCompletnessConditionForBMB(2)
     
    def test_check_InverseCompletnessConditionForBMB_rho3(self):
        self._check_InverseCompletnessConditionForBMB(3)
      
    def test_check_InverseCompletnessConditionForBMB_rho4(self):
        self._check_InverseCompletnessConditionForBMB(4)
     
    def test_check_InverseCompletnessConditionForBMB_rho5(self):
        self._check_InverseCompletnessConditionForBMB(5)
      
    def test_check_InverseCompletnessConditionForBMB_rho6(self):
        self._check_InverseCompletnessConditionForBMB(6)
       
    def test_check_InverseCompletnessConditionForBMB_rho10(self):
        self._check_InverseCompletnessConditionForBMB(10)
    
    def _checkSpecificBMBs(self, list_bmbs):
        """
        :list_bmbs = <list>( <tuple>:(qq.nn args, correct_value))
        """
        
        fails = []
        _fail_msg_template = "n1l1n2l2{} (lambda={}) nlNL{} got [{}] != [{}]"
        
        for qq_nn, correct_value in list_bmbs:
            result_ = BM_Bracket(*qq_nn)
            if abs(correct_value - result_) > self.TOLERANCE_BMB_BOOK:
                fails.append(_fail_msg_template.format(qq_nn[4:8], qq_nn[-1],  
                                                       qq_nn[:4], 
                                                       result_, correct_value))
            
        self.assertTrue(len(fails) == 0, "\n"+"\n".join(fails))
        
        self._numberBMBs = len(list_bmbs)
    
    def test_specificBMBracketsWithLiterature(self):
        """ BM_Bracket(n,l, N,L,  n1,l1, n2,l2, lambda_) """
        benchmarks = [
            ((0,3,0,3, 0,3,0,3, 5), -0.49999998),
            ((0,4,0,2, 0,3,0,3, 5), 0.0),
            ((0,5,0,4, 0,3,0,6, 4), 0.03553346),
            ((1,1,0,3, 0,1,1,3, 2), 0.33541020),
            ((2,4,0,3, 1,3,1,4, 7), 0.14419055),
            ((1,1,2,0, 0,1,2,2, 1), -0.20916500),
            ((3,2,0,3, 2,2,1,3, 4), -0.16522103),
            ((1,2,3,2, 3,0,3,0, 0), -0.06487820)
        ]
        
        self._checkSpecificBMBs(benchmarks)
    
    def test_timeBMB_rho5(self):
        
        for _ in range(1000):
            BM_Bracket(1,1,2,0, 0,1,2,2, 1)
            self._numberBMBs += 1
    
    def test_timeBMB_rho6(self):
        
        for _ in range(100):
            BM_Bracket(1,1,0,3, 0,1,1,3, 2)
            self._numberBMBs += 1
        
    def test_timeBMB_rho7(self):
        
        for _ in range(1000):
            BM_Bracket(1,1,2,0, 0,1,2,2, 1)
            self._numberBMBs += 1
    
    def test_timeBMB_rho11(self):
        
        for _ in range(100):
            BM_Bracket(2,4,0,3, 1,3,1,4, 7)
            self._numberBMBs += 1
    
    def test_timeBMB_rho12(self):
        
        for _ in range(100):
            BM_Bracket(1,2,3,2, 3,0,3,0, 0)
            self._numberBMBs += 1
        
    
    def tearDown(self):
        """ 
        Get time for all BMBs calculated, notice that BMBs with invalid 
        conditions prompt 0 rapidly, the results are underestimated. 
        """
        
        total_time = time.time() - self._tik
        try:
            period_ = round(1e+6 * total_time/self._numberBMBs, 3)
            bmb_per_sec = round(self._numberBMBs/total_time, 1)
        except ZeroDivisionError as e:
            period_ = '--'
            bmb_per_sec = '--'
        
        args = (period_,
                bmb_per_sec,
                self._numberBMBs,
                round(total_time, 6))
        print()
        print("[{}]us/bmb [{}] bmbs/s -> [{}] BMBs calculated:: [{}] seconds"
              .format(*args))
        print('dim of BMB memo:', len(_BMB_Memo), 
              ', size:', getsizeof(_BMB_Memo)//(4*(1024)), 'KB')



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    