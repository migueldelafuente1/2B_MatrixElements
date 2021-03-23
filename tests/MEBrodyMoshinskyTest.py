'''
Created on Mar 19, 2021

@author: Miguel
'''
import unittest


from helpers.WaveFunctions import QN_2body_L_Coupling, QN_1body_radial,\
    QN_2body_LS_Coupling, QN_1body_jj, QN_2body_jj_JT_Coupling
from helpers.Enums import PotentialForms, ForceParameters

from matrix_elements.CentralForces import CentralForce
from matrix_elements.TensorForces import TensorForce
from matrix_elements.SpinOrbitForces import SpinOrbitForce
from pandas.core.computation.expressions import evaluate
import time
from matrix_elements.transformations import _TalmiTransformationBase
from matrix_elements.BM_brackets import _BMB_Memo
from sys import getsizeof

#         sp_state_1   = QN_1body_radial(1, 2)
#         sp_state_2   = QN_1body_radial(1, 3)
#         sp_state_1_k = QN_1body_radial(1, 2)
#         sp_state_2_k = QN_1body_radial(1, 3)
#         
#         sp_state_1   = QN_1body_radial(2, 0)
#         sp_state_2   = QN_1body_radial(0, 2)
#         sp_state_1_k = QN_1body_radial(0, 1)
#         sp_state_2_k = QN_1body_radial(1, 3)
#         
#         _bra = QN_2body_L_Coupling(sp_state_1, sp_state_2, 2)
#         _ket = QN_2body_L_Coupling(sp_state_1_k, sp_state_2_k, 2)
#                 
#         CentralForce.setInteractionParameters(
#             potential = PotentialForms.Power,
#             mu_length = 1)
#         _me = CentralForce(_bra, _ket)
#         print("me_central:", _me.value)
#         df = _me.getDebuggingTable('me_table.csv')
#         print(df)
        
#         sp_state_1   = QN_1body_radial(2, 0)
#         sp_state_2   = QN_1body_radial(0, 2)
#         sp_state_1_k = QN_1body_radial(0, 1)
#         sp_state_2_k = QN_1body_radial(1, 3)
#          
#         _bra = QN_2body_LS_Coupling(sp_state_1, sp_state_2, 2, 1)
#         _ket = QN_2body_LS_Coupling(sp_state_1_k, sp_state_2_k, 2, 1)
#          
#         TensorForce.setInteractionParameters(
#             potential = PotentialForms.Power,
#             mu_length = 1)
#         _me = TensorForce(_bra, _ket, 3)
        
#         print("me: ", _me.value)
#         df = _me.getDebuggingTable('me_table.csv')
#         print(df)
class Test(unittest.TestCase):

    TOLERANCE_BM_BOOK = 1.e-6
    
    def setUp(self):
        self._tik = time.time()
        self._numberMEs = 0


    def tearDown(self):
        total_time = time.time() - self._tik

        args = (round(1e+6 * total_time/self._numberMEs, 3),
                round(self._numberMEs/total_time, 1),
                self._numberMEs,
                round(total_time, 6))
        print()
        print("[{}]us/me [{}] me/s -> [{}] MEs calculated:: [{}] seconds"
              .format(*args))
        print('dim of BMB memo:', len(_BMB_Memo), 
              ', size:', getsizeof(_BMB_Memo)//(4*(1024)), 'KB')
    
    def __printCodeForTable(self, bra, ket, J, force):
        code_ = '({}_V{}_{})_L{}S{}J{}'
        
        return code_.format(bra.shellStatesNotation, force.lower(), 
                            ket.shellStatesNotation, bra.L, bra.S, J)
    
    def _checkSpecificMEs(self, list_bmbs):
        """
        :list_bmbs = <list>( <tuple>:(qq.nn args, correct_value))
        """
        
        fails = []
        _fail_msg_template = "<{} |V |{}> got [{}] != [{}]"
                
        for bra, ket, J, force, correct_value in list_bmbs:
            
            if force == ForceParameters.Central:
                CentralForce.setInteractionParameters(
                    potential = PotentialForms.Power,
                    mu_length = 1,
                    n_power=0)
                CentralForce.DEBUG_MODE = True
                me = CentralForce(bra, ket, run_it=False)
                result_ = me.value
                me_str = self.__printCodeForTable(bra, ket, J, force)
                
            elif force == ForceParameters.Tensor:
                TensorForce.setInteractionParameters(
                    potential = PotentialForms.Power,
                    mu_length = 1,
                    n_power=0)
                TensorForce.DEBUG_MODE = True
                me = TensorForce(bra, ket, J, run_it=False)
                result_ = me.value
                me_str = self.__printCodeForTable(bra, ket, J, force)
                
            elif force == ForceParameters.Spin_Orbit:
                SpinOrbitForce.setInteractionParameters(
                    potential = PotentialForms.Power,
                    mu_length = 1,
                    n_power=0)
                SpinOrbitForce.DEBUG_MODE = True
                me = SpinOrbitForce(bra, ket, J, run_it=False)
                result_ = me.value
                me_str = self.__printCodeForTable(bra, ket, J, force)
            
            self._numberMEs += 1
            
            print("me: ", me.value)
            df = me.getDebuggingTable('me_{}_table.csv'.format(me_str))
            
            if abs(correct_value - result_) > self.TOLERANCE_BM_BOOK:
                fails.append(_fail_msg_template.format(str(bra), str(ket), 
                                                       result_, correct_value))
            
        self.assertTrue(len(fails) == 0, "\n"+"\n".join(fails))
                
    
    def test_specific_NonAntisymmetrizedME_WithBook(self):
        """
        V(r) = 1, mu_param=1, b_length=1, for central interactions is 
        orthogonality condition on the two states.
        """
        # (bra, ket_, J, force_name, correct_value)
        benchmarks = [
#             (QN_2body_L_Coupling(QN_1body_radial(1,2), QN_1body_radial(1,3), 2),
#              QN_2body_L_Coupling(QN_1body_radial(1,2), QN_1body_radial(1,3), 2),
#              0, ForceParameters.Central, 1.0),
#             (QN_2body_L_Coupling(QN_1body_radial(2,0), QN_1body_radial(0,2), 2),
#              QN_2body_L_Coupling(QN_1body_radial(0,1), QN_1body_radial(1,3), 2),
#              0, ForceParameters.Central, 0.0),
#             (QN_2body_LS_Coupling(QN_1body_radial(2,0), QN_1body_radial(0,2), 2, 1),
#              QN_2body_LS_Coupling(QN_1body_radial(0,1), QN_1body_radial(1,3), 2, 1),
#              3, ForceParameters.Tensor, 0.01079898),
            (QN_2body_LS_Coupling(QN_1body_radial(2,2), QN_1body_radial(3,2), 2, 1),
             QN_2body_LS_Coupling(QN_1body_radial(2,1), QN_1body_radial(3,3), 2, 1),
             3, ForceParameters.Tensor, 0.01079898),
#             (QN_2body_LS_Coupling(QN_1body_radial(2,2), QN_1body_radial(2,2), 2, 1),
#              QN_2body_LS_Coupling(QN_1body_radial(2,2), QN_1body_radial(2,2), 2, 1),
#              2, ForceParameters.Spin_Orbit, 0.01079898)
        ]
        
        self._checkSpecificMEs(benchmarks)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()