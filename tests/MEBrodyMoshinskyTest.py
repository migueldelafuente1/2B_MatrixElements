'''
Created on Mar 19, 2021

@author: Miguel
'''
import unittest


from helpers.WaveFunctions import QN_2body_L_Coupling, QN_1body_radial,\
    QN_2body_LS_Coupling, QN_1body_jj, QN_2body_jj_JT_Coupling
from helpers.Enums import PotentialForms, ForceEnum, AttributeArgs,\
    SHO_Parameters, CentralMEParameters

from matrix_elements.CentralForces import CentralForce
from matrix_elements.TensorForces import TensorForce
from matrix_elements.SpinOrbitForces import SpinOrbitForce
import time
from matrix_elements.transformations import _TalmiTransformationBase
from matrix_elements.BM_brackets import _BMB_Memo
from sys import getsizeof
from matrix_elements import switchMatrixElementType
from helpers.Helpers import valenceSpacesDict, readAntoine,\
    prettyPrintDictionary
from helpers.matrixElementHandlers import MatrixElementFilesComparator

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

    TOLERANCE_BM_BOOK = 1.0e-6
    TOLERANCE         = 5.0e-15
    
    def setUp(self):
        self._tik = time.time()
        self._numberMEs = 0
        self.failures = []
        print("\n***********************************************************\n")


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
    
    def _checkSpecificMEs(self, list_bmbs, ):
        """
        :list_bmbs = <list>( <tuple>:(qq.nn args, correct_value))
        """
        
        fails = []
        _fail_msg_template = "<{} |V |{}> got [{}] != [{}]"
        
        def aux_cond(self_):
            """ Central has JT scheme symmetrization inside, so L scheme is: """
            return True
        
        for bra, ket, J, force, correct_value in list_bmbs:
            
            if force == ForceEnum.Central:
                CentralForce._deltaConditionsForCOM_Iteration = aux_cond
                CentralForce.setInteractionParameters(
                    potential = PotentialForms.Power,
                    mu_length = 1,
                    n_power=0)
                CentralForce.DEBUG_MODE = True
                me = CentralForce(bra, ket, run_it=False)
                result_ = 0.5 * me.value # antisymmetrized
                me_str = self.__printCodeForTable(bra, ket, J, force)
                
            elif force == ForceEnum.Tensor:
                TensorForce._deltaConditionsForCOM_Iteration = aux_cond
                TensorForce.setInteractionParameters(
                    potential = PotentialForms.Power,
                    mu_length = 1,
                    n_power=0)
                TensorForce.DEBUG_MODE = True
                me = TensorForce(bra, ket, J, run_it=False)
                result_ = 0.5 * me.value
                me_str = self.__printCodeForTable(bra, ket, J, force)
                
            elif force == ForceEnum.SpinOrbit:
                SpinOrbitForce._deltaConditionsForCOM_Iteration = aux_cond
                SpinOrbitForce.setInteractionParameters(
                    potential = PotentialForms.Power,
                    mu_length = 1,
                    n_power=0)
                SpinOrbitForce.DEBUG_MODE = True
                me = SpinOrbitForce(bra, ket, J, run_it=False)
                result_ = 0.5 * me.value
                me_str = self.__printCodeForTable(bra, ket, J, force)
            
            self._numberMEs += 1
            
            print("me: ", me.value)
            # df = me.getDebuggingTable('me_{}_table{}.csv'.format(
            #                             me_str, time.strftime("%d%m")))
            
            if abs(correct_value - result_) > self.TOLERANCE_BM_BOOK:
                fails.append(_fail_msg_template.format(str(bra), str(ket), 
                                                       result_, correct_value))
            
        self.assertTrue(len(fails) == 0, "\n"+"\n".join(fails))
                
    def _testOrthogonality(self, a, b, c, d):
        
        J_min = max(abs(a.j - b.j), abs(c.j - d.j)) // 2
        J_max = min(a.j + b.j, c.j + d.j) // 2
        
        aux1 = (a.j == c.j) and (a.l == c.l) and (a.n == c.n)
        aux2 = (b.j == d.j) and (b.l == d.l) and (b.n == d.n)
        
        for J in range(J_min, J_max + 1):
            for T in (0, 1):
                bra = QN_2body_jj_JT_Coupling(a, b, J, T)
                ket = QN_2body_jj_JT_Coupling(c, d, J, T)
                
                if bra.nucleonsAreInThesameOrbit() or ket.nucleonsAreInThesameOrbit():
                    if (J + T) % 2 == 0: continue
                
                me = self.force(bra, ket)
                self._numberMEs += 1
                
                if (aux1 and aux2):
                    if abs(me.value - 1) > self.TOLERANCE:
                        self.failures.append(str(bra)+str(ket)+"="+str(me.value))
                else:
                    if abs(me.value) > self.TOLERANCE:
                        self.failures.append(str(bra)+str(ket)+"="+str(me.value))
        
    
    def test_specific_NonAntisymmetrizedME_WithBook(self):
        """
        V(r) = 1, mu_param=1, b_length=1, for central interactions is 
        orthogonality condition on the two states.
        """
        # (bra, ket_, J, force_name, correct_value)
        benchmarks = [
            (QN_2body_L_Coupling(QN_1body_radial(1,2,mt=1), QN_1body_radial(1,3,mt=1), 2),
             QN_2body_L_Coupling(QN_1body_radial(1,2,mt=1), QN_1body_radial(1,3,mt=1), 2),
             0, ForceEnum.Central, 1.0),
            (QN_2body_L_Coupling(QN_1body_radial(2,0,mt=1), QN_1body_radial(0,2,mt=1), 2),
             QN_2body_L_Coupling(QN_1body_radial(0,1,mt=1), QN_1body_radial(1,3,mt=1), 2),
             0, ForceEnum.Central, 0.0),
#             (QN_2body_LS_Coupling(QN_1body_radial(2,0), QN_1body_radial(0,2), 2, 1),
#              QN_2body_LS_Coupling(QN_1body_radial(0,1), QN_1body_radial(1,3), 2, 1),
#              3, ForceEnum.Tensor, 0.01079898),
#             (QN_2body_LS_Coupling(QN_1body_radial(2,2), QN_1body_radial(3,2), 2, 1),
#              QN_2body_LS_Coupling(QN_1body_radial(2,1), QN_1body_radial(3,3), 2, 1),
#              3, ForceEnum.Tensor, 0.01079898),
#             (QN_2body_LS_Coupling(QN_1body_radial(2,2), QN_1body_radial(2,2), 2, 1),
#              QN_2body_LS_Coupling(QN_1body_radial(2,2), QN_1body_radial(2,2), 2, 1),
#              2, ForceEnum.Spin_Orbit, 0.01079898)
        ]
        
        self._checkSpecificMEs(benchmarks)
    
    def testOrthonormalityOfMoshinskyTransformation(self):
    
        self.force = switchMatrixElementType(ForceEnum.Central)
        force_kwargs = {
            SHO_Parameters.b_length : 1.5,
            CentralMEParameters.mu_length : 0,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.constant  : 1.0,
            CentralMEParameters.n_power   : 0
            }
        self.force.setInteractionParameters(**force_kwargs)
        
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD'],
                     *valenceSpacesDict['PF']])
    
        for i in range(len(sts)):
            a = QN_1body_jj(*readAntoine(sts[i]))
            for j in range(i, len(sts)):
                b = QN_1body_jj(*readAntoine(sts[j]))
                for k in range(j, len(sts)):
                    c = QN_1body_jj(*readAntoine(sts[k]))
                    for l in range(k, len(sts)):
                        d = QN_1body_jj(*readAntoine(sts[l]))
    
                        self._testOrthogonality(a, b, c, d)
        
        if len(self.failures) > 0:
            print("FAILURES in Orthonormalization test:")
            for fail in self.failures:
                print(fail)
            self.assertEqual(len(self.failures), 0, 
                "there are [{}] non orthogonal m.e (TOL={})"
                .format(len(self.failures), self.TOLERANCE))
        
    
    def test_directComparisonGaussian(self):
        """ Compare with the Bruyeres le chatel m.e for a gaussian 
            V0 = -70 MeV    mu = 1.4    b = 1.5 fm
        """
        self._numberMEs = 1
        try:
            test = MatrixElementFilesComparator(
                'bench_me/TBME_JT_BLC_bench.sho',
                '../results/SPSDPF_Hamil_JT/1Gauss_SPSDPF_2.sho', 
                ignorelines=(5,4), verbose=False)
        except FileNotFoundError as e:
            print("ERROR :: Compute the file for the test or actualize the files to test.")
            raise(e)
        except ValueError as e:
            print("ERROR :: fix the 'ignorelines' argument for the files (lines"
                  " that are not the JT blocks, title, ...)")
            raise(e)
    
        test.compareDictionaries()
        print(" === TEST RESULTS:    =================================\n")
        prettyPrintDictionary(test.getResults())
        prettyPrintDictionary(test.getFailedME())
        test.plotFailedDifferences()
        prettyPrintDictionary(test.getMissingME())
        
        print("Missing in test file (but parity conserv)")
        miss = test.getMissingME()
        
        time.sleep(1)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()