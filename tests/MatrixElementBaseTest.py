'''
Created on Jul 26, 2021

@author: Miguel
'''
import unittest
from matrix_elements import switchMatrixElementType
from helpers.Enums import ForceEnum, SHO_Parameters, CentralMEParameters,\
    PotentialForms, BrinkBoekerParameters, AttributeArgs,\
    DensityDependentParameters
from copy import deepcopy
from helpers.Helpers import prettyPrintDictionary, valenceSpacesDict,\
    readAntoine
from itertools import combinations_with_replacement
from helpers.WaveFunctions import QN_2body_jj_JT_Coupling, QN_1body_jj
from helpers.matrixElementHandlers import MatrixElementFilesComparator

class MatrixElementBaseTest(unittest.TestCase):
    """
    This class perform basic checks on the matrix elements, phase exchanges and 
    test against bench matrix elements (verified by other methods) stored in 
    bench_me
    
    References:
        [L.M. Robledo]
    L. M. Robledo, HFBaxial code, (2002)
    J. L. Egido, L. M. Robledo, and R. R. Chasman, Phys. Lett. B 393, 13 (1997)
    L. M. Robledo and G. F. Bertsch, Phys. Rev. C 84, 014312 (2011).
    
    """
    def _exchangeAndExpectedPhase(self, bra, ket):
        J, T = bra.J, bra.T
        b_phs, b_exch = bra.exchange()
        k_phs, k_exch = ket.exchange()
        
        keys = [None]*7
        keys[0] = (ket, bra)                   # no   phase exch
        keys[1], keys[2] = (b_exch, ket), (ket, b_exch)     # bra  phase exch
        keys[3], keys[4] = (k_exch, bra), (bra, k_exch)     # ket_ phase exch
        keys[5], keys[6] = (k_exch, b_exch), (b_exch, k_exch) # both phase exch
        
        final = []
        for i in range(7):
            if i < 1:
                final.append((keys[i], 1)) 
            elif i < 3:
                p = ((b_exch.j1 + b_exch.j2) // 2) + J + T
                final.append((keys[i], (-1)**p))
            elif i < 5:
                p = ((k_exch.j1 + k_exch.j2) // 2) + J + T
                final.append((keys[i], (-1)**p))
            else:
                p = (b_exch.j1 + b_exch.j2 + k_exch.j1 + k_exch.j2) // 2
                final.append((keys[i], (-1)**p))
                    
        return final
    
    def _evaluateParityConservationOfMatrixElements(self, spss_tuple):
        
        spss = combinations_with_replacement(spss_tuple, 4)
        for sps in spss:
            bra, ket = sps[:2], sps[2:]
            s1 = QN_1body_jj(*readAntoine(bra[0]))
            s2 = QN_1body_jj(*readAntoine(bra[1]))
            s3 = QN_1body_jj(*readAntoine(ket[0]))
            s4 = QN_1body_jj(*readAntoine(ket[1]))
            
            # Lm = max(abs(s1.l - s2.l), abs(s3.l - s4.l))
            # LM = min(abs(s1.l + s2.l), abs(s3.l + s4.l))
            Jm = max(abs(s1.j - s2.j), abs(s3.j - s4.j)) // 2
            JM = min(s1.j + s2.j, s3.j + s4.j) // 2
            
            parity_break = (s1.l + s2.l) + (s3.l + s4.l) % 2 == 0
            
            for J in range(Jm, JM +1):
                for T in (0, 1):
                    
                    try:
                        bra = QN_2body_jj_JT_Coupling(s1, s2, J, T)
                        ket = QN_2body_jj_JT_Coupling(s3, s4, J, T)
                    except AssertionError:
                        print(s1, s2, s3, s4, J,"->",Jm, JM)
                        continue
                    self.force.setInteractionParameters(**self.force_kwargs)
                    self.me = self.force(bra, ket)
                    if parity_break and not (self.me.isnull()):
                        self.assertFalse(False, 
                            "Matrix element [{}] breaks parity but is not null [{}]"
                            .format(str(bra)+str(ket), self.me.value))
        
    
    def _evaluateSingleParticleStates(self, spss_tuple):
        
        spss = combinations_with_replacement(spss_tuple, 4)
        for sps in spss:
            bra, ket = sps[:2], sps[2:]
            s1 = QN_1body_jj(*readAntoine(bra[0]))
            s2 = QN_1body_jj(*readAntoine(bra[1]))
            s3 = QN_1body_jj(*readAntoine(ket[0]))
            s4 = QN_1body_jj(*readAntoine(ket[1]))
            
            # Lm = max(abs(s1.l - s2.l), abs(s3.l - s4.l))
            # LM = min(abs(s1.l + s2.l), abs(s3.l + s4.l))
            Jm = max(abs(s1.j - s2.j), abs(s3.j - s4.j)) // 2
            JM = min(s1.j + s2.j, s3.j + s4.j) // 2
            
            for J in range(Jm, JM +1):
                for T in (0, 1):
                    
                    try:
                        bra = QN_2body_jj_JT_Coupling(s1, s2, J, T)
                        ket = QN_2body_jj_JT_Coupling(s3, s4, J, T)
                    except AssertionError:
                        print(s1, s2, s3, s4, J,"->",Jm, JM)
                        continue
                    self.force.setInteractionParameters(**self.force_kwargs)
                    self.me = self.force(bra, ket)
                    
                    for sts, theor_phs in self._exchangeAndExpectedPhase(bra, ket):
                        
                        val = self.force(sts[0], sts[1]).value
                        val_teor = theor_phs * self.me.value
                        
                        self.assertAlmostEqual(val, val_teor, delta=0.0000001,
                            msg="phase invalid / or value don't match \n{}[{}] != {}[{}]"
                            .format(str(bra)+str(ket), val,
                                    str(sts[0])+str(sts[1]), val_teor))
    
    #===========================================================================
    # TESTS:
    #===========================================================================
    
    def test_paritySPSD_BrinkBoeker_diagonal(self):
        
        self.force = switchMatrixElementType(ForceEnum.Brink_Boeker)
        att = AttributeArgs.ForceArgs.Brink_Boeker
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            BrinkBoekerParameters.mu_length : {att.part_1: 0.6, att.part_2: 1.2},
            BrinkBoekerParameters.Wigner    : {att.part_1: 100, att.part_2: -50},
            BrinkBoekerParameters.Majorana  : {att.part_1: -50, att.part_2: -50},
            BrinkBoekerParameters.Heisenberg: {att.part_1: -50, att.part_2: -50},
            BrinkBoekerParameters.Bartlett  : {att.part_1: 100, att.part_2: -50}
            }
    
        for shell in ('S', 'P', 'SD'):
            sts = tuple([*valenceSpacesDict[shell]])
            self._evaluateParityConservationOfMatrixElements(sts)
    
    def test_exchangeSPSD_BrinkBoeker_diagonal(self):
    
        self.force = switchMatrixElementType(ForceEnum.Brink_Boeker)
        att = AttributeArgs.ForceArgs.Brink_Boeker
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            BrinkBoekerParameters.mu_length : {att.part_1: 0.6, att.part_2: 1.2},
            BrinkBoekerParameters.Wigner    : {att.part_1: 100, att.part_2: -50},
            BrinkBoekerParameters.Majorana  : {att.part_1: -50, att.part_2: -50},
            BrinkBoekerParameters.Heisenberg: {att.part_1: -50, att.part_2: -50},
            BrinkBoekerParameters.Bartlett  : {att.part_1: 100, att.part_2: -50}
            }
    
        for shell in ('S', 'P', 'SD'):
            spss = tuple([*valenceSpacesDict[shell]])
            self._evaluateSingleParticleStates(spss)
    
#------------------------------------------------------------------------------ 
    def test_paritySPSD_BrinkBoeker_off_diag(self):
        
        att = AttributeArgs.ForceArgs.Brink_Boeker
        self.force = switchMatrixElementType(ForceEnum.Brink_Boeker)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            BrinkBoekerParameters.mu_length : {att.part_1: 0.6, att.part_2: 1.2},
            BrinkBoekerParameters.Wigner    : {att.part_1: 100, att.part_2: -50},
            BrinkBoekerParameters.Majorana  : {att.part_1: -50, att.part_2: -50},
            BrinkBoekerParameters.Heisenberg: {att.part_1: -50, att.part_2: -50},
            BrinkBoekerParameters.Bartlett  : {att.part_1: 100, att.part_2: -50}
            }
    
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
        self._evaluateParityConservationOfMatrixElements(sts)
    
    def test_exchangeSPSD_BrinkBoeker_off_diag(self):
    
        att = AttributeArgs.ForceArgs.Brink_Boeker
        self.force = switchMatrixElementType(ForceEnum.Brink_Boeker)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            BrinkBoekerParameters.mu_length : {att.part_1: 0.6, att.part_2: 1.2},
            BrinkBoekerParameters.Wigner    : {att.part_1: 100, att.part_2: -50},
            BrinkBoekerParameters.Majorana  : {att.part_1: -50, att.part_2: -50},
            BrinkBoekerParameters.Heisenberg: {att.part_1: -50, att.part_2: -50},
            BrinkBoekerParameters.Bartlett  : {att.part_1: 100, att.part_2: -50}
            }
    
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
        self._evaluateSingleParticleStates(sts)
        
#------------------------------------------------------------------------------ 
    def test_paritySPSD_Central_diagonal(self):
        self.force = switchMatrixElementType(ForceEnum.Central)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.mu_length : 1,
            CentralMEParameters.potential : PotentialForms.Gaussian
            }
    
        for shell in ('S', 'P', 'SD'):
            sts = tuple([*valenceSpacesDict[shell]])
            self._evaluateParityConservationOfMatrixElements(sts)
    
    def test_exchangeSPSD_Central_diagonal(self):
    
        self.force = switchMatrixElementType(ForceEnum.Central)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.mu_length : 1,
            CentralMEParameters.potential : PotentialForms.Gaussian
            }
    
        for shell in ('S', 'P', 'SD'):
            spss = tuple([*valenceSpacesDict[shell]])
            self._evaluateSingleParticleStates(spss)
    
#------------------------------------------------------------------------------ 
    def test_paritySPSD_Central_off_diag(self):
        self.force = switchMatrixElementType(ForceEnum.Central)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.mu_length : 1,
            CentralMEParameters.potential : PotentialForms.Gaussian
            }
    
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
    
        self._evaluateParityConservationOfMatrixElements(sts)
    
    def test_exchangeSPSD_Central_off_diag(self):
    
        self.force = switchMatrixElementType(ForceEnum.Central)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.mu_length : 1,
            CentralMEParameters.potential : PotentialForms.Gaussian
            }
    
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
    
        self._evaluateSingleParticleStates(sts)
        
#------------------------------------------------------------------------------ 
    def test_paritySPSD_ShortRangeLS_diagonal(self):
        self.force = switchMatrixElementType(ForceEnum.SpinOrbitShortRange)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.mu_length  : 1
            }
    
        for shell in ('S', 'P', 'SD'):
            sts = tuple([*valenceSpacesDict[shell]])
            self._evaluateParityConservationOfMatrixElements(sts)
    
    def test_exchangeSPSD_ShortRangeLS_diagonal(self):
    
        self.force = switchMatrixElementType(ForceEnum.SpinOrbitShortRange)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.mu_length  : 1
            }
    
        for shell in ('S', 'P', 'SD'):
            spss = tuple([*valenceSpacesDict[shell]])
            self._evaluateSingleParticleStates(spss)
    
    def test_paritySPSD_ShortRangeLS_off_diag(self):
        self.force = switchMatrixElementType(ForceEnum.SpinOrbitShortRange)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.mu_length  : 1
            }
    
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
        self._evaluateParityConservationOfMatrixElements(sts)
    
    def test_exchangeSPSD_ShortRangeLS_off_diag(self):
    
        self.force = switchMatrixElementType(ForceEnum.SpinOrbitShortRange)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.mu_length  : 1
            }
    
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
    
        self._evaluateSingleParticleStates(sts)            
    
    def test_paritySPSD_SpinOrbit(self):
        self.force = switchMatrixElementType(ForceEnum.SpinOrbit)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.mu_length : 1,
            }
        
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
        
        self._evaluateParityConservationOfMatrixElements(sts)
    
    def test_exchangeSPSD_SpinOrbit(self):
        
        self.force = switchMatrixElementType(ForceEnum.SpinOrbit)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.mu_length : 1,
            }
        
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
        
        self._evaluateSingleParticleStates(sts)
#------------------------------------------------------------------------------ 
    def test_paritySPSD_DensityDependant(self):
        
        self.force = switchMatrixElementType(ForceEnum.Density_Dependent)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            SHO_Parameters.A_Mass   : 8,
            DensityDependentParameters.constant  : 1,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.mu_length : 1,
            DensityDependentParameters.alpha    : 1/3,
            DensityDependentParameters.x0 : 1,
            }
    
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
    
        self._evaluateParityConservationOfMatrixElements(sts)
    
    def test_exchangeSPSD_DensityDependant(self):
    
        self.force = switchMatrixElementType(ForceEnum.Density_Dependent)
        self.force_kwargs = {
            SHO_Parameters.b_length : 1,
            SHO_Parameters.A_Mass   : 8,
            DensityDependentParameters.constant  : 1,
            CentralMEParameters.potential : PotentialForms.Power,
            CentralMEParameters.mu_length : 1,
            DensityDependentParameters.alpha    : 1/3,
            DensityDependentParameters.x0 : 1,
            }
    
        sts = tuple([*valenceSpacesDict['S'], 
                     *valenceSpacesDict['P'], 
                     *valenceSpacesDict['SD']])
    
        self._evaluateSingleParticleStates(sts)
    
    #===========================================================================
    # DIRECT COMPARATION
    #===========================================================================
    
    def _compareResults(self, f_bench, f_test):
        # b_bench_diag, b_bench_off = _getJTSchemeMatrixElements(
        test = MatrixElementFilesComparator(f_bench, f_test, ignorelines=(4,4))
        # b_2test_diag, b_2test_off = _getJTSchemeMatrixElements(
        #     file_test, ignorelines=4)
        test.compareDictionaries()
        
        result = test.getResults()
        
        fails = result.get('FAIL')
        if fails > 0:
            print("There are matrix elements wrong ::\n"
                  "==============\n* SUMMARY \n==============")
            prettyPrintDictionary(result)
            print("==============\n* Failed Matrix Elements\n==============")
            prettyPrintDictionary(test.getFailedME())
            
        self.assertEqual(fails , 0, 
                         "Wrong matrix elements [{}/{}] for the interaction"
                         .format(fails, result.get('TOTAL')))
            
        self.assertGreater(result.get('TOTAL'), 0, 
                         "Test has not been run, [0] Matrix elements compared")
            
        
    def test_1Gaussian_SPSDPF_compareWithBenchFile(self):
        """ 
        Matrix Elements tested in comparison with HFB_axial [L.M. Robledo]
        energy results.
        """
        self._compareResults('bench_me/central_SPSDPF_bench.sho', 
                             'me2test/central_SPSDPF.sho')
    
    def test_BrinkBoeker_SPSDPF_compareWithBenchFile(self):
        """ 
        Parametrization_ for the original article (b = 1.4989 fm)
            mu         =    0.7     1.4   fm
            Wigner     =  595.55  -72.21  MeV
            Majorana   = -206.05  -68.39  MeV
        
        Matrix Elements tested in comparison with HFB_axial [L.M. Robledo].
        energy results.
        """
        self._compareResults('bench_me/BrinkBoeker_SPSDPF_bench.sho', 
                             'me2test/bb_SPSDPF.sho')
    
    def test_SpinOrbit_ShortRange_SPSDPF_compareWithBenchFile(self):
        """
        Spin orbit, short range approximation, tested with WLS = 1.0
        Results are compared with the original matrix elements and tested with
        HFB_axial [L.M. Robledo].
        """
        self._compareResults('bench_me/onlyLS_JT_SPSD.2b', 
                             'me2test/lsShort_SPSD.sho')
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()