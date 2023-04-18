"""
Created on Mar 10, 2021

@author: Miguel
"""
import numpy as np
from sympy import S

from helpers.Helpers import safe_racah, Constants, safe_wigner_6j,\
    safe_3j_symbols, almostEqual, safe_wigner_9j, safe_clebsch_gordan,\
    readAntoine

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    _TwoBodyMatrixElement_JCoupled, _TwoBodyMatrixElement_Antisym_JTCoupled, \
    MatrixElementException, _TwoBodyMatrixElement_Antisym_JCoupled
from matrix_elements.transformations import TalmiTransformation
from helpers.Enums import CouplingSchemeEnum, CentralMEParameters, AttributeArgs,\
    PotentialForms, SHO_Parameters, DensityDependentParameters,\
    MultipoleParameters
from helpers.Log import XLog
from helpers.integrals import _RadialDensityDependentFermi, _RadialIntegralsLS
from helpers.WaveFunctions import QN_1body_radial, QN_2body_jj_JT_Coupling,\
    QN_2body_jj_J_Coupling, QN_1body_jj
from numpy import dtype
from helpers.mathFunctionsHelper import _buildAngularYCoeffsArray,\
    _buildRadialFunctionsArray, angular_Y_KM_index, sphericalHarmonic,\
    _angular_Y_KM_memo_accessor, _radial_2Body_functions, _angular_Y_KM_me_memo
from helpers import SCIPY_INSTALLED
from copy import deepcopy

class CentralForce(TalmiTransformation):
    
    COUPLING = CouplingSchemeEnum.L
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a radial potential form V(r; mu_length, constant, n, ...)
        
        :b_length 
        :hbar_omega
        :potential       <str> in PotentialForms Enumeration
        :mu_length       <float>  fm
        :constant        <float>  MeV
        :n_power         <int>
        
        method bypasses calling from main or io_manager
        """
        
        if True in map(lambda a: isinstance(a, dict), kwargs.values()):
            # when calling from io_manager, arguments appear as dictionaries, 
            # parse them            
            _map = {
                CentralMEParameters.potential : (AttributeArgs.name, str),
                CentralMEParameters.constant  : (AttributeArgs.value, float),
                CentralMEParameters.mu_length : (AttributeArgs.value, float),
                CentralMEParameters.n_power   : (AttributeArgs.value, int)
            }
            #
            # for arg, value in kwargs.items():
            #     if arg in _map:
            #         attr_parser = _map[arg]
            #         attr_, parser_ = attr_parser
            #         kwargs[arg] = parser_(kwargs[arg].get(attr_))
            #     elif isinstance(value, str):
            #         kwargs[arg] = float(value) if '.' in value else int(value)
            kwargs = CentralForce._automaticParseInteractionParameters(_map, kwargs)
            
        super(CentralForce, cls).setInteractionParameters(*args, **kwargs)
        #cls.plotRadialPotential()
    
    def _validKet_relativeAngularMomentums(self):
        """ Central interaction only allows l'==l"""
        return (self._l, )
    
    def deltaConditionsForGlobalQN(self):
        """ 
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        NOTE: Redundant if run from JJ -> LS recoupling
        """
        if (self.L_bra != self.L_ket):
            return False
        return True
    
    def _deltaConditionsForCOM_Iteration(self):
        """ This condition ensure the antisymmetrization (without calling 
        exchanged the matrix element)"""
        if (((self.S_bra + self.T + self._l) % 2 == 1) and 
            ((self.S_ket + self.T + self._l_q) % 2 == 1)):
            return True
        return False
    
    def centerOfMassMatrixElementEvaluation(self):
        #TalmiTransformation.centerOfMassMatrixElementEvaluation(self)
        """ 
        Radial Brody-Moshinsky transformation, direct implementation for  
        central interaction.
        """
        if not self.deltaConditionsForGlobalQN():
            return 0
        
        return self._BrodyMoshinskyTransformation()
    
    
    def _globalInteractionCoefficient(self):
        # no special interaction constant for the Central ME
        return self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        return 1
    

    
    

class CentralForce_JTScheme(CentralForce, _TwoBodyMatrixElement_JTCoupled):
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    def __init__(self, bra, ket, run_it=True):
        _TwoBodyMatrixElement_JTCoupled.__init__(self, bra, ket, run_it=run_it)
    
    def _run(self):
        ## First method that runs antisymmetrization_ by exchange the quantum
        ## numbers (X2 time), change 2* _series_coefficient
        return _TwoBodyMatrixElement_JTCoupled._run(self)
    
    
    def _validKetTotalSpins(self):
        """ For Central Interaction, <S |Vc| S'> != 0 only if  S=S' """
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ For Central Interaction, <L |Vc| L'> != 0 only if  L=L' """
        return (self.L_bra, )
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        return self.centerOfMassMatrixElementEvaluation()
    

class CoulombForce(CentralForce, _TwoBodyMatrixElement_JCoupled):
    
    COUPLING = CouplingSchemeEnum.JJ
    
    _BREAK_ISOSPIN = True
    
    # COULOMB_CONST = 1.4522545041047  ## [MeV fm_ e^-2] K factor in natural units
    COULOMB_CONST = 1.44197028     ## constant extracted form HFBaxial code
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a radial potential form V(r; mu_length, constant, n, ...)
        
        :b_length 
        :hbar_omega
        :constant        <float>  MeV
        
        method bypasses calling from main or io_manager
        """
        
        for arg, value in kwargs.items():
            if isinstance(value, str):
                kwargs[arg] = float(value) if '.' in value else int(value)
                    
        kwargs[CentralMEParameters.potential] = PotentialForms.Coulomb
        kwargs[CentralMEParameters.constant]  = cls.COULOMB_CONST
        kwargs[CentralMEParameters.mu_length] = 1
        kwargs[CentralMEParameters.n_power]   = 0
        
        super(CentralForce, cls).setInteractionParameters(*args, **kwargs)
    
    def __init__(self, bra, ket, run_it=True):
        _TwoBodyMatrixElement_JCoupled.__init__(self, bra, ket, run_it=run_it)
            
    
    def _run(self):
    
        if self.isNullMatrixElement:
            return
        # if self.bra.isospin_3rdComponent != 1: 
        #     ## same number of p or n for bra and ket_ already verified.
        #     self._value = 0
        #     self._isNullMatrixElement = True
        #     return False
        else:
            _TwoBodyMatrixElement_JCoupled._run(self)
    
    def _nullConditionsOnParticleLabelStates(self):
        
        if self.bra.isospin_3rdComponent != 1: 
            ## same number of p or n for bra and ket_ already verified.
            self._value = 0
            self._isNullMatrixElement = True
            return False
        return True
    
    def _deltaConditionsForCOM_Iteration(self):
        """ This condition ensure the antisymmetrization (without calling 
        exchanged the matrix element)"""
        if (((self.S_bra + 1 + self._l) % 2 == 1) and 
            ((self.S_ket + 1 + self._l_q) % 2 == 1)):
            return True
        return False
    
    def _validKetTotalSpins(self):
        """ For Central Interaction, <S |Vc| S'> != 0 only if  S=S' """
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ For Central Interaction, <L |Vc| L'> != 0 only if  L=L' """
        return (self.L_bra, )
    
    def _LScoupled_MatrixElement(self):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        return self.centerOfMassMatrixElementEvaluation()
    

from helpers.Enums import DensityDependentParameters as dd_p
from helpers.Enums import AttributeArgs as atrE
# class DensityDependentForce_JTScheme(_TwoBodyMatrixElement_Antisym_JTCoupled):
class DensityDependentForce_JTScheme(_TwoBodyMatrixElement_JTCoupled):

    """
    Density term based on Fermi density distribution, (ordered filled up to A 
    mass number). 
    """
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    _BREAK_ISOSPIN = False
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a radial potential form V(r; mu_length, constant, n, ...)
        
        :b_length 
        :hbar_omega
        :potential       <str> in PotentialForms Enumeration
        :mu_length       <float>  fm
        :constant        <float>  MeV
        :n_power         <int>
        
        method bypasses calling from main or io_manager
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        _b = SHO_Parameters.b_length
        _a = SHO_Parameters.A_Mass
        _z = SHO_Parameters.Z
        
        cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
        cls.PARAMS_SHO[_a] = int(kwargs.get(_a))
        z = kwargs.get(_z)
        cls.PARAMS_SHO[_z] = int(z) if z else None
        
        cls.PARAMS_FORCE = {}
        cls.PARAMS_CORE  = {}
        cls._has_core_b_lenght = False
        fa = atrE.ForceArgs
        ## To avoid exception in DD sho aproximation
        if not dd_p.file in kwargs:
            kwargs[dd_p.file] = None
        
        for param in dd_p.members():
            aux = kwargs[param]
            if param == dd_p.core:
                b = aux.get(fa.DensDep.core_b_len, None)
                if (b != None) and (b != ''):
                    cls.PARAMS_CORE[fa.DensDep.core_b_len] = float(b)
                    cls._has_core_b_lenght = True
                _z = aux.get(fa.DensDep.protons, '0')
                _n = aux.get(fa.DensDep.neutrons,'0')
                if '.' in _z or '.' in _n: 
                    print("[Error] Set up d.d force core Z or N number is float["
                          ,_z,_n,"] use only integers")
                    if '.' in _z or '.' in _z: 
                        raise MatrixElementException("Invalid N/Z core. Stop")
                
                _z, _n = abs(int(_z)), abs(int(_n)) 
                # in case somebody set them <0
                if _z == 0 and _n == 0:
                    print("[Warning] Cannot set A=0 core, Z=N=0, so PARAMS_CORE",
                          "ignores arguments given, A,Z come form SHO_params:",
                          cls.PARAMS_SHO)
                    continue
                cls.PARAMS_CORE[fa.DensDep.protons ] = int(_z)
                cls.PARAMS_CORE[fa.DensDep.neutrons] = int(_n)  
            elif param == dd_p.file:
                if isinstance(aux, dict):
                    if AttributeArgs.name in aux:
                        aux = aux[AttributeArgs.name]
                cls.PARAMS_FORCE[param] = aux ## is string
            else:
                if isinstance(aux, dict):
                    aux = float(aux[AttributeArgs.value])
                cls.PARAMS_FORCE[param] = aux
                
        #cls.PARAMS_FORCE[CentralMEParameters.potential] = PotentialForms.Gaussian
    
    def _validKetTotalAngularMomentums(self):
        return (self.L_bra, )
    
    def _validKetTotalSpins(self):
        return (self.S_bra, )
    
    def _LScoupled_MatrixElement(self):
        
        phs = ((-1)**self.S_bra)
        fact = 1 - (phs * self.PARAMS_FORCE[DensityDependentParameters.x0])
        
        ## Antisymmetrization_ factor 
        fact *= (1 - ((-1)**(self.T + self.S_bra + 
                             self.L_bra + self.ket.l2 + self.ket.l1)))
        
        if self.isNullValue(fact):
            return 0.0
        fact *= ((2*self.bra.l1 + 1)*(2*self.bra.l2 + 1)
                 *(2*self.ket.l1 + 1)*(2*self.ket.l2 + 1))**0.5
        
        fact *= safe_3j_symbols(self.bra.l1, self.L_bra, self.bra.l2, 0, 0, 0)
        fact *= safe_3j_symbols(self.ket.l1, self.L_ket, self.ket.l2, 0, 0, 0)
        fact *= self.PARAMS_FORCE[DensityDependentParameters.constant]/ (4*np.pi)
        
        if self.isNullValue(fact):
            return 0.0
        
        _A = self.PARAMS_SHO.get(SHO_Parameters.A_Mass)
        _Z = self.PARAMS_SHO.get(SHO_Parameters.Z)
        fa = atrE.ForceArgs
        if ((fa.DensDep.protons in self.PARAMS_CORE)
             and (fa.DensDep.neutrons in self.PARAMS_CORE)):
            ## it only reset the DD core if there both parameters are defined
            _Z = self.PARAMS_CORE[fa.DensDep.protons]
            _A =  _Z + self.PARAMS_CORE[fa.DensDep.neutrons]
        
        args = (
            QN_1body_radial(self.bra.n1, self.bra.l1), 
            QN_1body_radial(self.bra.n2, self.bra.l2),
            QN_1body_radial(self.ket.n1, self.ket.l1),
            QN_1body_radial(self.ket.n2, self.ket.l2),
            self.PARAMS_SHO.get(SHO_Parameters.b_length), _A, _Z,
            self.PARAMS_FORCE.get(DensityDependentParameters.alpha),
            self.PARAMS_CORE .get(AttributeArgs.ForceArgs.DensDep.core_b_len)
        )
        if self.DEBUG_MODE:
            _RadialDensityDependentFermi.DEBUG_MODE = True
            
        _RadialDensityDependentFermi._DENSITY_APROX = False
            
        radial = _RadialDensityDependentFermi.integral(*args)
        
        return fact * radial


class DensityDependentForceFromFile_JScheme(_TwoBodyMatrixElement_Antisym_JCoupled,
                                            DensityDependentForce_JTScheme):
    
    """
    Force for the D1S integrated over a real Mean field density from Taurus:
        * final_wf.bin
        * get from txt (easy)
    Uses the same parameters as the D1S:
        TODO: Might be necessary the Levedeb_ integration or some sort of 
            importing for a very precise grid (i,e, Omega=20)
        # Laguerre_ is also valid x4 points
        
        
        # use J scheme to integrate directly over the j-m functions,
    
    """
    
    COUPLING        = CouplingSchemeEnum.JJ
    RECOUPLES_LS    = False
    _BREAK_ISOSPIN  = True
    
    _R_DIM = 25
    _A_DIM = 0
    _OMEGA = 15
    USING_LEBEDEV = False
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        super(DensityDependentForceFromFile_JScheme, cls).\
            setInteractionParameters(*args, **kwargs)
            
        # Get the filename
        cls._file2import    = None
        cls._density_matrix = None
        cls._kappa_matrix   = None
        cls._sh_states      = [] # <int> index in l_ge_10 format
        cls._sp_states      = [] # <QN_1body_jj> with m and mt
        cls._sp_dim         = None
        cls._orbital_max_sph_harmonic = 0
        
        cls._r   = []
        cls._ang = []
        cls._weight_r   = []
        cls._weight_ang = []
        cls._spatial_dens = None
        
        cls._radial_2b_wf_memo  = {}
        cls._sph_harmonic_memo  = {}
        
        ## Define the ANGULAR and RADIAL grid.
        cls.setIntegrationGrid(cls._R_DIM, cls._OMEGA)
        ## Set up the densities from Taurus_vap File 
        cls._importDensityMatrixWFAndSetUpBasis()
        ## Evaluate the spatial-density (integral points must be studied for the case)
        cls.evalutate_spatial_density()
        
        ## Reset the states to use in the new valence space.
        cls._sh_states      = [] # <int> index in l_ge_10 format
        cls._sp_states      = [] # <QN_1body_jj> with m and mt
        cls._sp_dim         = 0
        cls._orbital_max_sph_harmonic = 0
    
    @classmethod
    def setIntegrationGrid(cls, R_dim, OmegaOrd):
        """
        Reset the grid for integration for Laguerre-Associated (radial) 
        and Legendre (Angular).
            Implement Levedev Integral (At least one for a large Omega order)
        """
        if SCIPY_INSTALLED:
            from scipy.special import roots_genlaguerre, roots_legendre
        else:
            raise MatrixElementException("Scipy Required to use this MatrixElement")
        B_LEN  = cls.PARAMS_SHO[SHO_Parameters.b_length]
        ALPHA_ = cls.PARAMS_FORCE[dd_p.alpha]
        ## Radial Grid
        xR, wR, sum_ = roots_genlaguerre(R_dim, 0.5, True)
        cls._r = B_LEN * np.sqrt(xR / (2.0 + ALPHA_))
        cls._weight_r = wR
        
        ## Angular Grid: Choice between stored Lebedev_ grid mesh (commonly used).
        if cls.USING_LEBEDEV:
            ## Angular Grid from Lebedev_ Imported from /docs/LebedevPointsWeights
            if cls._OMEGA % 2 == 1:
                cls._OMEGA += 1
            if cls._OMEGA < 14: 
                print(" WARNING: Minimum Omega for Lebedev is 14, fixing to that.")
                cls._OMEGA = 14
            elif cls._OMEGA > 22:
                print(" WARNING: Maximum Omega for Lebedev is 22, fixing to that.")
                cls._OMEGA = 22
            ## read the file and convert the numbers
            with open(f'docs/LebedevPointsWeightsOmg{cls._OMEGA}.gut', 'r') as f:
                for line in f.readlines()[1:]:
                    line = line.strip().replace('D', 'E')
                    i, costh, phi, wA = line.split()
                    cls._ang.append( (float(costh), float(phi)) )
                    cls._weight_ang.append( float(wA) )
                    cls._A_DIM += 1
        else:
            ## Angular Grid (Theta_dim=Phi_dim = Ang_dim)
            xA, wA, sum_ = roots_legendre(OmegaOrd, True)
            costh = xA
            phi   = np.pi * xA
            for i in range(OmegaOrd):
                for j in range(OmegaOrd):
                    cls._ang.append( (costh[i], phi[j]) )
                    cls._weight_ang.append(wA[i] * wA[j])
            cls._A_DIM = cls._OMEGA**2       
    
    @classmethod
    def setBasis2BFunctions(cls, HO_b_length):
        """
        Once given the w.f. and having the b_lengths (core, m.e.)
        Set the angular and Radial functions and angular coefficents.
        """
                        
        radial2b = _buildRadialFunctionsArray(cls._sh_states, HO_b_length)
        cls._radial_2b_wf_memo = deepcopy(radial2b)
        
        # Build Base for Angular and Radial coefficients of the quantum numbers
        K_max = _buildAngularYCoeffsArray(cls._sh_states)
        cls._orbital_max_sph_harmonic = K_max
        
        # from scipy.special import sph_harm
        ## Angular basis (Common for both cases)       
        for K in range(cls._orbital_max_sph_harmonic +1):
            for M in range(-K, K +1):
                ind_k = angular_Y_KM_index(K, M, False)
                
                ang_func = [sphericalHarmonic(K, M, angle) for angle in cls._ang]
                ang_func = np.array(ang_func)
                cls._sph_harmonic_memo[ind_k] = deepcopy(ang_func)
    
    def _actualizeTheBasisFunctions(self):
        """ 
        Run each time the introduction of a new sh state, to be run at the
        constructor.
        """
        
        new_found = False      
        for st_ in (self.bra.sp_state_1.AntoineStrIndex_l_greatThan10, 
                    self.bra.sp_state_2.AntoineStrIndex_l_greatThan10, 
                    self.ket.sp_state_1.AntoineStrIndex_l_greatThan10, 
                    self.ket.sp_state_2.AntoineStrIndex_l_greatThan10):
            
            if st_ not in self._sh_states:
                self._sh_states.append(int(st_))
                n, l, j = readAntoine(st_, l_ge_10=True)
                self._sp_dim += 2*(j + 1) # counting p and n
                new_found = True
        
        if new_found:
            self._sp_states = []
            for mt in (-1, 1):
                for sh_st in self._sh_states:
                    n, l, j = readAntoine(sh_st, l_ge_10=True)
                    m_vals = [i for i in range(j, -j-1, -2)]
                    for m in m_vals:
                        self._sp_states.append(QN_1body_jj(n, l, j, m, mt))
            
            self.setBasis2BFunctions(self.PARAMS_SHO[SHO_Parameters.b_length])
        _=0
    
    @classmethod
    def _angularComponentsForDensity(cls, a,la,ja,mja,indx_a, b,lb,jb,mjb,indx_b):
        """ Find the K,M components of the two body angular wave function and
        add up to the spatial density 
        
        Note, in order to speed up the density matrix summation"""
        spO2 = cls._sp_dim // 2
        for K in range(max(0, abs(ja-jb)//2), (ja+jb)//2 +1):
            M = (mjb - mja) // 2
            if ((abs(M) > K) or ((K + la + lb) % 2 == 1)):
                continue
            
            indx_K  = angular_Y_KM_index( K,   M, False)
            key_abK = _angular_Y_KM_memo_accessor(indx_a,indx_b,indx_K)
            
            c_ang   = _angular_Y_KM_me_memo[key_abK]
            
            for ir in range(len(cls._r)):
                for ia in range(len(cls._ang)):
                    
                    angular = cls._sph_harmonic_memo[indx_K][ia] * c_ang
                    val = cls._aux_radial[ir] * angular
                    
                    val_p = val * cls._density_matrix[b, a]
                    val_n = val * cls._density_matrix[b + spO2, a + spO2]
                    
                    val = val_p + val_n
                    cls._spatial_dens_imag[ir, ia] += np.imag(val)
                    cls._spatial_dens[ir, ia]      += np.real(val)
        
    @classmethod
    def evalutate_spatial_density(cls):
        """
        Evaluate the radial and angular functions for the Mean field W.F.
        """
        global _angular_Y_KM_me_memo
        import time
        t_ = time.time()
        print(f" [ ] Evaluating Spatial Density ...")
        
        cls._spatial_dens = np.zeros( (cls._R_DIM, cls._A_DIM) )
        cls._spatial_dens_imag = np.zeros( (cls._R_DIM, cls._A_DIM) ) ## test
        tot_ = (cls._sp_dim // 2) * (cls._sp_dim // 2 + 1) // 2
        for a  in range(cls._sp_dim // 2):
            # print("  progr. a% =", tot_*(1 - a) - (a*(a-1)//2), "/", tot_)
            print("  progr a:", a, "/", cls._sp_dim // 2)
            sp_st_a = cls._sp_states[a]
            na, la, ja, mja = sp_st_a.n, sp_st_a.l, sp_st_a.j, sp_st_a.m
            indx_a = angular_Y_KM_index(ja, mja, True)
            
            for b in range(a, cls._sp_dim // 2):
                sp_st_b = cls._sp_states[b]
                
                if sp_st_a.m_t != sp_st_b.m_t :
                    continue
                nb, lb, jb, mjb = sp_st_b.n, sp_st_b.l, sp_st_b.j, sp_st_b.m
                indx_b = angular_Y_KM_index(jb, mjb, True)
                                
                rad_key = (na,la, nb,lb)
                cls._aux_radial  = cls._radial_2b_wf_memo[rad_key](cls._r)
                
                cls._angularComponentsForDensity(a,la,ja,mja,indx_a, 
                                                 b,lb,jb,mjb,indx_b)
                if a != b:
                    cls._angularComponentsForDensity(b,lb,jb,mjb,indx_b,
                                                     a,la,ja,mja,indx_a)
                
                ## TODO: optimize to do protons and neutrons at te same time
        integ_A = 0.0 
        ALPHA_  = cls.PARAMS_FORCE[dd_p.alpha]
        for ir in range(len(cls._r)):
            radial  = np.exp(( (2.0+ALPHA_) * (cls._r[ir] / cls._b_density)**2))
            for ia in range(len(cls._ang)):
                val = cls._spatial_dens[ir, ia] 
                val *= cls._weight_ang[ia] * cls._weight_r[ir]
                integ_A +=  radial * val
        integ_A *= (cls._b_density**3) / (2.0*((2.0 + ALPHA_)**1.5))
        if cls.USING_LEBEDEV:
            integ_A *= 4 * np.pi
        _= 0
        
        # import matplotlib.pyplot as plt
        #
        # fig_ = plt.figure()
        # for ia in range(0, len(cls._ang), cls._OMEGA):
        #     lab = f'ang={cls._ang[ia][0]:4.2f}, {cls._ang[ia][1]:4.2f}'
        #     plt.plot(cls._r, cls._spatial_dens[:, ia], label=lab)
        #
        # plt.legend()
        # plt.show()
        #
        # fig_ = plt.figure
        # off_diag = deepcopy(cls._density_matrix)
        # for i in range(cls._sp_dim):
        #     if off_diag[i,i] > 0.1:
        #         off_diag[i,i] = 0
        # plt.imshow(off_diag)
        # plt.show()
        # TODO:: Test the density
        t_ = (time.time() - t_)
        print(f" [DONE] Spatial Density has been imported and evaluated. ({t_:5.3f}s) A={integ_A:9.5f}")
                
        
    
    @classmethod
    def _setUpDensityMatrices(cls, dim_, u0_mat, v0_mat):
        """ Auxiliary to set the density matrices shared when read
            (once processed from text or binary file)"""
        dim_ = int (dim_**0.5)
        
        U, V = np.zeros((dim_, dim_)), np.zeros((dim_, dim_))
        cls._density_matrix = np.zeros((dim_, dim_))
        cls._kappa_matrix   = np.zeros((dim_, dim_)) 
        for i in range(dim_):
            for j in range(dim_):
                k = i*dim_ + j
                U[j, i] = u0_mat[k]
                V[j, i] = v0_mat[k]
        
        cls._sp_dim = dim_
        cls._density_matrix = np.matmul(V.conjugate(), np.transpose(V)) 
        cls._kappa_matrix   = np.matmul(V.conjugate(), np.transpose(U))
        
        ## Set the sp_states in the file order
        for mt in (-1, 1):
            for sh_st in cls._sh_states:
                n, l, j = readAntoine(sh_st, l_ge_10=True)
                m_vals = [i for i in range(j, -j-1, -2)]
                for m in m_vals:
                    cls._sp_states.append(QN_1body_jj(n, l, j, m, mt))
        
    @classmethod
    def _readBinary(cls):
        """
        Equivalent process to import the file as the txt.
        
        * Tested against the same WF, exported both txt/bin, for several basis
        """
        sh_dim = 0
        density_matrix_lines = []
        
        x = np.fromfile(cls._file2import, count=2, dtype='uint32')
        sh_dim = int(x[-1])
        x = np.fromfile(cls._file2import, count=sh_dim +4, dtype='uint32')
        cls._sh_states = list(x[4:])
        
        y = np.fromfile(cls._file2import, dtype=np.float64)
        y = y[sh_dim+2:] # counting the lines=[sh_dim, lines sh, label long]
        
        dim_   = (y.size - 1) // 2 #! there is an auxiliary line between U and V
        u0_mat = y[:dim_]
        v0_mat = y[dim_+1:]        #! there is an auxiliary line between U and V
        
        ## NOTE: verified with the txt file for the same function
        cls._setUpDensityMatrices(dim_, u0_mat, v0_mat)
    
    @classmethod
    def _readTXT(cls):
        """ 
        Get the file from U, V generated by Taurus. 
            # number of shell states
            1
            2 ... shell states
            hash label state (ignored)
            U[j,i] values
            V[j,i] values
        
        Required conversion to the density matrix    rho = V^* V^T
        """
        sh_dim = 0
        density_matrix_lines = []
        with open(cls._file2import, 'r') as f:
            data = f.readlines()
            for il, line in enumerate(data):
                line = line.strip()
                if il == 0:
                    sh_dim = int(line)
                elif il <= sh_dim:
                    cls._sh_states.append(int(line))
                elif il == sh_dim + 1:
                    continue
                else:
                    # if 'E' in line:
                    #     line = line.replace('E', 'e')
                    line = float(line)
                    density_matrix_lines.append(line)
        
        dim_ = len(density_matrix_lines) // 2
        u0_mat = density_matrix_lines[:dim_]
        v0_mat = density_matrix_lines[dim_:]
        
        ## NOTE: verified with the DIMENS_indexes_and_rhoLRkappas.gut file
        cls._setUpDensityMatrices(dim_, u0_mat, v0_mat)
            
    @classmethod
    def _importDensityMatrixWFAndSetUpBasis(cls):
        """ 
        Read the file for the Us and Vs and convert to density matrix.
        Shell states are also in the file, set up the radial/angular basis for
        the imported density. (radial has to be reset after)
        """
        cls._file2import = cls.PARAMS_FORCE.get(dd_p.file, None)
        if cls._file2import.endswith('.bin'):
            ## import binary file
            cls._readBinary()
        elif cls._file2import.endswith('.txt'):
            ## import txt file
            cls._readTXT()
        else:
            raise MatrixElementException(f"Invalid File to import [{cls._file2import}]")
        
        b_length = cls.PARAMS_SHO.get(SHO_Parameters.b_length, 1.0)
        ## Use the b_length for the calculation as default.
        if (dd_p.core in cls.PARAMS_FORCE):
            ## In case of b_core explicitly given, use it for the w.f.
            _b_core = cls.PARAMS_FORCE[dd_p.core].get(atrE.ForceArgs.DensDep.core_b_len)
            if ( (_b_core != None) and (_b_core.replace('.', '').isnumeric()) ):
                b_length = float(_b_core)
        cls._b_density = b_length
        
        ## Do the wave function setting for the density
        cls.setBasis2BFunctions(b_length)

    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_2body_jj_J_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_jj_J_Coupling>")
        if not isinstance(ket, QN_2body_jj_J_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_jj_J_Coupling>")
    
    def __init__(self, bra, ket, run_it=True):
        
        self.__checkInputArguments(bra, ket)
        
        self.bra = bra
        self.ket = ket
        
        self.J = bra.J
        
        self.exchange_phase = None
        self.exch_2bme = None
        
        if (bra.sp_state_1.m_t + bra.sp_state_2.m_t != 
            ket.sp_state_1.m_t + ket.sp_state_2.m_t):
            self._value = 0.0
        else:
            self._nullConditionForSameOrbit()
        
        if not self.isNullMatrixElement and run_it: # always run it
            self._run()
              
    def _run(self):
        
        if self.isNullMatrixElement:
            return
        
        self._actualizeTheBasisFunctions()
        
        if self.DEBUG_MODE: 
            XLog.write('nas_me', ket=self.ket.shellStatesNotation)
        
        # antisymmetrization_ taken in the inner evaluation            
        self._calculateJMatrixElement()
        
        if self.DEBUG_MODE:
            XLog.write('nas_me', value=self._value, norms=self.bra.norm()*self.ket.norm())
        
        self._value *= self.bra.norm() * self.ket.norm()
    
            
    def _calculateJMatrixElement(self):
        """
        Process has the antisymmetrization in the uncoupled matrix element.
        Process does not require the LS coupling
        """
        ma_valid = [m for m in range(-self.bra.j1, self.bra.j1+1, 2)]
        mb_valid = [m for m in range(-self.bra.j2, self.bra.j2+1, 2)]
        
        mc_valid = [m for m in range(-self.ket.j1, self.ket.j1+1, 2)]
        md_valid = [m for m in range(-self.ket.j2, self.ket.j2+1, 2)]
        
        me_value = 0.0
        for ma in ma_valid:
            for mb in mb_valid:
                if self.bra.M != (ma+mb)//2: continue
                
                args_b = (S(self.bra.j1)/2, S(self.bra.j2)/2, S(self.bra.J),
                          S(ma)/2, S(mb)/2, S(self.bra.M))
                ccg_b = safe_clebsch_gordan(*args_b)
                if self.isNullValue(ccg_b): continue
                
                for mc in mc_valid:
                    for md in md_valid:
                        
                        if self.ket.M != (mc+md)//2: continue
                        args_k = (S(self.ket.j1)/2, S(self.ket.j2)/2, S(self.ket.J),
                                  S(mc)/2, S(md)/2, S(self.ket.M))
                        
                        ang_recoup = ccg_b * safe_clebsch_gordan(*args_k)
                        
                        if self.isNullValue(ang_recoup): continue
                        
                        dd_integral = self._radialAngularIntegral(ma,mb,mc,md)
                        me_value += (ang_recoup * dd_integral *  
                                     self.PARAMS_FORCE[dd_p.constant])
        
        self._value = me_value
    
    def _isospinExchangeFactor(self):
        """ Matrix element factor associated with the isospin """
        ta,tb = self.bra.sp_state_1.m_t, self.bra.sp_state_2.m_t
        tc,td = self.ket.sp_state_1.m_t, self.ket.sp_state_2.m_t
        X_ac_bd  = ((ta==tc)*(tb==td))
        X_ac_bd -= self.PARAMS_FORCE[dd_p.x0]*((ta==td)*(tb==tc))
        X_ad_bc  = ((ta==td)*(tb==tc)) 
        X_ad_bc -= self.PARAMS_FORCE[dd_p.x0]*((ta==tc)*(tb==td))
        
        self.X_ac_bd = X_ac_bd
        self.X_ad_bc = X_ad_bc            
                 
    def _radialAngularIntegral(self, mja, mjb, mjc, mjd):
        
        self._isospinExchangeFactor()
        if abs(self.X_ac_bd - self.X_ad_bc) < 1.0e-6: 
            return 0.0
        
        ALPHA_ = self.PARAMS_FORCE[dd_p.alpha]
        B_LEN_ = self.PARAMS_SHO[SHO_Parameters.b_length]
        # for ir in range(len(cls._r)):
        na, la, ja = self.bra.n1, self.bra.l1, self.bra.j1
        nb, lb, jb = self.bra.n2, self.bra.l2, self.bra.j2
        nc, lc, jc = self.ket.n1, self.ket.l1, self.ket.j1
        nd, ld, jd = self.ket.n2, self.ket.l2, self.ket.j2
        
        indx_a = angular_Y_KM_index(ja, mja, True)        
        indx_b = angular_Y_KM_index(jb, mjb, True)
        indx_c = angular_Y_KM_index(jc, mjc, True)        
        indx_d = angular_Y_KM_index(jd, mjd, True)
        
        nl_ab = (na,la, nb,lb)
        nl_cd = (nc,lc, nd,ld)    
        
        radial  = self._radial_2b_wf_memo[nl_ab](self._r)
        radial *= self._radial_2b_wf_memo[nl_cd](self._r)
        radial *= self._weight_r * np.exp((2.+ALPHA_)* np.power(self._r/B_LEN_, 2))
        
        aux_d   = np.zeros(self._A_DIM)
        aux_e   = np.zeros(self._A_DIM)
        ## DIR
        if not self.isNullValue(self.X_ac_bd):
            for K1 in range(max(0, abs(ja-jc)//2), (ja+jc)//2 +1):
                M = (mjc - mja) // 2
                if ((abs(M) > K1) or ((K1 + la + lc) % 2 == 1)): continue
                
                indx_K1  = angular_Y_KM_index(K1, M, False)
                key_acK = _angular_Y_KM_memo_accessor(indx_a,indx_c,indx_K1)
                
                for K2 in range(max(0, abs(jb-jd)//2), (jb+jd)//2 +1):
                    M2 = (mjd - mjb) // 2
                    if ((abs(M2) > K2) or ((K2 + lb + ld) % 2 == 1)): continue
                
                    indx_K2 = angular_Y_KM_index(K2, M2, False)
                    key_bdK = _angular_Y_KM_memo_accessor(indx_b,indx_d,indx_K2)
                
                    c_ang    = _angular_Y_KM_me_memo [key_acK]
                    c_ang   *= _angular_Y_KM_me_memo [key_bdK]
                    sph_har  = self._sph_harmonic_memo[indx_K1]
                    sph_har *= self._sph_harmonic_memo[indx_K2]
                    aux_d    = aux_d + c_ang
            
        ## EXCH
        if not self.isNullValue(self.X_ad_bc):
            for K1 in range(max(0, abs(ja-jd)//2), (ja+jd)//2 +1):
                M = (mjd - mja) // 2
                if ((abs(M) > K1) or ((K1 + la + ld) % 2 == 1)): continue
                
                indx_K1  = angular_Y_KM_index(K1, M, False)
                key_adK = _angular_Y_KM_memo_accessor(indx_a,indx_d,indx_K1)
                
                for K2 in range(max(0, abs(jb-jc)//2), (jb+jc)//2 +1):
                    M2 = (mjc - mjb) // 2
                    if ((abs(M2) > K2) or ((K2 + lb + lc) % 2 == 1)): continue
                
                    indx_K2 = angular_Y_KM_index(K2, M2, False)
                    key_bcK = _angular_Y_KM_memo_accessor(indx_b,indx_c,indx_K2)
                
                    c_ang    = _angular_Y_KM_me_memo [key_adK]
                    c_ang   *= _angular_Y_KM_me_memo [key_bcK]
                    sph_har  = self._sph_harmonic_memo[indx_K1]
                    sph_har *= self._sph_harmonic_memo[indx_K2]
                    aux_e    = aux_e + c_ang * sph_har
        
        angular = (self.X_ac_bd * aux_d) - (self.X_ad_bc * aux_e)
        angular = angular * self._weight_ang
        
        me_val = 0.0
        for ir in range(len(self._r)):
            for ia in range(len(self._ang)):
                
                val = radial[ir] * angular [ia]
                me_val += val * (self._spatial_dens[ir,ia]**ALPHA_)
        
        me_val *= 0.5 * (B_LEN_ **3)
        me_val /= (2.0 + ALPHA_)**1.5
        if self.USING_LEBEDEV:
            me_val *= 4 * np.pi
        if abs(np.imag(me_val)) > 1.0e-10:
            raise MatrixElementException(f"DD term is complex <{self.bra} |v| {self.ket}>= {me_val}") 
        return np.real(me_val)
        
    ## return void LS valid L S for SpeedRunner to work with this m.e
    def _validKetTotalSpins(self):
        return tuple()
    
    def _validKetTotalAngularMomentums(self):
        return tuple()
    
class KineticTwoBody_JTScheme(_TwoBodyMatrixElement_Antisym_JTCoupled): #
    # _TwoBodyMatrixElement_JTCoupled): #
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    _BREAK_ISOSPIN = False
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a radial potential form V(r; mu_length, constant, n, ...)
        
        :b_length 
        :hbar_omega
        __ THIS INTERACTION CANNOT BE FIXED __
        
        method bypasses calling from main or io_manager
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        _b = SHO_Parameters.b_length
        _A = SHO_Parameters.A_Mass
        assert _A in kwargs and _b in kwargs, "A_mass and oscillator b length are mandatory"
        
        cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
        cls.PARAMS_SHO[_A] = int(kwargs.get(_A))
    
    
    def _validKetTotalAngularMomentums(self):
        return (self.L_bra, )
    
    def _validKetTotalSpins(self):
        return (self.S_bra, )
    
    def _nablaReducedMatrixElement(self, particle):
        part = str(particle)
        
        n_q , l_q = getattr(self.bra, "n"+part), getattr(self.bra, "l"+part)
        n , l     = getattr(self.ket, "n"+part), getattr(self.ket, "l"+part)
        
        A_, B_ = 0, 0
        if l_q == (l + 1):
            A_ = ((n**0.5) *(n_q==(n-1))) + (((n + l + 1.5)**0.5) *(n_q==n))
        if l_q == (l - 1):
            B_ = (((n + 1)**0.5) *(n_q==(n+1))) + (((n + l + 0.5)**0.5) *(n_q==n))
        
        return (-A_ * ((l + 1)**0.5)) - (B_ * (l**0.5))
    
    def _LScoupled_MatrixElement(self):
        
        if not hasattr(self, "_kin_factor"):
            ## Real constant, but  COM taurus multiplies by 2/(A*b^2) 
            # self._kin_factor = (Constants.HBAR_C**2) / (
            #      2 * Constants.M_MEAN
            #      * (self.PARAMS_SHO[SHO_Parameters.b_length]**2)
            #      * self.PARAMS_SHO[SHO_Parameters.A_Mass])
            
            self._kin_factor = 0.5 * (Constants.HBAR_C**2) / Constants.M_MEAN #= 20.74
            # self._kin_factor = 1
            # self._kin_factor = 1 / (self.PARAMS_SHO[SHO_Parameters.A_Mass] 
            #         * (self.PARAMS_SHO[SHO_Parameters.b_length]**2)
            #         * 2 * Constants.M_MEAN)
        
        fact = safe_wigner_6j(self.bra.l1, self.bra.l2, self.L_bra, 
                              self.ket.l2, self.ket.l1, 1)
        fact *= ((-1)**(self.ket.l1 + self.bra.l2 + self.L_bra))
            
        nabla_1 = self._nablaReducedMatrixElement(1)
        nabla_2 = self._nablaReducedMatrixElement(2)
        if self.DEBUG_MODE:
            XLog.write("Lme", nabla1=nabla_1, nabla2=nabla_2, f=fact, 
                       kin_f= self._kin_factor)
            
        return fact * self._kin_factor * nabla_1 * nabla_2



class _Multipole_JTScheme(_TwoBodyMatrixElement_Antisym_JTCoupled):
    """
    Analytical SDI matrix element don't require LS recoup_ nor explicit antisymm_
    override the __init__ method to directly evaluate it (run_it= ignored)
    """
    RECOUPLES_LS = False
    SEPARABLE_MULTIPOLE = False # set true if the radial integral is exchange dependent
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a general potential form delta(r; mu_length, constant)
        
        :b_length 
        :hbar_omega
        :constants <dict> 
            available constants A, B, C and D:
            V(r1-r2)*(A + B*s(1)s(2) + C*t(1)t(2) + D*s(1)s(2)*t(1)t(2))        
        """
        
        for param in AttributeArgs.ForceArgs.SDI.members():
            val = kwargs[MultipoleParameters.constants].get(param)
            if val != None:
                val = float(val)
            kwargs[param] = val
        del kwargs[MultipoleParameters.constants]
        
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        params_and_defaults = {
            SHO_Parameters.b_length     : 1,
            SHO_Parameters.hbar_omega   : 1,
            AttributeArgs.ForceArgs.Multipole.A    : 1,
            AttributeArgs.ForceArgs.Multipole.B    : 0,
            AttributeArgs.ForceArgs.Multipole.C    : 0,
            AttributeArgs.ForceArgs.Multipole.D    : 0
        }
        
        for param, default in params_and_defaults.items():
            value = kwargs.get(param)
            value = default if value == None else value
            
            if param in SHO_Parameters.members():
                cls.PARAMS_SHO[param] = value
            else:
                cls.PARAMS_FORCE[param] = value
                
        B = cls.PARAMS_FORCE[AttributeArgs.ForceArgs.Multipole.B]
        D = cls.PARAMS_FORCE[AttributeArgs.ForceArgs.Multipole.D]
        
        cls._evaluateSpinParts = False
        ## spin dependent parts are highly consuming,
        if not (almostEqual(B, 0, 1.e-9) and almostEqual(D, 0, 1.e-9)):
            cls._evaluateSpinParts = True
    
    
    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_2body_jj_JT_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_jj_JT_Coupling>")
        if not isinstance(ket, QN_2body_jj_JT_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_jj_JT_Coupling>")
    
    def __init__(self, bra, ket, run_it=True):
        
        self.__checkInputArguments(bra, ket)
        
        self.bra = bra
        self.ket = ket
        
        self.J = bra.J
        self.T = bra.T
        
        self.exchange_phase = None
        self.exch_2bme = None
        
        if (bra.J != ket.J) or (bra.T != ket.T):
            print("Bra JT [{},{}]doesn't match with ket's JT [{},{}]"
                  .format(bra.J, bra.T, ket.J, ket.T))
            self._value = 0.0
        else:
            self._nullConditionForSameOrbit()
        
        if not self.isNullMatrixElement and run_it: # always run it
            self._run() 
    
    def _run(self):
        
        if self.isNullMatrixElement:
            return
        
        phase, exchanged_ket = self.ket.exchange()
        self.exchange_phase = phase
        self.exch_2bme = self.__class__(self.bra, exchanged_ket, run_it=False)
        
        if self.DEBUG_MODE: 
            XLog.write('nas_me', ket=self.ket.shellStatesNotation)
        
        _L_min = max(abs(self.bra.l1-self.bra.l2), abs(self.ket.l1-self.ket.l2))
        _L_max = min(    self.bra.l1+self.bra.l2 ,     self.ket.l1+self.ket.l2 )
        
        self._value = 0.0
        for L in range(_L_min, _L_max+1, 1):
            ang_cent_d, ang_cent_e = 0, 0
            ang_spin_d, ang_spin_e = 0, 0
            
            ang_cent_d = self._AngularCoeff_Central(L)
            ang_cent_e = self.exch_2bme._AngularCoeff_Central(L)
            
            if self.DEBUG_MODE:
                XLog.write('nas_me', lambda_=L, value=self._value, 
                           norms=self.bra.norm()*self.ket.norm())
            
            if self._evaluateSpinParts:
                ## V_MSDI = V_SDI + B * <tau(1) * tau(2)> + C (only off-diagonal)
                ang_spin_d = self._AngularCoeff_Spin(L)
                ang_spin_e = self.exch_2bme._AngularCoeff_Spin(L)
                
            if almostEqual(abs(ang_spin_d)+abs(ang_spin_e)+abs(ang_cent_d)+
                               abs(ang_cent_e), 0, self.NULL_TOLERANCE):
                continue
            rad_d = self._RadialCoeff(L)
            rad_e = rad_d
            if not self.SEPARABLE_MULTIPOLE:
                rad_e = self.exch_2bme._RadialCoeff(L) 
            
            self._value += ((ang_cent_d + ang_spin_d)*rad_d) - \
                           ((ang_cent_e + ang_spin_e)*rad_e*self.exchange_phase)
                
        self._value *= self.bra.norm() * self.ket.norm()
    
    def _RadialCoeff(self, L):
        raise MatrixElementException("Abstract method, implement multipole Radial function")
        
    def _AngularCoeff_Central(self, lambda_):
        
        j_a, j_b = self.bra.j1, self.bra.j2
        j_c, j_d = self.ket.j1, self.ket.j2
        
        isos_f  = self.PARAMS_FORCE[AttributeArgs.ForceArgs.Multipole.A]
        isp     = self.T - (3*(1 - self.T))
        isos_f += self.PARAMS_FORCE[AttributeArgs.ForceArgs.Multipole.C] * isp
        phs = (-1)**((j_b + j_c)//2 + self.J)
        
        if ( ((self.bra.l1 + self.ket.l1 + lambda_) % 2 == 1) or  
             ((self.bra.l2 + self.ket.l2 + lambda_) % 2 == 1)) :
            return 0 # parity condition form _redAngCoeff
        
        val = safe_wigner_6j(j_a / 2, j_b / 2, self.J, 
                             j_d / 2, j_c / 2, lambda_)
        if not almostEqual(val, 0, self.NULL_TOLERANCE): 
            val *= (  safe_3j_symbols(j_a / 2, j_c / 2, lambda_, .5, -.5, 0)
                    * safe_3j_symbols(j_b / 2, j_d / 2, lambda_, .5, -.5, 0))
        
        factor  = ((j_a + 1) * (j_b + 1) * (j_c + 1) * (j_d + 1))**0.5 
        factor /= 4 * np.pi * ((-1)**((j_c + j_d)//2  - 1))
        
        return phs * factor * isos_f * val
    
    def _AngularCoeff_Spin(self, lambda_):
        
        j_a, j_b = self.bra.j1, self.bra.j2
        j_c, j_d = self.ket.j1, self.ket.j2
        l_a, l_b = self.bra.l1, self.bra.l2
        l_c, l_d = self.ket.l1, self.ket.l2
        
        isos_f  = self.PARAMS_FORCE[AttributeArgs.ForceArgs.Multipole.B]
        isp     = self.T - (3*(1 - self.T))
        isos_f += self.PARAMS_FORCE[AttributeArgs.ForceArgs.Multipole.D] * isp
        phs = (-1)**((j_b + j_c)//2 + self.J + lambda_ + 1)

        if (((l_a + l_c + lambda_)%2 == 1) or ((l_b + l_d + lambda_)%2 == 1)):
            return 0 # parity condition form _redAngCoeff
        
        total = 0
        for j in range(abs(lambda_ - 1), lambda_ + 1 +1):
            
            val  = safe_wigner_6j(j_a / 2, j_b / 2, self.J, j_d, j_c, j)
            if not almostEqual(val, 0, self.NULL_TOLERANCE): 
                val *= (  
                    safe_3j_symbols(l_a, lambda_, l_c, 0, 0, 0) * 
                    safe_3j_symbols(l_b, lambda_, l_d, 0, 0, 0) *
                    safe_wigner_9j(l_a, .5, j_a / 2, 
                                   l_c, .5, j_c / 2, 
                                   lambda_, 1, j) *
                    safe_wigner_9j(l_b, .5, j_b / 2, 
                                   l_d, .5, j_d / 2, 
                                   lambda_, 1, j)
                    )
            
            total += ((-1)**j) *val * ((2*j) + 1)
        
        factor  = ((j_a + 1) * (j_b + 1) * (j_c + 1) * (j_d + 1))**0.5 
        factor *= ((2*l_a + 1)*(2*l_b + 1)*(2*l_c + 1)*(2*l_d + 1))**0.5
        factor *= ((2*lambda_) + 1) / (4 * np.pi)
        
        return phs * factor * isos_f * val
    
    
    ## return void LS valid L S for SpeedRunner to work with this m.e
    def _validKetTotalSpins(self):
        return ()
        #raise MatrixElementException("You shall not pass here for this m.e!")
    
    def _validKetTotalAngularMomentums(self):
        return ()
        # raise MatrixElementException("You shall not pass here for this m.e!")



class MultipoleDelta_JTScheme(_Multipole_JTScheme):
    
    SEPARABLE_MULTIPOLE = True
    
    def _RadialCoeff(self, lambda_):
        """Implementation of the delta integral (multipole independent)"""
        
        b = self.PARAMS_SHO.get(SHO_Parameters.b_length)
        
        qnr_a = QN_1body_radial(self.bra.n1, self.bra.l1) # conjugated
        qnr_b = QN_1body_radial(self.bra.n2, self.bra.l2) # conjugated
        qnr_c = QN_1body_radial(self.ket.n1, self.ket.l1)
        qnr_d = QN_1body_radial(self.ket.n2, self.ket.l2)
        
        rad = _RadialIntegralsLS.integral(2, qnr_a, qnr_b, qnr_c, qnr_d, b)
        
        return rad * (b**3)
        
