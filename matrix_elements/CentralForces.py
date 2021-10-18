'''
Created on Mar 10, 2021

@author: Miguel
'''
import numpy as np

from helpers.Helpers import safe_racah, Constants, safe_wigner_6j,\
    safe_3j_symbols

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    _TwoBodyMatrixElement_JCoupled, _TwoBodyMatrixElement_Antisym_JTCoupled
from matrix_elements.transformations import TalmiTransformation
from helpers.Enums import CouplingSchemeEnum, CentralMEParameters, AttributeArgs,\
    PotentialForms, SHO_Parameters, DensityDependentParameters
from helpers.Log import XLog
from helpers.integrals import _RadialDensityDependentFermi
from helpers.WaveFunctions import QN_1body_radial

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
            
    
    # def _run(self):
    #
    #     if self.isNullMatrixElement:
    #         return
    #     if self.bra.isospin_3rdComponent != 1: 
    #         ## same number of p or n for bra and ket_ already verified.
    #         self._value = 0
    #         self._isNullMatrixElement = True
    #         return False
    #     else:
    #         _TwoBodyMatrixElement_JCoupled._run(self)
    
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
        from helpers.Enums import DensityDependentParameters as dd_p
        
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
        
        for param in dd_p.members():
            aux = kwargs[param]
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
        args = (
            QN_1body_radial(self.bra.n1, self.bra.l1), 
            QN_1body_radial(self.bra.n2, self.bra.l2),
            QN_1body_radial(self.ket.n1, self.ket.l1),
            QN_1body_radial(self.ket.n2, self.ket.l2),
            self.PARAMS_SHO.get(SHO_Parameters.b_length),
            self.PARAMS_SHO.get(SHO_Parameters.A_Mass),
            self.PARAMS_SHO.get(SHO_Parameters.Z),
            self.PARAMS_FORCE.get(DensityDependentParameters.alpha)
        )
        if self.DEBUG_MODE:
            _RadialDensityDependentFermi.DEBUG_MODE = True
            
        _RadialDensityDependentFermi._DENSITY_APROX = False
        radial = _RadialDensityDependentFermi.integral(*args)
        
        return fact * radial



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
        