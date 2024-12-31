"""
Created on Mar 10, 2021

@author: Miguel
"""
import numpy as np
from sympy import S

from helpers.Helpers import Constants, safe_wigner_6j

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    _TwoBodyMatrixElement_JCoupled, _TwoBodyMatrixElement_Antisym_JTCoupled, \
    _OneBodyMatrixElement_jjscheme, _standardSetUpForCentralWithExchangeOps
from matrix_elements.transformations import TalmiTransformation,\
    TalmiGeneralizedTransformation
from helpers.Enums import CouplingSchemeEnum, CentralMEParameters, AttributeArgs,\
    PotentialForms, SHO_Parameters, BrinkBoekerParameters, CentralGeneralizedMEParameters
from helpers.Log import XLog

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
            for attr in (CentralMEParameters.opt_cutoff,
                         CentralMEParameters.opt_mu_2,
                         CentralMEParameters.opt_mu_3):
                if attr in kwargs:
                    _map[attr] = (AttributeArgs.value, float)
            
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
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Tensor interaction calculation. 
        
        Modification to import Exchange operators in the Brink-Boeker form.
        """
        cls = _standardSetUpForCentralWithExchangeOps(cls, **kwargs) 
        
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = []
    
    def _validKetTotalSpins(self):
        """ For Central Interaction, <S |Vc| S'> != 0 only if  S=S' """
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ For Central Interaction, <L |Vc| L'> != 0 only if  L=L' """
        return (self.L_bra, )
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        
        (1-st without exchange)    return self.centerOfMassMatrixElementEvaluation()
        """
        # Radial Part for Gaussian Integral
        radial_energy = self.centerOfMassMatrixElementEvaluation()
        
        if self.DEBUG_MODE:
            XLog.write('BB', mu=self.PARAMS_FORCE[CentralMEParameters.mu_length])
        
        # Exchange Part
        # W + P(S)* B - P(T)* H - P(T)*P(S)* M
        _S_aux = (-1)**(self.S_bra + 1)
        _T_aux = (-1)**(self.T)
        _L_aux = (-1)**(self.T + self.S_bra + 1)
        
        exchange_energy = (
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Wigner),
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Bartlett)   * _S_aux,
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Heisenberg) * _T_aux,
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Majorana)   * _L_aux
        )
        
        # Add up
        prod_part = radial_energy * sum(exchange_energy)
        
        if self.DEBUG_MODE:
            XLog.write('BB', radial=radial_energy, exch=exchange_energy, 
                       exch_sum=sum(exchange_energy), val=prod_part)
        
        return prod_part
    

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
        exchanged the matrix element)
        
        ## Note, this is the antisymetry for lS instead of lST, 
            since phase lST = (-)^[(S-1) + (T+1) + l]   
            that +1 is not from T=1 of the pp
        """
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
    

class ElectromagneticCentral_JScheme(CentralForce, _TwoBodyMatrixElement_JCoupled):
    pass

class _Kinetic_1BME(_OneBodyMatrixElement_jjscheme):
    
    """ Matrix Element for the kinetic one-body operator. """
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        _b  = SHO_Parameters.b_length
        _ho = SHO_Parameters.hbar_omega
        _A  = SHO_Parameters.A_Mass
        assert _A in kwargs and _b in kwargs, "A_mass and oscillator b length are mandatory"
        
        b_len = float(kwargs.get(_b))
        cls.PARAMS_SHO[_b] = b_len
        cls.PARAMS_SHO[_A] = int(kwargs.get(_A))
        hbaromega = (Constants.HBAR_C**2) / (Constants.M_MEAN * (b_len**2))
        cls.PARAMS_SHO[_ho] = hbaromega
        
    def _run(self):
        
        if self.isNullMatrixElement: return
        if self.DEBUG_MODE: 
            XLog.write('nas_me', ket=self.ket.shellStatesNotation)
        
        self._value = 0.0
        
        n_a, n_b = self.bra.n, self.ket.n
        l_a, l_b = self.bra.l, self.ket.l
        j_a, j_b = self.bra.j, self.ket.j
        
        if ((l_a != l_b) or (j_a != j_b)) : return 
        
        val = 0.0
        if (n_a == n_b):
            val = 2*n_a + l_a + 1.5
        elif (n_a == n_b - 1):
            val = (n_b * (2*n_b + l_b + 0.5))**.5
        elif (n_a == n_b + 1):
            val = (n_a * (2*n_a + l_a + 0.5))**.5
        
        self._value  = 0.5 * val * self.PARAMS_SHO[SHO_Parameters.hbar_omega]
    

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

class Quadratic_OrbitalMomentum_JTScheme(CentralForce_JTScheme):
    
    """
    Interaction for   V(r) * (Exchange_terms) * (l)^2
    
        This interaction uses the decomposition in relative j angular momentum
    
     Inhereted methods ::
        def _validKetTotalSpins(self):                 S=S'
        def _validKet_relativeAngularMomentums(self):  l=l_q
        def deltaConditionsForGlobalQN(self):          L_bra == Lket
        def _deltaConditionsForCOM_Iteration(self):    (l + S + T, even)
    
    """
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    _BREAK_ISOSPIN = False
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Tensor interaction calculation. 
        
        Modification to import Exchange operators in the Brink-Boeker form.
        """
        
        cls = _standardSetUpForCentralWithExchangeOps(cls, **kwargs) 
        
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = []
    
    def _validKetTotalAngularMomentums(self):
        """ There is delta in the lambda_ = lambda_' from the COM m.e. 6j  """
        return (self.L_bra, )
        
    def centerOfMassMatrixElementEvaluation(self):
        #TalmiTransformation.centerOfMassMatrixElementEvaluation(self)
        """ 
        Radial Brody-Moshinsky transformation, direct implementation for  
        non-central spin orbit force.
        """       
        if not self.deltaConditionsForGlobalQN(): return 0
        
        aux = self._BrodyMoshinskyTransformation()
        return  aux
    
    def _globalInteractionCoefficient(self):
        """ _globalInteractionCoefficient() * sum_p { I(p) * series_(p) } """
        
        factor = np.sqrt((2*self.L_bra + 1))
    
        return  factor * self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        ## l_q = l        
        return self._l * (self._l + 1)
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """    
        # Radial Part for Gaussian Integral
        radial_energy = self.centerOfMassMatrixElementEvaluation()
        
        # Exchange Part
        # W + P(S)* B - P(T)* H - P(T)*P(S)* M
        _S_aux = (-1)**(self.S_bra + 1)
        _T_aux = (-1)**(self.T)
        _L_aux = (-1)**(self.T + self.S_bra + 1)
        
        exchange_energy = (
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Wigner),
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Bartlett)   * _S_aux,
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Heisenberg) * _T_aux,
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Majorana)   * _L_aux
        )
        # Add up
        prod_part = radial_energy * sum(exchange_energy)
        
        if self.DEBUG_MODE:
            XLog.write('BB', radial=radial_energy, exch=exchange_energy, 
                       exch_sum=sum(exchange_energy), val=prod_part)
        
        return prod_part



class CentralGeneralizedForce(TalmiGeneralizedTransformation):
    """
    Interaction with central, also exchange operators for both R and r.
    Extension from Central V(r) -> V(R) * U(R)
    
        * Same central parameters for usual integrations 
        * Extended parameters (optional mu lengths, cutoffs, WSaxon...) only for r
        * Both might contribute redundantly (i.e. constant, constant_R, ...)
        
    """
    
    COUPLING = CouplingSchemeEnum.L
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a radial potential form V(r; mu_length, constant, n, ...)
        
        :b_length 
        :hbar_omega
            Arguments followed by '_r' (relative) and '_R' (total)
        :potential       <str> in PotentialForms Enumeration
        :mu_length       <float>  fm
        :constant        <float>  MeV
        :n_power         <int>
        
        method bypasses calling from main or io_manager
        """
        GENMEparm = CentralGeneralizedMEParameters
        if True in map(lambda a: isinstance(a, dict), kwargs.values()):
            # when calling from io_manager, arguments appear as dictionaries, 
            # parse them            
            _map = {
                GENMEparm.potential   : (AttributeArgs.name, str),
                GENMEparm.constant    : (AttributeArgs.value,float),
                GENMEparm.mu_length   : (AttributeArgs.value,float),
                GENMEparm.n_power     : (AttributeArgs.value,int),
                GENMEparm.potential_R : (AttributeArgs.name, str),
                GENMEparm.constant_R  : (AttributeArgs.value,float),
                GENMEparm.mu_length_R : (AttributeArgs.value,float),
                GENMEparm.n_power_R   : (AttributeArgs.value,int)
            }
            for attr in (CentralMEParameters.opt_cutoff,
                         CentralMEParameters.opt_mu_2,
                         CentralMEParameters.opt_mu_3):
                if attr in kwargs:
                    _map[attr] = (AttributeArgs.value, float)
            
            kwargs = CentralGeneralizedForce._automaticParseInteractionParameters(_map, kwargs)
        
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = []
        cls._integrals_q_max = -1
        cls._talmiIntegrals_R= []
        #cls.plotRadialPotential()
    
    def _validKet_relativeAngularMomentums(self):
        """ Central interaction only allows l'==l"""
        return (self._l, )
    
    def _validKet_totalAngularMomentums(self):
        """ Central interaction only allows l'==l"""
        return (self._L, )
    
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
        valid_lST = (((self.S_bra + self.T + self._l) % 2 == 1) and 
                     ((self.S_ket + self.T + self._l_q) % 2 == 1))
        valid_LST = False
        if valid_lST or valid_LST:
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
        return self.PARAMS_FORCE.get(CentralGeneralizedMEParameters.constant) * \
               self.PARAMS_FORCE.get(CentralGeneralizedMEParameters.constant_R)
    
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        return 1
    


#===============================================================================
#
# GENERALIZED Talmi Interactions - V(R, r)
#
#===============================================================================


class CentralGeneralizedForce_JTScheme(CentralGeneralizedForce, 
                                       _TwoBodyMatrixElement_JTCoupled):
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    def __init__(self, bra, ket, run_it=True):
        _TwoBodyMatrixElement_JTCoupled.__init__(self, bra, ket, run_it=run_it)
    
    def _run(self):
        ## First method that runs antisymmetrization_ by exchange the quantum
        ## numbers (X2 time), change 2* _series_coefficient
        return _TwoBodyMatrixElement_JTCoupled._run(self)
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Tensor interaction calculation. 
    
        Modification to import Exchange operators in the Brink-Boeker form.
        """
        cls = _standardSetUpForCentralWithExchangeOps(cls, generalizedCentral=True,
                                                      **kwargs)
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = []
        cls._integrals_q_max = -1
        cls._talmiIntegrals_R= []
    
    def _validKetTotalSpins(self):
        """ For Central Interaction, <S |Vc| S'> != 0 only if  S=S' """
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ For Central Interaction, <L |Vc| L'> != 0 only if  L=L' """
        return (self.L_bra, )
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        
        (1-st without exchange)    return self.centerOfMassMatrixElementEvaluation()
        """
        if self.DEBUG_MODE:
            XLog.write('BB', mu=self.PARAMS_FORCE[CentralGeneralizedMEParameters.mu_length])
            
        # Radial Part for Gaussian Integral
        radial_energy = self.centerOfMassMatrixElementEvaluation()
        
        # Exchange Part
        # W + P(S)* B - P(T)* H - P(T)*P(S)* M
        _S_aux = (-1)**(self.S_bra + 1)
        _T_aux = (-1)**(self.T)
        _L_aux = (-1)**(self.T + self.S_bra + 1)
        
        exchange_energy = (
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Wigner),
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Bartlett)   * _S_aux,
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Heisenberg) * _T_aux,
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Majorana)   * _L_aux
        )
        
        # Add up
        prod_part = radial_energy * sum(exchange_energy)
        
        if self.DEBUG_MODE:
            XLog.write('BB', radial=radial_energy, exch=exchange_energy, 
                       exch_sum=sum(exchange_energy), val=prod_part)
        
        return prod_part
