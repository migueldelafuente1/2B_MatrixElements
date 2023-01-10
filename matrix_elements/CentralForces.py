"""
Created on Mar 10, 2021

@author: Miguel
"""
import numpy as np

from helpers.Helpers import safe_racah, Constants, safe_wigner_6j,\
    safe_3j_symbols, almostEqual, safe_wigner_9j

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    _TwoBodyMatrixElement_JCoupled, _TwoBodyMatrixElement_Antisym_JTCoupled, \
    MatrixElementException
from matrix_elements.transformations import TalmiTransformation
from helpers.Enums import CouplingSchemeEnum, CentralMEParameters, AttributeArgs,\
    PotentialForms, SHO_Parameters, DensityDependentParameters,\
    MultipoleParameters
from helpers.Log import XLog
from helpers.integrals import _RadialDensityDependentFermi, _RadialIntegralsLS
from helpers.WaveFunctions import QN_1body_radial, QN_2body_jj_JT_Coupling

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
        raise MatrixElementException("You shall not pass here for this m.e!")
    
    def _validKetTotalAngularMomentums(self):
        raise MatrixElementException("You shall not pass here for this m.e!")



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
        
