'''
Created on Mar 8, 2021

@author: Miguel
'''
import numpy as np
from copy import deepcopy
from inspect import getclasstree

from helpers.Enums import CouplingSchemeEnum, CentralMEParameters,\
    BrinkBoekerParameters, PotentialForms
from helpers.Enums import AttributeArgs, SHO_Parameters
from helpers.Helpers import safe_racah, safe_clebsch_gordan, safe_3j_symbols,\
    safe_wigner_6j

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    MatrixElementException, _standardSetUpForCentralWithExchangeOps
from matrix_elements.transformations import TalmiTransformation
from helpers.WaveFunctions import QN_2body_LS_Coupling, QN_1body_radial
from helpers.integrals import _SpinOrbitPartialIntegral, _RadialIntegralsLS
from helpers.Log import XLog


class SpinOrbitForce(TalmiTransformation): # _TwoBodyMatrixElement_JTCoupled, 
    
    COUPLING = (CouplingSchemeEnum.L, CouplingSchemeEnum.S)
    
    def __init__(self, bra, ket, J, run_it=True):
        self.__checkInputArguments(bra, ket, J)
        
        # TODO: Might accept an LS coupled wave functions (when got that class)
        self.J = J
        
        self.S_bra = bra.S
        self.S_ket = ket.S
        
        TalmiTransformation.__init__(self, bra, ket, run_it=run_it)
        #raise Exception("Implement with jj coupled w.f.")
    
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
            
            kwargs = SpinOrbitForce._automaticParseInteractionParameters(_map, kwargs)
        
        super(SpinOrbitForce, cls).setInteractionParameters(*args, **kwargs)
    
    def __checkInputArguments(self, bra, ket, J):
        if not isinstance(J, int):
            raise MatrixElementException("J is not <int>")
        if not isinstance(bra, QN_2body_LS_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_LS_Coupling>")
        if not isinstance(ket, QN_2body_LS_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_LS_Coupling>")
    
    def _validKet_relativeAngularMomentums(self):
        """  Spin orbit interaction only allows l'==l"""
        return (self._l, )
    
    def deltaConditionsForGlobalQN(self):
        """ 
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        NOTE: Redundant if run from JJ -> LS recoupling
        """
        if (abs(self.L_bra - self.L_ket) > 1) :
            return False
        return True
    
    def _deltaConditionsForCOM_Iteration(self):
        """ For the antisymmetrization_ of the wave functions. """
        if (((self.S_bra + self.T + self._l) % 2 == 1) and 
            ((self.S_ket + self.T + self._l_q) % 2 == 1)):
                return True
        return False
    
    def _totalSpinTensorMatrixElement(self):
        """ <1/2 1/2 (S) | S^[1]| 1/2 1/2 (S)>, only non zero for S=S'=1 
        boolean for skip"""
        if (self.S_bra != self.S_ket) or (self.S_bra == 0):
            return True, 0.0
        
        return False, 2.449489742783178 ## = np.sqrt(6)
    
    
    def centerOfMassMatrixElementEvaluation(self):
        #TalmiTransformation.centerOfMassMatrixElementEvaluation(self)
        """ 
        Radial Brody-Moshinsky transformation, direct implementation for  
        non-central spin orbit force.
        """
        # the spin matrix element is 0 unless S=S'=1
        skip, spin_me = self._totalSpinTensorMatrixElement()
        if skip:
            return 0
    
        factor = safe_wigner_6j(self.L_bra, self.S_bra, self.J,
                                self.S_ket, self.L_ket,      1)
        if self.isNullValue(factor) or not self.deltaConditionsForGlobalQN():
            return 0
    
        phase   = (-1)**(self.rho_bra + self.J)
        factor *= np.sqrt((2*self.L_bra + 1)*(2*self.L_ket + 1))
    
        aux = factor * spin_me * phase * self._BrodyMoshinskyTransformation()
        return  aux
    
    def _globalInteractionCoefficient(self):
        # no special interaction constant for the Spin-Orbit
        return self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    def _interactionConstantsForCOM_Iteration(self):
        """ Common Matrix element in the (nlNL,n'l') Moshinsky series"""
        factor = safe_wigner_6j(self._l,    self.L_bra, self._L, 
                                self.L_ket, self._l_q,        1)
        if self.isNullValue(factor):
            return 0
        
        factor *= np.sqrt(self._l * (self._l + 1) * (2*self._l + 1))
        return factor
        


class SpinOrbitForce_JTScheme(_TwoBodyMatrixElement_JTCoupled, SpinOrbitForce):
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    def __init__(self, bra, ket, run_it=True):
        
        _TwoBodyMatrixElement_JTCoupled.__init__(self, bra, ket, run_it=run_it)
    
    
    def _deltaConditionsForCOM_Iteration(self):
        """ Antisymmetrization condition (*2 in BrodyMoshinkytransformation)"""
        #return True
        if (((self.S_bra + self.T + self._l) % 2 == 1) and 
            ((self.S_ket + self.T + self._l_q) % 2 == 1)):
                return True
        return False
    
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin, for tensor force impose 
        S = S' = 1, return nothing to skip the bracket spin S=0
        """
        if self.S_bra == 0:
            return []
        return (1, )
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        
        OJO: Moshinski, lambda' = lambda, lambda +- 1!!! as condition
        in the C_LS
        """
        _L_min = max(0, self.L_bra - 1)
        _L_max =        self.L_bra + 1
        gen_ = (l_q for l_q in range(_L_min, _L_max +1))
        return tuple(gen_)
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        return self.centerOfMassMatrixElementEvaluation()



class SpinOrbitFiniteRange_JTScheme(SpinOrbitForce_JTScheme):
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Tensor interaction calculation. 
        
        Modification to import Exchange operators in the Brink-Boeker form.
        """
        cls = _standardSetUpForCentralWithExchangeOps(cls, **kwargs)
        
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = []
    
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

class _Quadratic_SpinOrbit_jrelativeEval_JTScheme(SpinOrbitFiniteRange_JTScheme):
    
    """
    Interaction for   V(r) * (Exchange_terms) * (l*S)^2
    
        This interaction uses the decomposition in relative j angular momentum
    
     Inhereted methods ::
        def _validKet_relativeAngularMomentums(self):  l=l_q
        def deltaConditionsForGlobalQN(self):          L_bra - Lket <= 1
        def _deltaConditionsForCOM_Iteration(self):    (l + S + T, even)
        def _totalSpinTensorMatrixElement(self):       (NOT USED)
        def _validKetTotalSpins(self):                 S_bra = S_ket = 1
        
    ## NOTE: the definition of the relative- j decomposition values imply a null
             <nlSj|ls|n'l'Sj> = 0 for S=0, also l'=l from l^2 operator.
    """
    _angular_me_ls2 = {}
    
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
        phase   = (-1)**(self.L_bra + self.L_ket)
    
        factor = np.sqrt((2*self.L_bra + 1)*(2*self.L_ket + 1))
    
        return phase * factor * self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
        
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        ## l_q = l
        tpl = "_".join([str(x) for x in (self.J, self.L_bra, self.L_ket,
                                         self._l, self._L)])
        if tpl in self._angular_me_ls2:
            return self._angular_me_ls2[tpl]
        
        ## Breakpoint to waring the dimension of the spinOrbit_me dictionary
        assert self._angular_me_ls2.__len__() < 1e7, "I might be exploding. Fix me!"        
        
        j_min = max( abs(self.S_bra - self._l), abs(self.J - self._L))
        j_max = min(     self.S_bra + self._l,      self.J + self._L)
        
        sum_ = 0.0
        for j in range(j_min, j_max +1):
            self._j_rel = j
            fac_1 = safe_wigner_6j(self._l, self.L_bra,  self._L, 
                                   self.J , self._j_rel, self.S_bra)
            if self.isNullValue(fac_1): continue
            fac_2 = safe_wigner_6j(self._l_q, self.L_ket,  self._L, 
                                   self.J ,   self._j_rel, self.S_ket)
            if self.isNullValue(fac_2): continue
    
            factor = (2*j + 1) * fac_1 * fac_2
            j_me   = self._spin_isospin_forCOM_Iteration()
            sum_  += factor * j_me
        
        self._angular_me_ls2[tpl] = sum_
        return sum_
    
    def _spin_isospin_forCOM_Iteration(self):
        """
        This matrix element sumarize the <lSjT |v| l'SjT> evaluation.
        
            Implemented (l*S)^2 = (j^2 - l^2 - S^2)^2
        """
        jjp1 = self._j_rel * (self._j_rel + 1)
        llp1 = self._l * (self._l + 1)
        ssp1 = self.S_bra * (self.S_bra + 1)
        
        aux = llp1*(llp1 + 2*ssp1) + jjp1*(jjp1 - 2*llp1) + ssp1*(ssp1 - 2*jjp1)
        
        return 0.25 * aux
        # TEST: (for l*S) return 0.5 * (jjp1 - llp1 - ssp1)
    
    def _LScoupled_MatrixElement(self):
        """ The exchange operators can be done here, reusing the method."""
        return SpinOrbitFiniteRange_JTScheme._LScoupled_MatrixElement(self)
        

class _Quadratic_SpinOrbit_standardCOM_JTScheme(SpinOrbitFiniteRange_JTScheme):
    
    """
    Interaction for   V(r) * (Exchange_terms) * (l*S)^2
        
    """
    _angular_me_ls2 = {}
    
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin, for tensor force impose 
        S = S' = 1, return nothing to skip the bracket spin S=0
        """
        if self.S_bra == 0:
            return []
        return (1, )
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        
        ## Quadratic Spin-orbit involves a rank 2 tensor. the range is up to 2.
        """
        _L_min = max(0, self.L_bra - 2)
        _L_max =        self.L_bra + 2
        gen_ = (l_q for l_q in range(_L_min, _L_max +1))
        return tuple(gen_)
    
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
        skip, spin_me = self._totalSpinTensorMatrixElement()
        if skip:
            return 0
        
        phase   = (-1)**(self.S_bra + self.J)
    
        factor = np.sqrt((2*self.L_bra + 1)*(2*self.L_ket + 1)) * spin_me
    
        return phase * factor * self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    def _totalSpinTensorMatrixElement(self):
        """ <1/2 1/2 (S) | S^[1]| 1/2 1/2 (S)>, only non zero for S=S'=1 
        boolean for skip"""
        if (self.S_bra != self.S_ket) or (self.S_bra == 0):
            return True, 0.0
        
        return False, 6 ## = np.sqrt(6)
    
    def _interactionConstantsForCOM_Iteration(self):
        """
        Explicit expression from the 12-j symbol for rank 2
        """
        tpl = "_".join([str(x) for x in (self.J, self.L_bra, self.L_ket, 
                                         self._l, self._L)])
        if tpl in self._angular_me_ls2:
            return self._angular_me_ls2[tpl]
        
        ## Breakpoint to waring the dimension of the spinOrbit_me dictionary
        assert self._angular_me_ls2.__len__() < 1e7, "I might be exploding. Fix me!"
        
        sum_ = 0.0
        for rr in range(3):
            aux = [0, 0, 0, 0]
            args = [
                (1, 1, rr,  1, 1, 1),
                (self.L_bra, self.S_bra, self.J,   self.S_ket, self.L_ket, rr), 
                (self._l, self._l, rr,  1, 1, self._l),
                (self._l, self._l, rr,  self.L_ket, self.L_bra, self._L),
            ]
            aux[0] = -1/3 if (rr == 0) else 1/6   ## analytical formula
            for k in range(1, 4):
                aux[k] = safe_wigner_6j(*args[k])
                if self.isNullValue(aux[k]): break
    
            factor = (2*rr + 1) * np.prod(aux)
            sum_ += factor
        
        sum_ *= ((-1)**(self._l + self._L))
        sum_ *= self._l * (self._l + 1) * (2*self._l + 1) 
        
        self._angular_me_ls2[tpl] = sum_
        return sum_

## Select only ONE approach to evaluate the quadratic matrix element.
class Quadratic_SpinOrbit_JTScheme(
        # _Quadratic_SpinOrbit_jrelativeEval_JTScheme,
        _Quadratic_SpinOrbit_standardCOM_JTScheme,
    ):
    """
    Select ONE and ONLY ONE of the approaches to evaluate the Quadratic LS
    """
    @classmethod
    def __new__(cls, *args, **kwargs):
        parents = getclasstree([Quadratic_SpinOrbit_JTScheme, ])
        parents = [A.__name__ for A in parents[1][0][1]]
        
        if ((_Quadratic_SpinOrbit_jrelativeEval_JTScheme.__name__ in parents) and 
            (_Quadratic_SpinOrbit_standardCOM_JTScheme.__name__   in parents)):
            raise BaseException("LS quadratic cannot inherit from both "
                                         "J-relative and Standard LS2 classes." )
        
        return super(Quadratic_SpinOrbit_JTScheme, cls).__new__(cls)
    
    @classmethod
    def getMatrixElementClassLogs(cls):
        return f"Size of stored angular momentum matrix elements [{len(cls._angular_me_ls2):}]"
    
class ShortRangeSpinOrbit_JTScheme(SpinOrbitForce_JTScheme):
    
    """
    Short Range Approximation of Spin Orbit force:
          -i W_LS *(p1-p2)* x delta(r1-r2) (p1-p2) * (s1 + s2)
          
    decoupling of the JT reduced m.e. and direct evaluation from gradient formula
    and recurrence relations.
    """
    
    _PARAMETERS_SETTED = False
    NULL_TOLERANCE = 1.0e-9
    
    def _run(self):
        """ Calculate the antisymmetric matrix element value. """
    
        if self.isNullMatrixElement:
            return
        if self.T != 1:
            # Spin-orbit approximation antisymmetrization_ condition 1 - PxPsPt =
            # 1 + delta(tau1, tau2) => T=0 is null (factor 2 in the m.e. below) 
            self._value = 0.0
            self._isNullMatrixElement = True
            return
    
        if self.DEBUG_MODE: 
            XLog.write('nas', ket=self.ket.shellStatesNotation)
    
        self._value = self._LS_recoupling_ME()
    
        self._value *= self.bra.norm() * self.ket.norm()
    
        if self.DEBUG_MODE:
            XLog.write('nas', value=self._value)
            XLog.write('nas', norms=self.bra.norm()*self.ket.norm())
    
    def _LScoupled_MatrixElement(self):
        """
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (JT)>
        This matrix element don't call directly to centerOfMassMatrixElementEvaluation
        since this name is reserved for the center of mass transformation.
        
        Extracted from Dr. Tomas Gonzalez Llarena (1999)
        """
        
        skip, spin_me = self._totalSpinTensorMatrixElement()
        if skip or self.T == 0:
            return 0
        
        # factor = safe_racah(self.L_bra, self.L_ket, self.S_bra, self.S_ket, 1, self.J)
        factor = safe_wigner_6j(self.L_bra,      1,  self.L_ket,
                                self.S_bra, self.J, self.S_ket)
         
        factor *= (-1)**(self.L_ket + self.J) * spin_me
        
        if (self.isNullValue(factor) 
            or not self.deltaConditionsForGlobalQN()):
            # same delta conditions act here, since tensor is rank 1
            return 0
        if self.DEBUG_MODE: XLog.write("LSme")
        
        dir_  = self._L_tensor_MatrixElement()
        exch  = self._L_tensor_MatrixElement(exchanged=True)
        # Notice that the matrix element is not antisymmetrized_  (exchanged is
        # just a name), tbme_ are permutable, then the factor *2 to antisymmetr_ 
        
        factor *= self.PARAMS_FORCE.get(CentralMEParameters.constant)
        aux = 2 * factor * (dir_ + ((-1)**(self.L_bra + self.L_ket))*exch)
        #     ^-- factor 2 for the antisymmetrization_
        
        if self.DEBUG_MODE: 
            XLog.write("LSme", factor=factor, dir=dir_, exch=exch, value=aux)
        return  aux
    
    def _getQRadialNumbers(self, exchanged):
        """ Auxiliary method to exchange the quantum numbers 
        returns: 
            bra_l1, bra_l2, ket_l1, ket_l2, <tuple>(bra1, bra2, ket1, ket2)
        """
        if exchanged:
            l1, l2      = self.ket.l2, self.ket.l1
            l1_q, l2_q  = self.bra.l2, self.bra.l1
            
            qn_cc1 = QN_1body_radial(self.bra.n2, self.bra.l2) # conjugated
            qn_cc2 = QN_1body_radial(self.bra.n1, self.bra.l1) # conjugated
            qn_3   = QN_1body_radial(self.ket.n2, self.ket.l2)
            qn_4   = QN_1body_radial(self.ket.n1, self.ket.l1)
            if self.DEBUG_MODE: 
                XLog.write('Ltens', t="EXCH", wf=(qn_cc1, qn_cc2, qn_3, qn_4))
        else:
            l1, l2      = self.ket.l1, self.ket.l2
            l1_q, l2_q  = self.bra.l1, self.bra.l2
            
            qn_cc1 = QN_1body_radial(self.bra.n1, self.bra.l1) # conjugated
            qn_cc2 = QN_1body_radial(self.bra.n2, self.bra.l2) # conjugated
            qn_3   = QN_1body_radial(self.ket.n1, self.ket.l1)
            qn_4   = QN_1body_radial(self.ket.n2, self.ket.l2)
            if self.DEBUG_MODE: 
                XLog.write('Ltens', t="DIR", wf=(qn_cc1, qn_cc2, qn_3, qn_4))
                
        return l1_q, l2_q, l1, l2, (qn_cc1, qn_cc2, qn_3, qn_4)
    
    def _L_tensor_MatrixElement(self, exchanged=False):
        
        b = self.PARAMS_SHO.get(SHO_Parameters.b_length)
        
        l1_q, l2_q, l1, l2, qqnn = self._getQRadialNumbers(exchanged)
        qn_cc1, qn_cc2, qn_3, qn_4 = qqnn
        
        if ((l1 + l2) + (l1_q + l2_q))%2 == 1:
            return 0
        
        factor = np.sqrt((2*self.L_bra + 1) * (2*self.L_ket + 1)
                         * (2*l1 + 1) * (2*l2 + 1) * (2*l1_q + 1) * (2*l2_q + 1))
        factor /= 4 * np.pi * (b**6)
        
        # Factor include the effect of oscillator length b!= 1
        
        aux0 = ((-1)**self.L_ket) * np.sqrt(2*l2*(l2 + 1)) * (
              safe_3j_symbols(l1_q, l2_q,   self.L_bra, 0,0, 0)
            * safe_3j_symbols(l2,   l1,     self.L_ket, 1,0,-1)
            * safe_3j_symbols(1,    self.L_bra, self.L_ket, 1,0,-1))
        
        aux1 = ((-1)**self.L_bra) * np.sqrt(2*l1_q*(l1_q + 1)) * (
              safe_3j_symbols(l1_q, l2_q,   self.L_bra, 1,0,-1)
            * safe_3j_symbols(l2,   l1,     self.L_ket, 0,0, 0)
            * safe_3j_symbols(1, self.L_ket, self.L_bra, 1,0,-1))
        
        aux2 = ((-1)**(self.L_bra + self.L_ket + l1 + l2)) \
            * np.sqrt(l1_q*(l1_q + 1) * l2*(l2 + 1)) * (
                  safe_3j_symbols(l1_q, l2_q,   self.L_bra, 1,0,-1)
                * safe_3j_symbols(l2,   l1,     self.L_ket, 1,0,-1)
                * safe_3j_symbols(1,    self.L_ket, self.L_bra, 0,1,-1))
        
        if self.DEBUG_MODE:
            XLog.write('Ltens', fact= factor, aux0=aux0, aux1=aux1, aux2=aux2)
            _RadialIntegralsLS.DEBUG_MODE = True
        
        if not self.isNullValue(aux0):
            aux0 *= _RadialIntegralsLS.integral(1, qn_cc1, qn_cc2, qn_3, qn_4, b)
        if not self.isNullValue(aux1):
            aux1 *= _RadialIntegralsLS.integral(1, qn_4, qn_3, qn_cc2, qn_cc1, b)
        if not self.isNullValue(aux2):
            aux2 *= _RadialIntegralsLS.integral(2, qn_cc1, qn_cc2, qn_3, qn_4, b)
        
        if self.DEBUG_MODE: 
            XLog.write('Ltens', value= factor * (aux0 + aux1 + aux2))
        
        return factor * (aux0 + aux1 + aux2)
    
        
    #===========================================================================
    # FIRST VERSION
    #===========================================================================
    
    def _diagonalMatrixElement_Test(self):
        """ 
        Analytic value for diagonal and same orbit matrix elements:
            Moshinsky, Nucl. Phys 8, 19-40 (1958)
            
        Method called after checking S'=S= 1
        """
        if not hasattr(self, 'test_value_diagonal'):
            self.test_value_diagonal = [{}, {}]
        
        if self.L_bra != self.L_ket or self.L_bra % 2 == 0:
            return
        elif (self.bra.n1 != self.bra.n2) and (self.bra.l1 != self.bra.l2):
            return
        else:
            # fool the __eq__ method for jj functions (doesn't matter their js)
            aux_bra = deepcopy(self.bra)
            aux_ket = deepcopy(self.ket)
            
            aux_bra.j1 = aux_bra.j2 
            aux_ket.j1 = aux_bra.j1
            aux_ket.j2 = aux_bra.j1
            
            if not aux_bra == aux_ket:
                return 
        
        l = self.bra.l1
        L = self.L_bra
        
        aux = safe_racah(L, L, 1, 1, 1, self.J) * np.sqrt(6)\
            * ((-1)**(L+1+self.J)) /3
        
        int_R4 = 1 # TODO: calculate
        
        aux_l = np.sqrt((2*L + 1)/(L*(L + 1)))
        aux_l *= ((l + 1)*(2*l + 3)*((2*l + 1)**3)
                  * (safe_clebsch_gordan((l+1), l, L, 0,0,0)**2)
                  * (safe_racah(l,(l+1), L,L, 1,l)**2)
                  * int_R4
                  )
        
        key_ = str(aux_bra)
        
        self.test_value_diagonal[0][key_] = aux * aux_l
        self.test_value_diagonal[1][key_] = aux_l
    
    def _LScoupled_MatrixElement_version1(self):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (JT)>
        This matrix element don't call directly to centerOfMassMatrixElementEvaluation
        since this name is reserved for the center of mass transformation.
        
        This overwrite acts on gradient-tensor 
        """
        # the spin matrix element is 0 unless S=S'=1
        skip, spin_me = self._totalSpinTensorMatrixElement()
        if skip:
            return 0
        
        self._diagonalMatrixElement_Test()
        
        factor = safe_racah(self.L_bra, self.L_ket, 
                            self.S_bra, self.S_ket,
                            1, self.J)
        factor *= np.sqrt(2*self.J + 1)
        # phase resulting from wigner-eckart and the racah_W to 6j coefficients
        factor *= (-1)**(self.S_ket + self.L_bra + self.J)
        
        if (self.isNullValue(factor) 
            or not self.deltaConditionsForGlobalQN()):
            #or ((self.L_ket + self.L_bra) % 2 != 1)): # no parity condition
            # same delta conditions act here, since tensor is rank 1
            _ =0
            return 0
        
        factor *= self.PARAMS_FORCE.get(CentralMEParameters.constant)
        aux = factor * spin_me * self._L_tensor_MatrixElement()
        return  aux
    
    def _L_tensor_MatrixElement_version1(self):
        """
        Tensor that acts on the cross (velocity dependent) product on SHO wave 
        functions
            T^(1) = -i W_LS(r) *[(p1-p2)* x delta(r1-r2) (p1-p2)]^(q)
        
        performs the invert Wigner_Eckart theorem (by getting only the m.e. 
        for q=0, then M'=M) and the uncoupling decomposition for the matrix 
        element (l, m_l)
        """
        
        M_aux = min(self.L_bra, self.L_ket) # it could only fail in L=L'=0
        
        factor = np.sqrt(2*self.L_bra + 1) \
            / safe_clebsch_gordan(self.L_bra, 1, self.L_ket, M_aux,0, M_aux)
        
        qn_cc1 = QN_1body_radial(self.bra.n1, self.bra.l1) # conjugated
        qn_cc2 = QN_1body_radial(self.bra.n2, self.bra.l2) # conjugated
        qn_3   = QN_1body_radial(self.ket.n1, self.ket.l1)
        qn_4   = QN_1body_radial(self.ket.n2, self.ket.l2)
        
        # LS decoupling ensure the L =(l1 + l2) != 0 geometric condition
        sum_ = 0
        for m1 in range(-self.bra.l1, self.bra.l1+1 +1):
            
            m2 = M_aux - m1
            if abs(m2) > self.bra.l2:
                continue
            
            clg_bra = safe_clebsch_gordan(qn_cc1.l, qn_cc2.l, self.L_bra,
                                          m1, m2, M_aux)
            qn_cc1.m_l = m1
            qn_cc2.m_l = m2
            if self.isNullValue(clg_bra):
                continue
            
            # ket part 
            for m3 in range(-self.ket.l1, self.ket.l1+1 +1):
                
                m4 = M_aux - m3
                if abs(m4) > self.ket.l2:
                    continue
                
                clg_ket = safe_clebsch_gordan(qn_3.l, qn_4.l, self.L_ket,
                                               m3, m4 ,M_aux)
                qn_3.m_l = m3
                qn_4.m_l = m4
                if self.isNullValue(clg_ket):
                    continue
                
                aux  = self._gradientMEIntegral(qn_cc1, qn_cc2, qn_3, qn_4)
                aux -= self._gradientMEIntegral(qn_cc2, qn_cc1, qn_3, qn_4)
                aux -= self._gradientMEIntegral(qn_cc1, qn_cc2, qn_4, qn_3)
                aux += self._gradientMEIntegral(qn_cc2, qn_cc1, qn_4, qn_3)
                
                sum_ += clg_bra * clg_ket * aux
        
        return factor * sum_
    
    def _gradientMEIntegral(self, qn_cc_i, qn_cc_j, qn_k, qn_j):
        """
        I(q=0)[ij][kl] ~ (j*)k(grad(i,+1)* grad(k,-1) - grad(i,-1)* grad(k,+1))
        
        calls _SpinOrbitPartialIntegral evaluation.
        """
        # prepare the class arguments:
        if not self._PARAMETERS_SETTED:
            
            args = [
                self.PARAMS_FORCE.get(CentralMEParameters.potential),
                self.PARAMS_SHO.get(SHO_Parameters.b_length),
                self.PARAMS_FORCE.get(CentralMEParameters.mu_length),
                self.PARAMS_FORCE.get(CentralMEParameters.n_power),
            ]
            _SpinOrbitPartialIntegral.setInteractionParameters(*args)
            
            self._PARAMETERS_SETTED = True
            
        return _SpinOrbitPartialIntegral(qn_cc_i, qn_cc_j, qn_k, qn_j).value()
        
        
    

         
        
