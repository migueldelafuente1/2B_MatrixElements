'''
Created on Mar 8, 2021

@author: Miguel
'''
import numpy as np

from helpers.Enums import CouplingSchemeEnum, CentralMEParameters
from helpers.Enums import AttributeArgs, SHO_Parameters
from helpers.Helpers import safe_racah, safe_clebsch_gordan, safe_3j_symbols,\
    safe_wigner_6j

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    MatrixElementException
from matrix_elements.transformations import TalmiTransformation
from helpers.WaveFunctions import QN_2body_LS_Coupling, QN_1body_radial
from helpers.integrals import _SpinOrbitPartialIntegral, _RadialIntegralsLS
from copy import deepcopy
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
            
            for arg, value in kwargs.items():
                if arg in _map:
                    attr_parser = _map[arg]
                    attr_, parser_ = attr_parser
                    kwargs[arg] = parser_(kwargs[arg].get(attr_))
                elif isinstance(value, str):
                    kwargs[arg] = float(value) if '.' in value else int(value)
        
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
    
        factor = safe_racah(self.L_bra, self.L_ket, 
                            self.S_bra, self.S_ket,
                            1, self.J)
        if self.isNullValue(factor) or not self.deltaConditionsForGlobalQN():
            return 0
    
        return  factor * spin_me * self._BrodyMoshinskyTransformation()
    
    def _globalInteractionCoefficient(self):
        phase = (-1)**(self.S_bra + self.L_bra - self.J)
        #phase = (-1)**(self._l + self._L - self.J)
        factor = 1#np.sqrt((2*self.L_bra + 1)*(2*self.L_ket + 1))
    
        return phase * factor * self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        factor = safe_racah(self.L_bra, self.L_ket, 
                            self._l, self._l_q,
                            1, self._L)
        if self.isNullValue(factor):
            return 0
    
        return factor * np.sqrt(self._l * (self._l + 1) * (2*self._l + 1))
    
    
    #===========================================================================
    # Version From "The Harmonic Oscillator" book (excessive values)
    #===========================================================================
    # def centerOfMassMatrixElementEvaluation(self):
    #     #TalmiTransformation.centerOfMassMatrixElementEvaluation(self)
    #     """ 
    #     Radial Brody-Moshinsky transformation, direct implementation for  
    #     non-central spin orbit force.
    #     """
    #     # the spin matrix element is 0 unless S=S'=1
    #     # skip, spin_me = self._totalSpinTensorMatrixElement()
    #     # if skip:
    #     #     return 0
    #
    #     factor = np.sqrt((2*self.L_bra + 1) * (2*self.L_ket + 1))
    #     factor *= ((-1)**(self.L_bra + self.L_ket))
    #
    #     return  factor * self._BrodyMoshinskyTransformation()
    #
    # def _globalInteractionCoefficient(self):
    #     factor = np.sqrt((2*self.L_bra + 1) * (2*self.L_ket + 1))
    #     phase  = ((-1)**(self.L_bra + self.L_ket))
    #
    #     return phase * factor * self.PARAMS_FORCE.get(CentralMEParameters.constant)
    #
    #
    # def _interactionConstantsForCOM_Iteration(self):
    #     # no special internal c.o.m interaction constants for the Central ME
    #     S = self.S_bra
    #     if self.DEBUG_MODE:
    #         XLog.write('C_ls')
    #     factor = 0
    #     for j in range(abs(self.S_bra - self._l), self.S_bra + self._l +1):
    #
    #         aux  = safe_wigner_6j(j,  self._L, self.J,
    #                                  self.L_bra, S, self._l)
    #         aux *= safe_wigner_6j(j,  self._L, self.J,
    #                                  self.L_ket, S, self._l_q)
    #
    #         aux *= 0.5*((j*(j + 1)) - (self._l*(self._l + 1)) - (S*(S + 1)))
    #         aux *= (j*(j + 1))
    #         factor += aux
    #         if self.DEBUG_MODE:
    #             XLog.write('C_ls', j=j, aux_j=aux)
    #
    #     if self.DEBUG_MODE:
    #         XLog.write('C_ls', value=factor)
    #     return factor




class SpinOrbitForce_JTScheme(_TwoBodyMatrixElement_JTCoupled, SpinOrbitForce):
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    def __init__(self, bra, ket, run_it=True):
        
        _TwoBodyMatrixElement_JTCoupled.__init__(self, bra, ket, run_it=run_it)
    
    # def _run(self):
    #     ## First method that runs antisymmetrization by exchange the quantum
    #     ## numbers (X2 time), change 2* _series_coefficient
    #     return _TwoBodyMatrixElement_JTCoupled._run(self)
    
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
        _L_min = max(0, self.L_bra - self.S_bra)
        
        return (L_ket for L_ket in range(_L_min, self.L_bra + self.S_bra +1))
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        return self.centerOfMassMatrixElementEvaluation()







class ShortRangeSpinOrbit_JTScheme(SpinOrbitForce_JTScheme):
    
    """
    Short Range Approximation of Spin Orbit force:
          -i W_LS(r) *(p1-p2)* x delta(r1-r2) (p1-p2) * (s1 + s2)
          
    decoupling of the JT reduced m.e. and direct evaluation from gradient formula
    and recurrence relations.
    """
    
    _PARAMETERS_SETTED = False
    
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
    
        self._value = 2 * self._LS_recoupling_ME()
    
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
        
        factor *= self.PARAMS_FORCE.get(CentralMEParameters.constant)
        aux = factor * (dir_ + ((-1)**(self.L_bra+self.L_ket))*exch)
        
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
        
        
    

         
        
