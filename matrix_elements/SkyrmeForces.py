'''
Created on Dec 28, 2022

@author: Miguel

TODO: Write documentation in docs/

'''
import numpy as np
from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    _standardSetUpForCentralWithExchangeOps
from helpers.Enums import CouplingSchemeEnum, SkyrmeBulkParameters,\
    AttributeArgs, SHO_Parameters, CentralMEParameters, BrinkBoekerParameters
from helpers.Helpers import safe_3j_symbols, almostEqual, safe_wigner_6j,\
    gradientRadialNablaMatrixElements
from helpers.Log import XLog
from helpers.WaveFunctions import QN_1body_radial
from helpers.integrals import _RadialIntegralsLS
from matrix_elements.transformations import TalmiTransformation
from mpmath import cot


class MomentumSquaredDelta_JTScheme(TalmiTransformation, _TwoBodyMatrixElement_JTCoupled):
    '''
    Evaluation of the k^2 operator, being k the relative momentum acting on the
    ket:  1/2i * (\nabla_1 - \nabla_2)
    The decomposition in the COM system will produce the evaluation only in the
    r dependent functions, simplifying the angular momentum algebra
    '''
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
        """ 
        Return ket states <tuple> of the total spin
        """
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum.
        """
        return (self.L_bra, )
    
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
        """ This condition ensure the anti-symmetrization (without calling 
        exchanged the matrix element)"""
        if ((abs(self._n - self._n_q) < 2) or 
            ((self.S_bra + self.T + self._l) % 2 == 1)):
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
    
    def _globalInteractionCoefficient(self):
        # no special interaction constant for the Central ME
        return self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    def talmiIntegral(self):
        """ Delta interaction can only have the term p=0 which is constant """
        return 0 if self._p != 0 else 0.7978845608028654  ## np.sqrt(2 / np.pi)
    
    def _interactionConstantsForCOM_Iteration(self):
        """
        Includes the radial matrix elements and the angular terms 
        """
        factor = np.sqrt((2*self.L_bra + 1) / 3) / ((2*self._l + 1) * 4)
        n, l, n_q = self._n, self._l, self._n_q
        
        ##  l'' = l + 1
        # n'' = n (aux1) and n - 1 (aux2) 
        aux1, aux2 = 0, 0
        if (n_q == n):
            aux1 = n + l + 1.5
            if (n > 0): 
                aux2 = n
        elif (n_q == n + 1):
            aux1 = np.sqrt((n + l + 1.5) * (n + 1))
        elif (n > 0) and (n_q == n - 1):
            aux2 = np.sqrt((n + l + 0.5) * n )
        
        F_Lp1  = (l + 1) * (aux1 + aux2)
        
        ##  l'' = l + 1
        # n'' = n (aux1) and n + 1 (aux2) 
        F_Lm1  = 0
        if (l > 0):
            aux1, aux2 = 0, 0
            if (n_q == n):
                aux1 = np.sqrt((n + l + 1.5) * (n + l + 0.5))
                aux2 = n + 1
            elif (n_q == n + 1):
                aux2 = np.sqrt((n + l + 1.5) * (n + 1))
            elif (n > 0) and (n_q == n - 1):
                aux1 = np.sqrt((n + l + 0.5) * n )
            
            F_Lm1 = l * (aux1 + aux2)
        
        return factor * (F_Lp1 + F_Lm1)

class ShortRangeSpinOrbit_COM_JTScheme(MomentumSquaredDelta_JTScheme):
    """
    Alternative Matrix element for S*((k*)\times k)
    """
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin
        """
        if self.S_bra == 0:
            return tuple()
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum.
        """
        Lmin = max(0, self.L_bra - 1)
        return (L for L in range(Lmin, self.L_bra + 1) )
    
    def _validKet_relativeAngularMomentums(self):
        """ 
        Analyzing the radial matrix elements, the decomposition allows
        l''=l +/-1 and l' +/- 1, therefore l' = l, l +/- 2, but there is a 6j
            {1  1   1}
            {l' l l''}  allowing only l + l' = 1, then l' = l only.
        """
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
        """ This condition ensure the anti-symmetrization (without calling 
        exchanged the matrix element) Note: l' = l,  S' = S = 1 """
        # if self._p != 0: return False
        if ((abs(self._n - self._n_q) < 2) and 
            ((self.S_bra + self.T + self._l) % 2 == 1)):
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
        factor  = safe_wigner_6j(self.L_bra, 1, self.J,
                                 1, self.L_ket, 1)
        factor *= np.sqrt(6 * 3 * (2*self.L_bra + 1) * (2*self.L_ket + 1))
        factor *= (-1)**(self.J + 1)
        return factor * self.PARAMS_FORCE.get(CentralMEParameters.constant) / 2
    
    def _interactionConstantsForCOM_Iteration(self):
        """
        Includes the radial matrix elements and the angular terms 
        """
        factor  = self._radialSeries_matrixElements()
        if self.isNullValue(factor): 
            return 0
        factor *= safe_wigner_6j(self._l, self._L, self.L_bra,
                                 self.L_ket,    1, self._l_q)
        factor *= (-1)**(self._L + self._l_q + 1)
        return factor / (4*np.pi * self.PARAMS_SHO[SHO_Parameters.b_length]**2)
        
    def _radialSeries_matrixElements(self):
        """
        
        """
        ## Note: Redundant
        assert (self._l == self._l_q), "Error, l' != l"
        assert abs(self._n - self._n_q) < 2, "Error, |n-n'| != 0,1 "
        
        n, l, n_q = self._n, self._l, self._n_q
        B_nlnl_onrun = self._B_coefficient()
        # if self.isNullValue(B_nlnl_onrun): return 0
        
        sum_ = 0
        _min = l-1 if l > 0 else l+1
        for l_qq in range(_min, l+2, 2):  # l-1,    l+1
            fac_lp1 = safe_wigner_6j(1, 1, 1, self._l_q, self._l, l_qq)
            
            _min = n-1 if n>0 else n
            for n_qq in range(_min, n+2): # n-1, n, n+1
                
                f_nl = gradientRadialNablaMatrixElements(n_qq, l_qq, n, l)
                if self.isNullValue(f_nl): continue
                
                f_nl *= safe_wigner_6j(1, 1, 1,
                                       self._l_q, self._l, l_qq)
                if self.isNullValue(f_nl): continue
                
                if (l_qq == l + 1):
                    f_nl *=  (2*l + 1)**3 * np.sqrt((l + 1) / (2*l + 3))
                elif (l_qq == l - 1):
                    f_nl *= -(2*l + 1)**3 * np.sqrt( l / (2*l - 1))
                
                aux  = 0
                _min = n_q-1 if n_q>0 else n_q
                for N in range(_min, n_q+2): # n'-1, n', n'+1
                    f_NL = gradientRadialNablaMatrixElements(N, l_qq, n_q, l, 
                                                             only_radial=True)
                    if self.isNullValue(f_NL): continue
                    
                    B_Nn = self.BCoefficient(n_q, l_qq, N, l_qq, 0,
                                             self.PARAMS_SHO[SHO_Parameters.b_length])
                    if self.isNullValue(B_Nn): continue
                    
                    aux += f_NL * B_Nn
                
                sum_ += fac_lp1 * aux * f_nl
        
        return sum_ #/ B_nlnl_onrun

class SkrymeBulk_JTScheme(_TwoBodyMatrixElement_JTCoupled):
# class SkrymeBulk_JTScheme(_TwoBodyMatrixElement_Antisym_JTCoupled):
    
    '''
    Set up the Delta and Momentum parts of the Skyrme interaction, this element
    skips the spin-orbit short range and density dependent term as in the D1S.
    '''
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    _PARAMETERS_SETTED = False
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments the Bulk part of the Syrme III interaction, this method 
        bypasses calling from main or io_manager by instanciating and executing 
        the momentum matrix elements and the delta.
        
        :b_length <float> fm  oscillator lenght
        :t0  <float> MeV fm^3 delta part:             t0*(1 + x0*P_s)delta(r12)
        :t1  <float> MeV fm^5 momentum modulus:   t1*(|k|^2+|k'|^2)delta(r12)/2
        :t2  <float> MeV fm^5 scalar prod of momentum:        t2*k'*delta(r12)k  
        :x0  <float>            Spin exchange factor for the spatial delta part
        """
        
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        t_args_map = {
            SkyrmeBulkParameters.t0  : (AttributeArgs.value, float),
            SkyrmeBulkParameters.x0  : (AttributeArgs.value, float),
            SkyrmeBulkParameters.t1  : (AttributeArgs.value, float),
            SkyrmeBulkParameters.t2  : (AttributeArgs.value, float),
        }
        if True in map(lambda a: isinstance(a, dict), kwargs.values()):
            # when calling from io_manager, arguments appear as dictionaries, 
            # parse them            
            kwargs = cls._automaticParseInteractionParameters(t_args_map, kwargs)
        
        _b = SHO_Parameters.b_length
        assert _b in kwargs
        cls.PARAMS_SHO[_b] = float(kwargs[_b])
        
        ## pass the t and x params (all must be in)
        for param in t_args_map:
            assert param in kwargs, f"[{param}] must be present for interaction arguments."
            cls.PARAMS_FORCE[param] = kwargs[param]
    
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
        
    def _run(self):
        ## the parity condition must be included, then check the usual restrictions
        if self.parity_bra != self.parity_ket:
            self._value = 0
        # return _TwoBodyMatrixElement_Antisym_JTCoupled._run(self)
        return _TwoBodyMatrixElement_JTCoupled._run(self)
    
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin, for delta interaction, 
        <S|| ||S'>= delta_SS'
        """
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        
        OJO: Moshinski, lambda' = lambda, lambda +- 1!!! as condition
        in the C_LS
        """
        # _L_min = max(0, self.L_bra - self.S_bra)
        # gen_ = (L_ket for L_ket in range(_L_min, self.L_bra + self.S_bra +1))
        # return tuple(gen_)
        return (self.L_bra, )
    
    def _LScoupled_MatrixElement(self):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)> This term call
        the tree terms for the LS scheme, all have the same LS=LS' restrictions.
        """
        delta_term = self._LS_DeltaTerm_Sky()
        modsk_term = self._LS_ModsKterm_Sky()
        lksrk_term = self._LS_KLdotKRterm_Sky()
        
        # antisymfact = (1 - ((-1)**(self.T + 1 +
        #                            self.J + (self.ket.j1 + self.ket.j2)//2)) )
        
        return (delta_term + modsk_term + lksrk_term) #* antisymfact
        
        
        
        
        
    
    def _LS_DeltaTerm_Sky(self):
        if self.DEBUG_MODE: XLog.write("DSky_me")
        
        dir_ = (safe_3j_symbols(self.bra.l1, self.bra.l2,  self.L_bra, 0,0,0) *
                safe_3j_symbols(self.ket.l1, self.ket.l2,  self.L_ket, 0,0,0))
        
        factor  = ((2*0  + 1)*(2*self.bra.l1 + 1)*(2*self.bra.l2 + 1)*
                   (2*self.ket.l1 + 1)*(2*self.ket.l2 + 1))**.5 
        factor *= self.PARAMS_FORCE.get(SkyrmeBulkParameters.t0) / (4*np.pi)
        factor *= (1 - (self.PARAMS_FORCE.get(SkyrmeBulkParameters.x0)
                        *((-1)**self.S_bra)) )
        
        phase = (1 - ((-1)**(self.S_ket + self.T)))
        aux = factor * dir_ * phase
        #     ^-- factor 2 for the antisymmetrization_
        
        if self.DEBUG_MODE: _RadialIntegralsLS.DEBUG_MODE = True
        if not almostEqual(aux, 0.0):
            _, _, _, _, qqnn = self._getQRadialNumbers(False)
            qn_cc1, qn_cc2, qn_3, qn_4 = qqnn
            b_ = self.PARAMS_SHO.get(SHO_Parameters.b_length)
            aux *= _RadialIntegralsLS.integral(2, qn_cc1,qn_cc2, qn_3,qn_4, b_)
            aux *= b_*b_
            
            ## comparison with SDI Suhonen
            # aux *= -4*np.pi
            # aux *= (-1)**(self.bra.n1+self.bra.n2+self.ket.n1+self.ket.n2)
        
        if self.DEBUG_MODE: XLog.write("SkyDlt", factor=factor, dir=dir_, value=aux)
        return  aux
    
    
    def _LS_ModsKterm_Sky(self):
        
        return 0
    
    
    def _LS_KLdotKRterm_Sky(self):
        
        return 0

    
        