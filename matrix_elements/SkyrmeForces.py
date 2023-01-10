'''
Created on Dec 28, 2022

@author: Miguel
'''
import numpy as np
from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled
from helpers.Enums import CouplingSchemeEnum, SkyrmeBulkParameters,\
    AttributeArgs, SHO_Parameters
from helpers.Helpers import safe_3j_symbols, almostEqual
from helpers.Log import XLog
from helpers.WaveFunctions import QN_1body_radial
from helpers.integrals import _RadialIntegralsLS


 


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

    
        