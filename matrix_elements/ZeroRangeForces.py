'''
Created on Oct 14, 2021

@author: Miguel
'''
from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    MatrixElementException
from helpers.Enums import CentralMEParameters, AttributeArgs, SHO_Parameters,\
    SDIParameters
from helpers.Log import XLog
from helpers.Helpers import safe_3j_symbols, almostEqual
from helpers.WaveFunctions import QN_2body_jj_JT_Coupling

_SDI_Attributes = AttributeArgs.ForceArgs.SDI

class SDI_JTScheme(_TwoBodyMatrixElement_JTCoupled):
    '''
    Analytical SDI matrix element don't require LS recoup_ nor explicit antisymm_
    override the __init__ method to directly evaluate it (run_it= ignored)
    '''
    RECOUPLES_LS = False
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a SDI potential form delta(r; mu_length, constant)
        
        :b_length 
        :hbar_omega
        :constants <dict> 
            with available constants AT1, AT0 for SDI and B, C for MSDI
        
        Brussaard_ & Glaudemans_ book (1977)
        *      Parameters available in Table 6.3 (pg 116)
        *       **  Considerations of the parameters:
        *     Change of variables with Suhonen_'s book Notation:
        *         A_T(suh_) = 2 A_T(Bruss_)
        *         B_1(suh_) = B(Bruss_) +   C(Bruss_) 
        *         B_0(suh_) = C(Bruss_) - 3*B(Bruss_)
        """
        
        for param in AttributeArgs.ForceArgs.SDI.members():
            val = kwargs[SDIParameters.constants].get(param)
            if val != None:
                val = float(val)
            kwargs[param] = val
        del kwargs[SDIParameters.constants]
            
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        params_and_defaults = {
            SHO_Parameters.b_length     : 1,
            SHO_Parameters.hbar_omega   : 1,
            AttributeArgs.ForceArgs.SDI.A_T0    : 0.5,
            AttributeArgs.ForceArgs.SDI.A_T1    : 0.5,
            AttributeArgs.ForceArgs.SDI.B       : 0,
            AttributeArgs.ForceArgs.SDI.C       : 0
        }
        
        for param, default in params_and_defaults.items():
            value = kwargs.get(param)
            value = default if value == None else value
            
            if param in SHO_Parameters.members():
                cls.PARAMS_SHO[param] = value
            else:
                cls.PARAMS_FORCE[param] = value
                
        B = cls.PARAMS_FORCE[AttributeArgs.ForceArgs.SDI.B]
        C = cls.PARAMS_FORCE[AttributeArgs.ForceArgs.SDI.C]
        
        cls._evaluateMSDI = False
        if not (almostEqual(B, 0, 1e-9) and almostEqual(C, 0, 1e-9)):
            cls._evaluateMSDI = True
            cls._msdi_T0 = C - 3*B
            cls._msdi_T1 = B + C
            
    
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
            print("Bra JT [{}]doesn't match with ket's JT [{}]"
                  .format(bra.J, bra.T, ket.J, ket.T))
            self._value = 0.0
        else:
            self._nullConditionForSameOrbit()
        
        if not self.isNullMatrixElement and run_it: # always run it
            self._run()
              
    def _run(self):
        
        if self.isNullMatrixElement:
            return
    
        if self.DEBUG_MODE: 
            XLog.write('nas_me', ket=self.ket.shellStatesNotation)
    
        # antisymmetrization_ taken in the inner evaluation            
        self._value  = self._RadialCoeff()
        self._value *= self._AngularCoeff()
        
        if self.DEBUG_MODE:
            XLog.write('nas_me', value=self._value, norms=self.bra.norm()*self.ket.norm())
        
        self._value *= self.bra.norm() * self.ket.norm()
        
        if self._evaluateMSDI:
            ## V_MSDI = V_SDI + B * <tau(1) · tau(2)> + C (only off-diagonal)
            if self.T == 0:
                self._value += self._msdi_T0 
            elif self.T == 1:
                self._value += self._msdi_T1
        
    def _RadialCoeff(self):
        ## TODO: Implement
        phs = ((-1)**(self.bra.n1 + self.bra.n2 + self.ket.n1 + self.ket.n2 + 1))
        if self.T == 0:
            return 0.25 * phs * self.PARAMS_FORCE[AttributeArgs.ForceArgs.SDI.A_T0]
        if self.T == 1:
            return 0.25 * phs * self.PARAMS_FORCE[AttributeArgs.ForceArgs.SDI.A_T1]
    
    def _AngularCoeff(self):
        
        j_a, j_b = self.bra.j1, self.bra.j2
        j_c, j_d = self.ket.j1, self.ket.j2
        
        phs = 1 + ((-1)**(self.bra.l1 + self.bra.l2 + self.ket.l1 + self.ket.l2))
        factor = ((j_a + 1) * (j_b + 1) * (j_c + 1) * (j_d + 1))**0.5
        
        dir_, exh_ = 0, 0
        
        if self.T == 0:
            dir_ = 2 * (  safe_3j_symbols(j_a / 2, j_b / 2, self.J, .5, .5, -1)
                        * safe_3j_symbols(j_c / 2, j_d / 2, self.J, .5, .5, -1))
        
        if (self.ket.l1 + self.ket.l2 + self.J + self.T) % 2 == 1:
            exh_ = 2 * (  safe_3j_symbols(j_a / 2, j_b / 2, self.J, .5, -.5, 0)
                        * safe_3j_symbols(j_c / 2, j_d / 2, self.J, .5, -.5, 0))
            exh_ *= (-1)**(self.bra.l1 + self.ket.l1 + (j_b + j_d)/2)
        
        return phs * factor * (dir_ - exh_)
        
    ## return void LS valid L S for SpeedRunner to work with this m.e
    def _validKetTotalSpins(self):
        return tuple()
    
    def _validKetTotalAngularMomentums(self):
        return tuple()
    