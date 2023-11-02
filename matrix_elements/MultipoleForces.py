'''
Created on 23 oct 2023

@author: delafuente
'''
import numpy as np

from helpers.Helpers import  safe_wigner_6j,\
    safe_3j_symbols, almostEqual, safe_wigner_9j

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_Antisym_JTCoupled, \
    MatrixElementException, _OneBodyMatrixElement_jjscheme
from helpers.Enums import CentralMEParameters, AttributeArgs,\
    SHO_Parameters, MultipoleParameters
from helpers.Log import XLog
from helpers.integrals import _RadialIntegralsLS, _RadialMultipoleMoment
from helpers.WaveFunctions import QN_1body_radial, QN_2body_jj_JT_Coupling

from copy import deepcopy, copy



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


class _MultipoleMoment_1BME(_OneBodyMatrixElement_jjscheme):
    
    _VERSION_QQ_EXPLICIT = False
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        print(" [WARNING] setInteractionParameters must be called from the "
              "two body class, current parameters:")
        print("* PARAMS_FORCE: {}\nPARAMS_SHO: {}"
              .format(cls.PARAMS_FORCE, cls.PARAMS_SHO))
        pass
        
    def _run(self):
        
        if self.isNullMatrixElement: return
        if self.DEBUG_MODE: 
            XLog.write('nas_me', ket=self.ket.shellStatesNotation)
        
        self._value = 0.0
        
        L = self.PARAMS_FORCE[CentralMEParameters.n_power]
        C = self.PARAMS_FORCE[CentralMEParameters.constant]
        
        if not self._VERSION_QQ_EXPLICIT:
            ang_cent  = self._AngularCoeff_Central(L)
        else:
            ang_cent = self._AngularCoeff_Central_explicit(L)
        
        if almostEqual(abs(ang_cent), 0, self.NULL_TOLERANCE): return
        
        if self.DEBUG_MODE:
            XLog.write('nas_me', lambda_=L, value=self._value)
        
        if self._VERSION_QQ_EXPLICIT: L = 2*L  ## 2*L if using the old expression
        
        rad_d = self._RadialCoeff(L)   
        self._value  = C * ang_cent * rad_d
    
    def _AngularCoeff_Central_explicit(self, lambda_):
        """ First version, it is wrong """
        j_a, j_b = self.bra.j, self.ket.j
    
        if ( abs(self.bra.l - self.ket.l) > lambda_ or 
                (self.bra.l + self.ket.l  < lambda_)): 
            return 0
        if ( ((self.bra.l + self.ket.l) % 2 == 1) or (j_a != j_b)) :
            return 0 # parity condition form _redAngCoeff
    
        val = (  safe_3j_symbols(lambda_, lambda_, 0,  0, 0, 0)
               * safe_3j_symbols(j_a / 2, j_b / 2, 0, .5, -.5, 0))
    
        factor = ((j_a + 1) * (2*lambda_ + 1)) / (4 * np.pi)
        phs = (-1)**((j_a + j_b)//2 + (1 - j_a)//2 + lambda_)
    
        return phs * factor * val
    
    def _AngularCoeff_Central(self, lambda_):
        """ 
        Direct evaluation from the < alpha|Q|beta> expresion and the WE-theorem.
        """
        j_a, j_b = self.bra.j, self.ket.j
        
        if ( abs(self.bra.l - self.ket.l) > lambda_ or 
                (self.bra.l + self.ket.l  < lambda_)): 
            return 0
        if ( ((self.bra.l + self.ket.l + lambda_) % 2 == 1) ) :
            return 0 # parity condition form _redAngCoeff
        
        val = safe_3j_symbols(j_a / 2, j_b / 2, lambda_, .5, -.5, 0)
        
        factor = np.sqrt((j_a + 1) * (j_b + 1) * (2*lambda_ + 1) / (4 * np.pi))
        phs = (-1)**((3 + j_b)//2 + lambda_)
        
        return phs * factor * val
    
    def _RadialCoeff(self, lambda_):
        """
        Implementation of the integral for the multipole
        TEST(passed), Look the integral table 6.2 from Suhonen rad_ac/b**lambda_
        """
        b = self.PARAMS_SHO.get(SHO_Parameters.b_length)
        
        qnr_a = QN_1body_radial(self.bra.n, self.bra.l) # conjugated
        qnr_b = QN_1body_radial(self.ket.n, self.ket.l)
        
        rad_ac = _RadialMultipoleMoment.integral(lambda_, qnr_a, qnr_b, b)
        
        return rad_ac 
        
        
        


class MultipoleMoment_JTScheme(_Multipole_JTScheme):
    
    """
    Matrix element for the Central Quadrupole-Quadrupole Interaction:
        V = Q_2 * Q_2     Q_2m = r^2 Y_2m
    differs from the Multipole-Delta interaction on having only L=2 term.
    """
    
    ONEBODY_MATRIXELEMENT = _MultipoleMoment_1BME
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """
        Arguments for a general potential form delta(r; mu_length, constant)
        
        :b_length 
        :hbar_omega
        :constants <dict> 
            constant : <float>
            n_order  : <integer>      
        """
        cst_ = CentralMEParameters.constant
        lmd_ = CentralMEParameters.n_power
        kwargs[cst_] = float(kwargs[cst_][AttributeArgs.value])
        kwargs[lmd_] = int  (kwargs[lmd_][AttributeArgs.value])
        
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        params_and_defaults = {
            SHO_Parameters.b_length      : 1,
            SHO_Parameters.hbar_omega    : 1,
            CentralMEParameters.constant : 1,
            CentralMEParameters.n_power  : 0,
        }
        
        for param, default in params_and_defaults.items():
            value = kwargs.get(param)
            value = default if value == None else value
            
            if param in SHO_Parameters.members():
                cls.PARAMS_SHO[param] = value
            else:
                if param in (CentralMEParameters.potential,
                             CentralMEParameters.mu_length,) :
                    print(" [WARNING] Multipole interaction has only power-type"
                          " potential and mu_length = 1, ignoring these arguments.")
                cls.PARAMS_FORCE[param] = value
        
        cls.ONEBODY_MATRIXELEMENT.PARAMS_FORCE = deepcopy(cls.PARAMS_FORCE)
        cls.ONEBODY_MATRIXELEMENT.PARAMS_SHO   = deepcopy(cls.PARAMS_SHO  )
    
    def _run(self):
        
        if self.isNullMatrixElement: return
        
        phase, exchanged_ket = self.ket.exchange()
        self.exchange_phase = phase
        self.exch_2bme = self.__class__(self.bra, exchanged_ket, run_it=False)
        
        if self.DEBUG_MODE: 
            XLog.write('nas_me', ket=self.ket.shellStatesNotation)
        
        self._value = 0.0
        
        L = self.PARAMS_FORCE[CentralMEParameters.n_power]
        C = self.PARAMS_FORCE[CentralMEParameters.constant]
        
        #-----------------------------------------------------------------------
        # NOTE ON THE CONDITIONS FOR L:
        # The following condition _L_min(max) is for the possible orbital L to be
        # between bra and ket (like for any central force), but L order is only 
        # available for the range between bra and ket scalar components 
        # (expand the interaction in LS scheme and see the conditions on the 6j)
        # 
        # The correct condition is la + L = lc(ld) and lb + L = ld(lc) for the 
        # direct(exchange) terms, but require to apply for dir and exch separately
        # 
        # Whit this condition and the parity check, the previous one is fulfilled
        #-----------------------------------------------------------------------
        _L_min = max(abs(self.bra.l1-self.bra.l2), abs(self.ket.l1-self.ket.l2))
        _L_max = min(    self.bra.l1+self.bra.l2 ,     self.ket.l1+self.ket.l2 )
        if (L < _L_min) or (L > _L_max):
            return
        
        ang_cent_d  = self._AngularCoeff_Central(L)
        ang_cent_e  = self.exch_2bme._AngularCoeff_Central(L) 
        ang_cent_e *= self.exchange_phase
        
        if self.DEBUG_MODE:
            XLog.write('nas_me', lambda_=L, value=self._value, 
                       norms=self.bra.norm()*self.ket.norm())
        
        rad_d = self._RadialCoeff(L)
        rad_e = self.exch_2bme._RadialCoeff(L) 
        
        self._value  = (ang_cent_d * rad_d) - (ang_cent_e * rad_e)
        self._value *= C * self.bra.norm() * self.ket.norm()
    
    
    def _AngularCoeff_Central(self, lambda_):
        
        j_a, j_b = self.bra.j1, self.bra.j2
        j_c, j_d = self.ket.j1, self.ket.j2
        
        if ( ((self.bra.l1 + self.ket.l1 + lambda_) % 2 == 1) or  
             ((self.bra.l2 + self.ket.l2 + lambda_) % 2 == 1)) :
            return 0 # parity condition form _redAngCoeff
        
        val = safe_wigner_6j(j_a / 2, j_b / 2, self.J, 
                             j_d / 2, j_c / 2, lambda_)
        if not almostEqual(val, 0, self.NULL_TOLERANCE): 
            val *= (  safe_3j_symbols(j_a / 2, j_c / 2, lambda_, .5, -.5, 0)
                    * safe_3j_symbols(j_b / 2, j_d / 2, lambda_, .5, -.5, 0))
        
        factor  = ((j_a + 1) * (j_b + 1) * (j_c + 1) * (j_d + 1))**0.5
        factor *= ((2*lambda_) + 1 ) / (4 * np.pi)
        
        phs = (-1)**((j_a + j_b + j_c + j_d)//2 + self.J - 1)
        
        return phs * factor * val
    
    
    def _RadialCoeff(self, lambda_):
        
        """Implementation of the integral for the multipole"""
        
        b = self.PARAMS_SHO.get(SHO_Parameters.b_length)
        
        qnr_a = QN_1body_radial(self.bra.n1, self.bra.l1) # conjugated
        qnr_b = QN_1body_radial(self.bra.n2, self.bra.l2) # conjugated
        qnr_c = QN_1body_radial(self.ket.n1, self.ket.l1)
        qnr_d = QN_1body_radial(self.ket.n2, self.ket.l2)
        
        rad_ac = _RadialMultipoleMoment.integral(lambda_, qnr_a, qnr_c, b)
        rad_bd = _RadialMultipoleMoment.integral(lambda_, qnr_b, qnr_d, b)
        
        return rad_ac * rad_bd
    


