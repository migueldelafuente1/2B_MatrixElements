'''
Created on 21 mar 2025

@author: delafuente
'''
import numpy as np

from matrix_elements.transformations import TalmiIndependentMoshinskyTransformation
from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled
from helpers.Enums import CentralMEParameters, SHO_Parameters,\
    CouplingSchemeEnum
from helpers.Helpers import gamma_half_int, Constants




class RelativeMomentumSquared_JTScheme(TalmiIndependentMoshinskyTransformation, 
                                       _TwoBodyMatrixElement_JTCoupled):
    
    """
    Evaluates the interaction for k^2 = Laplacian, such as in the kinetic term
    for the relative momentum = p_1 - p_2
    
    require to specify the radialRelativeMatrixElement matrix element
    """
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    _BREAK_ISOSPIN = False
    
    def __init__(self, bra, ket, run_it=True):
        _TwoBodyMatrixElement_JTCoupled.__init__(self, bra, ket, run_it=run_it)
    
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
        
        cls._integrals_p_max = dict([(i, -1) for i in range(-2, 3, 1)])
        cls._talmiIntegrals  = dict([(i, list()) for i in range(-2, 3, 1)])
    
    def _globalInteractionCoefficient(self):
        """
        for _kinetic interaction is 
        """
        b_len      = self.PARAMS_SHO.get(SHO_Parameters.b_length)
        hbar_omega = self.PARAMS_SHO.get(SHO_Parameters.hbar_omega)
        if b_len == None:
            return hbar_omega * 2
        else:
            return 2 * (Constants.HBAR_C / b_len)**2 / Constants.M_MEAN
    
    def deltaConditionsForGlobalQN(self):
        return True
    
    def _deltaConditionsForCOM_Iteration(self):
        """ Antisymmetrization condition (*2 in BrodyMoshinkytransformation)"""
        if ((self.S_bra + self.T + self._l) % 2 == 1):
            return True
        return False
    
    def _validKet_relativeAngularMomentums(self):
        return (self._l, )
    
    def _validKetTotalAngularMomentums(self):
        return (self.L_bra, )
    
    def _validKetTotalSpins(self):
        return (self.S_bra, )
    
    def _LScoupled_MatrixElement(self):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """    
        # Radial Part
        return self.centerOfMassMatrixElementEvaluation()
    
    def centerOfMassMatrixElementEvaluation(self):
        #TalmiTransformation.centerOfMassMatrixElementEvaluation(self)
        """ 
        Radial Brody-Moshinsky transformation
        """       
        if not self.deltaConditionsForGlobalQN(): return 0
        
        return  self._BrodyMoshinskyTransformation()
    
    def _interactionConstantsForCOM_Iteration(self):
        """
        angular dependent terms for l, l' or simmilar
        """
        # return (1/1) * (2*self.L_bra + 1) / (2*self._l + 1)
        return (1/16) # / (2*self._l + 1)
    
    @staticmethod
    def _radialMultipole(p, N):
        """
        I_p of (r^N)
        """
        aux = gamma_half_int(2*p + 3 + N) - gamma_half_int(2*p + 3)
        return np.exp(aux + 0.5*N*np.log(2))
    
    @classmethod
    def _calculateIntegrals(cls, N, n_integrals =1):
        """
        MODIFIED: obtains directly the r^N expressed here
        """        
        for p in range(cls._integrals_p_max[N] + 1, 
                       cls._integrals_p_max[N] + n_integrals +1):
            
            cls._talmiIntegrals [N].append(cls._radialMultipole(p, N))
            
            cls._integrals_p_max[N] += 1
    
    def talmiIntegral(self, N):
        """ 
        Get or update Talmi integrals for the calculations
        :N is the order of the power r^N
        """
        assert N in (-2, -1, 0, 1, 2), "Invalid power"
        if self._p > self._integrals_p_max[N]:
            self._calculateIntegrals(N, n_integrals = max(self.rho_bra, 
                                                          self.rho_ket, 1), )
        return self._talmiIntegrals[N].__getitem__(self._p)
    
    def _matrixElementRadial_byPower(self, l_2, N):
        
        _min =  1 if l_2 > 0 else 0
        _max = -1 if l_2 > 0 else 0
        sum_ = 0
        self._p = 0
        for p in range(self._l + _min, 
                       self._l + self._n + self._n_q + _max +1):
            self._p = p
            # sum_ += self.BCoefficient(self._n, self._l, self._n_q-N, self._l_q+N, p) * self.talmiIntegral(N)
            qqnn = (self._n, self._l, self._n_q-l_2, self._l_q+l_2, p)
            sum_ += self._B_coefficient(b_param=1, specific_qqnn=qqnn) * self.talmiIntegral(N)
        return sum_
    
    def radialRelativeMatrixElement(self):
        """
        Evaluating the different radial quantum numbers corresponding to 
        the COM system.
        """
        n_q, l = self._n_q, self._l
        b_len = 1 #self.PARAMS_SHO[SHO_Parameters.b_length]
        
        ## NOTE: b_param is moved to a global constant 1/b^2 
        ##       (taken out from the <nl|r^N|n''l''> and the B coefficient)
        
        aux_0 = [-(2*l + 3) / (b_len**2),    b_len**(-4)]
        aux_0[0] *= self._matrixElementRadial_byPower(0,  0)
        aux_0[1] *= self._matrixElementRadial_byPower(0,  2)
        aux_0 = sum(aux_0)
        
        aux_1, aux_2 = 0, 0
        if (n_q > 0):
            aux_1 = [2*l + 3, -2 / (b_len**2)]
            aux_1[0] *= self._matrixElementRadial_byPower(1, -1)
            aux_1[1] *= self._matrixElementRadial_byPower(1,  1)
            aux_1 = sum(aux_1) * 2 * np.sqrt(n_q) / b_len
            
            if (n_q > 1):
                aux_2  = 4 * np.sqrt(n_q * (n_q - 1)) / (b_len**2)
                aux_2 *= self._matrixElementRadial_byPower(2,  0)
        
        return aux_0 - aux_1 + aux_2

class TotalMomentumSquared_JTScheme(RelativeMomentumSquared_JTScheme):
    
    """
    Evaluation of total kinetic term P = p1 + p2 (center of mass)
    
    Requires the modification of the laplacian operator, in this case, it will 
    act on the NL system, having the same radial and angular expressions on this
    quantum numbers.
    
    NOTE: Probably useless, since the Moshinsky transformation is symmetrical
    for N and L (each group of nlNL will have another NLnl for the same state).
    
    However, if want to compute p1 + p2 = P_R, then the with respect of the 
    relative momentum p1 - p2 = 2 * p_r, the result should be divided by 2.
    """
    
    def _globalInteractionCoefficient(self):
        """
        for _kinetic interaction is 
        """
        b_len      = self.PARAMS_SHO.get(SHO_Parameters.b_length)
        hbar_omega = self.PARAMS_SHO.get(SHO_Parameters.hbar_omega)
        if b_len == None:
            return hbar_omega
        else:
            return 0.5 * (Constants.HBAR_C / b_len)**2 /  Constants.M_MEAN 