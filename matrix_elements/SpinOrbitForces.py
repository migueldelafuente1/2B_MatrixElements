'''
Created on Mar 8, 2021

@author: Miguel
'''
import numpy as np
from sympy.physics.wigner import racah

from helpers.Enums import CouplingSchemeEnum, CentralMEParameters
from helpers.Enums import AttributeArgs, SHO_Parameters
from helpers.Helpers import safe_racah

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    MatrixElementException
from matrix_elements.transformations import TalmiTransformation
from helpers.WaveFunctions import QN_2body_LS_Coupling

class SpinOrbitForce(TalmiTransformation): # _TwoBodyMatrixElement_JTCoupled, 
    
    COUPLING = (CouplingSchemeEnum.L, CouplingSchemeEnum.S)
    
    def __init__(self, bra, ket, J, run_it=True):
        self.__checkInputArguments(bra, ket, J)
        
        # TODO: Might accept an LS coupled wave functions (when got that class)
        self.J = J
        
        self._S_bra = bra.S
        self._S_ket = ket.S
        
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
        if (abs(self._L_bra - self._L_ket) > 1) :
            #or ((self._S_bra != self._S_ket) and (self._S_bra != 1)):
            
            # TODO: Remove debug
            self.details = "deltaConditionsForGlobalQN = False spin orbit {}\n {}"\
                .format(str(self.bra), str(self.ket))
            return False
        
        return True
    
    def _totalSpinTensorMatrixElement(self):
        """ <1/2 1/2 (S) | S^[1]| 1/2 1/2 (S)>, only non zero for S=S'=1 """
        if (self._S_bra != self._S_ket) or (self._S_bra == 0):
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
        
        factor = safe_racah(self._L_bra, self._L_ket, 
                            self._S_bra, self._S_ket,
                            1, self.J)
        if self.isNullValue(factor) or not self.deltaConditionsForGlobalQN():
            return 0
        
        return  factor * spin_me * self._BrodyMoshinskyTransformation()
    
    def _globalInteractionCoefficient(self):
        # no special interaction constant for the Central ME
        phase = (-1)**(self._S_bra + self._L_bra - self.J)
        factor = np.sqrt((2*self._L_bra + 1)*(2*self._L_ket + 1))
        
        return phase * factor * self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        factor = safe_racah(self._L_bra, self._L_ket, 
                            self._l, self._l_q,
                            1, self._L)
        if self.isNullValue(factor):
            return 0
        
        return factor * np.sqrt(self._l * (self._l + 1) * (2*self._l + 1))
    
    




class SpinOrbitForce_JTScheme(_TwoBodyMatrixElement_JTCoupled, SpinOrbitForce):
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    def __init__(self, bra, ket, run_it=True):
        
        _TwoBodyMatrixElement_JTCoupled.__init__(self, bra, ket, run_it=run_it)

    def _run(self):
        _TwoBodyMatrixElement_JTCoupled._run(self)
    
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin, for tensor force impose 
        S = S' = 1, return nothing to skip the bracket spin S=0
        """
        if self._S_bra == 0:
            return []
        return (1, )
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        
        OJO: Moshinski, lambda' = lambda, lambda +- 1!!! as condition
        in the C_LS
        """
        _L_min = max(0, self._L_bra - self._S_bra)
        
        return (_L_ket for _L_ket in range(_L_min, self._L_bra + self._S_bra +1))
    
    def _LScoupled_MatrixElement(self):#, L, S, _L_ket=None, _S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        return self.centerOfMassMatrixElementEvaluation()
