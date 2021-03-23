'''
Created on Mar 10, 2021

@author: Miguel
'''
import numpy as np

from helpers.Helpers import safe_racah

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled
from matrix_elements.transformations import TalmiTransformation
from helpers.Enums import CouplingSchemeEnum, CentralMEParameters, AttributeArgs

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
            
            for arg, value in kwargs.items():
                if arg in _map:
                    attr_parser = _map[arg]
                    attr_, parser_ = attr_parser
                    kwargs[arg] = parser_(kwargs[arg].get(attr_))
                elif isinstance(value, str):
                    kwargs[arg] = float(value) if '.' in value else int(value)
        
        super(CentralForce, cls).setInteractionParameters(*args, **kwargs)
    
    def _validKet_relativeAngularMomentums(self):
        """ Central interaction only allows l'==l"""
        return (self._l, )
    
    
    def deltaConditionsForGlobalQN(self):
        """ 
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        NOTE: Redundant if run from JJ -> LS recoupling
        """
        if (self._L_bra != self._L_ket):
#             or (self._S_bra != self._S_ket):
            # TODO: Remove debug
            self.details = "deltaConditionsForGlobalQN = False central {}\n {}"\
                .format(str(self.bra), str(self.ket))
            return False
        
        return True
    
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
        _TwoBodyMatrixElement_JTCoupled._run(self)
        
    def _validKetTotalSpins(self):
        """ For Central Interaction, <S |Vc| S'> != 0 only if  S=S' """
        return (self._S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ For Central Interaction, <L |Vc| L'> != 0 only if  L=L' """
        return (self._L_bra, )
    
    def _LScoupled_MatrixElement(self):#, L, S, _L_ket=None, _S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        
        return self.centerOfMassMatrixElementEvaluation()
    
    
    