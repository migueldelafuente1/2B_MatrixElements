'''
Created on Feb 23, 2021

@author: Miguel
'''
import numpy as np

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled
from matrix_elements.transformations import TalmiTransformation

from helpers.Enums import BrinkBoekerParameters as bb_p, CentralMEParameters,\
    PotentialForms
from helpers.Enums import AttributeArgs
from helpers.Enums import SHO_Parameters
from helpers.integrals import talmiIntegral

class BrinkBoeker(_TwoBodyMatrixElement_JTCoupled, TalmiTransformation):
    
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Brink-Boeker interaction calculation. 
        
        
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}

        for param in SHO_Parameters.members():
            cls.PARAMS_SHO[param] = float(kwargs.get(param))
        
        part_1 = AttributeArgs.ForceArgs.Brink_Boeker.part_1
        part_2 = AttributeArgs.ForceArgs.Brink_Boeker.part_2
        
        cls.PARAMS_FORCE[0] = {}
        cls.PARAMS_FORCE[1] = {}
        
        for param in bb_p.members():
            cls.PARAMS_FORCE[0][param] = float(kwargs[param].get(part_1))
            cls.PARAMS_FORCE[1][param] = float(kwargs[param].get(part_2))
        
        cls.PARAMS_FORCE[CentralMEParameters.potential] = PotentialForms.Gaussian
        
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = ([], [])    
        
    def _validKetTotalSpins(self):
        """ For Central Interaction, <S |Vc| S'> != 0 only if  S=S' """
        return (self._S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ For Central Interaction, <L |Vc| L'> != 0 only if  L=L' """
        return (self._L_bra, )
    
    def _validKet_relativeAngularMomentums(self):
        """ Central interaction only allows l'==l"""
        return (self._l, )
    
    def _globalInteractionCoefficient(self):
        # no special interaction constant for the Central ME
        return 1
    
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        return 1
    
    def deltaConditionsForGlobalQN(self):
        """ 
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        NOTE: Redundant if run from JJ -> LS recoupling
        """
        if ((self._L_bra != self._L_ket)
            or (self._S_bra != self._S_ket)
            or (self._l != self._l_q)):
            
            # TODO: Remove debug
            self.details = "deltaConditionsForGlobalQN = False central BB {}\n {}"\
                .format(str(self.bra), str(self.ket))
            return False
        
        return True
    
    def _deltaConditionsForCOM_Iteration(self):
        
        #return True
        if (((self._S_bra + self.T + self._l) % 2 == 1) and 
            ((self._S_ket + self.T + self._l_q) % 2 == 1)):
                return True
        return False
#         if self.bra.nucleonsAreInThesameOrbit():
#             if ((self._S_bra + self.T + self._l) % 2 == 0):
#                 return False
#         if self.ket.nucleonsAreInThesameOrbit():
#             if ((self._S_ket + self.T + self._l_q) % 2 == 0):
#                 return False
#         
#         return True
    
    @classmethod
    def _calculateIntegrals(cls, n_integrals=1):
        """
        >> Overwrite to have two parts
        """
        for part in (0, 1): 
            args = [
                cls.PARAMS_FORCE.get(CentralMEParameters.potential),
                cls.PARAMS_SHO.get(SHO_Parameters.b_length),
                cls.PARAMS_FORCE[part].get(CentralMEParameters.mu_length),
                cls.PARAMS_FORCE[part].get(CentralMEParameters.n_power)
            ]
            
            for p in range(cls._integrals_p_max + 1, 
                           cls._integrals_p_max + n_integrals +1):
                
                cls._talmiIntegrals[part].append(talmiIntegral(p, *args))
                
                if part: # do not count twice
                    cls._integrals_p_max += 1
            
    
    def talmiIntegral(self):
        """ 
        >> Overwrite to have two parts
        Get or update Talmi integrals for the calculations
        """
        if self._p > self._integrals_p_max:
            self._calculateIntegrals(n_integrals = max(self.rho_bra, self.rho_ket, 1))
        return self._talmiIntegrals[self._part].__getitem__(self._p)
    
    def centerOfMassMatrixElementEvaluation(self):
        #TalmiTransformation.centerOfMassMatrixElementEvaluation(self)
        """
        Radial Brody-Moshinsky transformation, direct implementation for  
        central force.
        """
        return self._BrodyMoshinskyTransformation()
    
    
    def _LScoupled_MatrixElement(self):#, L, S, _L_ket=None, _S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        aux_sum = 0.0
        
        # Sum of gaussians and projection operators
        for i in range(2):
            self._part = i
            
            # Radial Part for Gauss Integral (L == lambda)
            radial_energy = self.centerOfMassMatrixElementEvaluation()
            
            # Exchange Part
#             _L_aux = (self._L_bra == self._L_ket) * (-1)**(self._L_bra)
#             _S_aux = (self._S_bra == self._S_ket) * (-1)**(self._S_bra)
            
            #_L_aux = (-1)**(self._L_bra)
            _S_aux = (-1)**(self._S_bra)
            _T_aux = (-1)**(self.T)
            _L_aux = -1 * _S_aux * _T_aux
            
            exchange_energy = (
                self.PARAMS_FORCE[i].get(bb_p.Wigner),
                self.PARAMS_FORCE[i].get(bb_p.Majorana)* _L_aux,
                self.PARAMS_FORCE[i].get(bb_p.Bartlett) * _S_aux,
                self.PARAMS_FORCE[i].get(bb_p.Heisenberg) * _T_aux
            )
            
            # Add up
            aux_sum += radial_energy * sum(exchange_energy)
        
        return aux_sum
            


    
    

    
    
    
    
    
    
    
    