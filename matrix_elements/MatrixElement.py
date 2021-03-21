'''
Created on Feb 23, 2021

@author: Miguel
'''
import numpy as np
# from sympy.physics.wigner import wigner_9j

import matrix_elements.BM_brackets as bmb
from matrix_elements.BM_brackets import fact

from helpers.WaveFunctions import QN_2body_jj_JT_Coupling
from helpers.Helpers import safe_wigner_9j
from helpers.Enums import Enum
# from helpers.WaveFunctions import 

class MatrixElementException(BaseException):
    pass

class _TwoBodyMatrixElement:
    '''
    Abstract class to be implemented according to reduced (and way)
    
    <Bra(1,2) | V(1,2) [lambda, mu]| Ket(1,2)>
    '''
    
    PARAMS_FORCE = {}
    PARAMS_SHO   = {}
    
    COUPLING = None
    
    # TODO: ?? DEBUG = False
    
    NULL_TOLERANCE = 1.e-9
    
    def __init__(self, bra, ket, run_it=True):
        raise MatrixElementException("Abstract method, implement me!")
    
    def __checkInputArguments(self, *args, **kwargs):
        raise MatrixElementException("Abstract method, implement me!")
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ Implement the parameters for the interaction calculations. """
        raise MatrixElementException("Abstract method, implement me!")
    
    def permute(self):
        """ Permute single particle functions """
        raise MatrixElementException("Abstract method, implement me!")
    
    def __call__(self):
        return self.value
        
    
    def _run(self):
        """ 
        Calculate the final numerical value, must get a totally antisymmetrized
        and normalized matrix element evaluation. 
        """
        raise MatrixElementException("Abstract method, implement me!")
    
    @property
    def value(self):
        """ If a matrix element has not been evaluated or discarded, evaluates it. """
        if not hasattr(self, '_value'):
            self._run()
        return self._value
    
    @property
    def isNullMatrixElement(self):
        """ 
        If a matrix element is null, that means it does not fulfill the necessary 
        symmetry requirements or it has been evaluated with 0.0 results.
        """
        if hasattr(self, '_isNullMatrixElement'):
            return self._isNullMatrixElement
        
        if hasattr(self, '_value'):
            self._isNullMatrixElement = self.isNullValue(self._value)
        else:
            self._isNullMatrixElement = False
            
        return self._isNullMatrixElement
    
    @property
    def details(self):
        return self._details
    
    @details.setter
    def details(self, detail):
        if not hasattr(self, '_details'):
            self._details = (detail, )
        else:
            self._details = (*self._details, detail)
    
    
    def isNullValue(self, value):
        """ Method to fix cero values and stop calculations."""
        if abs(value) < self.NULL_TOLERANCE:
            return True
        return False
    
    

#===============================================================================
# 
#===============================================================================


class _TwoBodyMatrixElement_JTCoupled(_TwoBodyMatrixElement):
    
    def __init__(self, bra, ket, run_it=True):
        
        self.__checkInputArguments(bra, ket)
        
        self.bra = bra
        self.ket = ket
        
        self.J = bra.J
        self.T = bra.T
        
        if (bra.J != ket.J) or (bra.T != ket.T):
            print("Bra JT [{}]doesn't match with ket's JT [{}]"
                  .format(bra.J, bra.T, ket.J, ket.T))
            self._value = 0.0
        elif bra.T not in (0, 1):
            print("T[{}] for 2 fermion W.F is not 0 or 1".format(bra.T))
            self._value = 0.0
            # TODO: Create an Angular Condition on J for elements (also with WF checkers)
        else:
            self._oddConditionOnJTForSameOrbit()
        
        if not self.isNullMatrixElement and run_it:
            # evaluate the normal and antisymmetrized me
            self._run()
        
    #---------------------------------------------------------------------------
    
    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_2body_jj_JT_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_jj_JT_Coupling>")
        if not isinstance(ket, QN_2body_jj_JT_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_jj_JT_Coupling>")
    
    def _oddConditionOnJTForSameOrbit(self):
        """ When the  two nucleons of the bra or the ket are in the same orbit,
        total J and T must obey angular momentum coupling restriction. 
        """
        if (self.bra.nucleonsAreInThesameOrbit() or 
            self.ket.nucleonsAreInThesameOrbit()):
            if (self.J + self.T)%2 != 1:
                self._value = 0.0
                self._isNullMatrixElement = True
    
    def _run(self):
        """ Calculate the antisymmetric matrix element value. """
        if self.isNullMatrixElement:
            return
        
        # construct the exchange ket
        phase, exchanged_ket = self.ket.exchange()
#         exch_2bme = _TwoBodyMatrixElement_JTCoupled(self.bra, exchanged_ket)
        #a = self.__class__
        exch_2bme = self.__class__(self.bra, exchanged_ket, run_it=False)
        
        direct = self._non_antisymmetrized_ME()
        exchan = exch_2bme._non_antisymmetrized_ME()
        self._value =  direct - (phase * exchan)
        
        self._value *= self.bra.norm() * self.ket.norm() * (2*self.J + 1)
        
    
    def _LScoupled_MatrixElement(self):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        raise MatrixElementException("abstract method, implement LS based interaction")
    
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin, depending of the Force 
        """
        raise MatrixElementException("abstract method, implement based on the force")
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        """
        raise MatrixElementException("abstract method, implement based on the force")
    
    def _angularRecouplingCoefficiens(self):
        """ 
        :L for the Bra  <int>
        :S for the Bra  <int>
        :L for the Ket  <int>
        :S for the Ket  <int>
        
            Return
        :null_values          <bool>  To skip Moshinky transformation
        :recoupling_coeff     <float> 
        """
        
        # j attribute are defined as 2*j
        
        w9j_bra = safe_wigner_9j(
            *self.bra.getAngularSPQuantumNumbers(1, j_over2=True), 
            *self.bra.getAngularSPQuantumNumbers(2, j_over2=True),
            self._L_bra, self._S_bra, self.bra.J)        

        if not self.isNullValue(w9j_bra):
            recoupling = np.sqrt((self.bra.j1 + 1)*(self.bra.j2 + 1)) * w9j_bra
            
            w9j_ket = safe_wigner_9j(
                *self.ket.getAngularSPQuantumNumbers(1, j_over2=True), 
                *self.ket.getAngularSPQuantumNumbers(2, j_over2=True),
                self._L_ket, self._S_ket, self.ket.J)
            
            if not self.isNullValue(w9j_ket):
                recoupling *= w9j_ket
                recoupling *= np.sqrt((self.ket.j1 + 1)*(self.ket.j2 + 1))
                recoupling *= np.sqrt((2*self._S_bra + 1)*(2*self._L_bra + 1)
                                      *(2*self._S_ket + 1)*(2*self._L_ket + 1))
                
                return (False, recoupling)
        return (True, 0.0)
    
    
    def _non_antisymmetrized_ME(self):
        """ 
        Obtains the non antisymmetrized matrix elements by recoupling to total
        L and S and call the Inner Interaction recoupled to LS - T scheme.
        """
        
        sum_ = 0.
        L_max = min((self.bra.l1+self.bra.l2), (self.ket.l1+self.ket.l2))
        L_min = max(abs(self.bra.l1-self.bra.l2), abs(self.ket.l1-self.ket.l2))    
        
        for S in (0, 1):
            self._S_bra = S
            
            for S_ket in self._validKetTotalSpins():
                self._S_ket = S_ket
                
                for L in range(L_min, L_max +1):
                    self._L_bra = L
                    
                    for L_ket in self._validKetTotalAngularMomentums():
                        self._L_ket = L_ket
                        
                        null, coupling = self._angularRecouplingCoefficiens()
                        if null:
                            continue
                        
                        sum_ += coupling * self._LScoupled_MatrixElement()
        return sum_
        
    
