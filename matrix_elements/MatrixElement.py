'''
Created on Feb 23, 2021

@author: Miguel
'''
import numpy as np
# from sympy.physics.wigner import wigner_9j

import matrix_elements.BM_brackets as bmb
from matrix_elements.BM_brackets import fact

from helpers.WaveFunctions import QN_2body_jj_JT_Coupling,\
    QN_2body_jj_J_Coupling
from helpers.Helpers import safe_wigner_9j
from helpers.Enums import Enum, CouplingSchemeEnum
from helpers.Log import XLog
# from helpers.WaveFunctions import 

class MatrixElementException(BaseException):
    pass

class _TwoBodyMatrixElement:
    '''
    Abstract class to be implemented according to reduced matrix element
    
    <Bra(1,2) | V(1,2) [lambda, mu]| Ket(1,2)>
    
    don't care M, Mt, Ml or Ms in further implementations
    '''
    
    PARAMS_FORCE = {}
    PARAMS_SHO   = {}
    
    COUPLING = None
    _BREAK_ISOSPIN = None
    
    DEBUG_MODE = False
    
    NULL_TOLERANCE = 1.e-10
    
    def __init__(self, bra, ket, run_it=True):
        raise MatrixElementException("Abstract method, implement me!")
    
    def __checkInputArguments(self, *args, **kwargs):
        raise MatrixElementException("Abstract method, implement me!")
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ Implement the parameters for the interaction calculations. """
        raise MatrixElementException("Abstract method, implement me!")
    
    @classmethod
    def resetInteractionParameters(cls, also_SHO=False):
        """ Reset All interaction arguments to be changed """
        cls.PARAMS_FORCE = {}
        if also_SHO:
            cls.PARAMS_SHO   = {}
    
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
    def breakIsospin(self):
        return self._BREAK_ISOSPIN
    
    @property
    def details(self):
        return self._details
    
    @details.setter
    def details(self, detail):
        if not hasattr(self, '_details'):
            self._details = (detail, )
        else:
            self._details = (*self._details, detail)
    
    @classmethod
    def turnDebugMode(cls, on=True):
        if on:
            cls.DEBUG_MODE = True
        else:
            cls.DEBUG_MODE = False
        XLog.resetLog()
        XLog('root')
    
    def saveXLog(self, title=None):
        #raise MatrixElementException("define the string for the log xml file (cannot have ':/\\.' etc)")
        if title == None:
            title = 'me'
        
        XLog.getLog("{}_({}_{}_{}).xml"
                    .format(title, 
                            self.bra.shellStatesNotation.replace('/2', '.2'),
                            self.__class__.__name__,
                            self.ket.shellStatesNotation.replace('/2', '.2')))
    
    def isNullValue(self, value):
        """ Method to fix cero values and stop calculations."""
        if abs(value) < self.NULL_TOLERANCE:
            return True
        return False
    
    

#===============================================================================
# 
#===============================================================================
class _TwoBodyMatrixElement_JCoupled(_TwoBodyMatrixElement):
    
    _BREAK_ISOSPIN = True
    COUPLING = CouplingSchemeEnum.JJ
    
    def __init__(self, bra, ket, run_it=True):
        
        self.__checkInputArguments(bra, ket)
        
        self.bra = bra
        self.ket = ket
        
        self.J = bra.J
        
        if (bra.J != ket.J):
            print("Bra J [{}]doesn't match with ket's J [{}]"
                  .format(bra.J, ket.J))
            self._value = 0.0
        else:
            self._nullConditionForSameOrbit()
        
        if not self.isNullMatrixElement and run_it:
            # evaluate the normal and antisymmetrized me
            if self.DEBUG_MODE: 
                XLog.write('nas', 
                           bra=bra.shellStatesNotation, ket=ket.shellStatesNotation)
            self._run()

    
    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_2body_jj_J_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_jj_J_Coupling>")
        if not isinstance(ket, QN_2body_jj_J_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_jj_J_Coupling>")
        
        ## Wave functions do not change the number of protons or neutrons_
        if bra.isospin_3rdComponent != ket.isospin_3rdComponent:
            self._value = 0.0
            self._isNullMatrixElement = True
        
    def saveXLog(self, title=None):
        if title == None:
            title = 'me'
        auxT_ = 'T{}'.format(self.T) if hasattr(self, 'T') else ''
    
        XLog.getLog("{}_({}_{}_{})J{}{}.xml"
                    .format(title, 
                            self.bra.shellStatesNotation.replace('/2', '.2'),
                            self.__class__.__name__,
                            self.ket.shellStatesNotation.replace('/2', '.2'),
                            self.J, auxT_))
        
    def _nullConditionForSameOrbit(self):
        """ When the  two nucleons of the bra or the ket are in the same orbit,
        total J and T must obey angular momentum coupling restriction. 
        """
        if (self.bra.nucleonsAreInThesameOrbit() or 
            self.ket.nucleonsAreInThesameOrbit()):
            if self.J % 2 == 1:
                self._value = 0.0
                self._isNullMatrixElement = True
    
    
    def _run(self):
        """ Calculate the antisymmetric matrix element value. """
        if self.isNullMatrixElement:
            return
        
        # construct the exchange ket
        phase, exchanged_ket = self.ket.exchange()
        exch_2bme = self.__class__(self.bra, exchanged_ket, run_it=False)
        
        if self.DEBUG_MODE: 
            XLog.write('na_me', p='DIRECT', ket=self.ket.shellStatesNotation)
        
        direct = self._non_antisymmetrized_ME()
        
        if self.DEBUG_MODE:
            XLog.write('na_me', value=direct)
            XLog.write('na_me', p='EXCHANGED', ket=exchanged_ket.shellStatesNotation)
            
        exchan = exch_2bme._non_antisymmetrized_ME()
        
        self._value =  direct - (phase * exchan)
        # value is always M=0, M_T=0
        self._value *= self.bra.norm() * self.ket.norm()
        
        if self.DEBUG_MODE: 
            XLog.write('na_me', value=exchan, phs=phase)
            XLog.write('nas', norms=(self.bra.norm(), self.ket.norm()), value=self._value)
    
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
        
        # j attribute is defined as 2*j
        
        w9j_bra = safe_wigner_9j(
            *self.bra.getAngularSPQuantumNumbers(1, j_over2=True), 
            *self.bra.getAngularSPQuantumNumbers(2, j_over2=True),
            self._L_bra, self._S_bra, self.bra.J)        

        if not self.isNullValue(w9j_bra):
            recoupling = np.sqrt((self.bra.j1 + 1)*(self.bra.j2 + 1)) * w9j_bra
            if self.DEBUG_MODE:
                re1 = ((self.bra.j1 + 1)*(self.bra.j2 + 1)*(2*self._S_bra + 1)
                       *(2*self._L_bra + 1))**.5 * w9j_bra 
                XLog.write('recoup', Lb=self._L_bra, Sb=self._S_bra, val_b=re1)
            
            w9j_ket = safe_wigner_9j(
                *self.ket.getAngularSPQuantumNumbers(1, j_over2=True), 
                *self.ket.getAngularSPQuantumNumbers(2, j_over2=True),
                self._L_ket, self._S_ket, self.ket.J)
            
            if not self.isNullValue(w9j_ket):
                recoupling *= w9j_ket
                recoupling *= np.sqrt((self.ket.j1 + 1)*(self.ket.j2 + 1))
                recoupling *= np.sqrt((2*self._S_bra + 1)*(2*self._L_bra + 1)
                                      *(2*self._S_ket + 1)*(2*self._L_ket + 1))
                
                if self.DEBUG_MODE:
                    re2 = ((self.ket.j1 + 1)*(self.ket.j2 + 1)*(2*self._S_ket + 1)
                           *(2*self._L_ket + 1))**.5 * w9j_ket
                    XLog.write('recoup', Lk=self._L_ket, Sk=self._S_ket, val_k=re2)
                
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
        
    
class _TwoBodyMatrixElement_JTCoupled(_TwoBodyMatrixElement_JCoupled):
    
    """ 
        Base normalized & antisimetrized_ two body matrix element for isospin_
    symmetric interactions. 
        Based on general J interaction (perform the explicit exchange and the 
    LS decoupling). Overwrites the constructor for the T arguments and the JT 
    odd condition for the same orbit states.
    """
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    _BREAK_ISOSPIN = False
    
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
        else:
            self._nullConditionForSameOrbit()
        
        if not self.isNullMatrixElement and run_it:
            # evaluate the normal and antisymmetrized me
            if self.DEBUG_MODE: XLog.write('nas', 
                                           bra=bra.shellStatesNotation, 
                                           ket=ket.shellStatesNotation)
            self._run()
        
    #---------------------------------------------------------------------------
    
    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_2body_jj_JT_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_jj_JT_Coupling>")
        if not isinstance(ket, QN_2body_jj_JT_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_jj_JT_Coupling>")
    
    # def saveXLog(self, title=None):
    #     if title == None:
    #         title = 'me'
    #
    #     XLog.getLog("{}_({}_{}_{})J{}T{}.xml"
    #                 .format(title, 
    #                         self.bra.shellStatesNotation.replace('/2', '.2'),
    #                         self.__class__.__name__,
    #                         self.ket.shellStatesNotation.replace('/2', '.2'),
    #                         self.J, self.T))
        
    def _nullConditionForSameOrbit(self):
        """ When the  two nucleons of the bra or the ket are in the same orbit,
        total J and T must obey angular momentum coupling restriction. 
        """
        if (self.bra.nucleonsAreInThesameOrbit() or 
            self.ket.nucleonsAreInThesameOrbit()):
            if (self.J + self.T)%2 != 1:
                self._value = 0.0
                self._isNullMatrixElement = True
    
    
    
