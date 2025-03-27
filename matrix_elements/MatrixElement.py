'''
Created on Feb 23, 2021

@author: Miguel
'''
import numpy as np
import inspect

from helpers.WaveFunctions import QN_2body_jj_JT_Coupling, QN_2body_jj_J_Coupling,\
    QN_1body_jj, _1Body_WaveFunction, _WaveFunction
from helpers.Helpers import safe_wigner_9j
from helpers.Enums import CouplingSchemeEnum, SHO_Parameters,\
    BrinkBoekerParameters, AttributeArgs, CentralMEParameters, PotentialForms,\
    CentralGeneralizedMEParameters
from helpers.Log import XLog

class MatrixElementException(BaseException):
    pass

class _OneBodyMatrixElement:
    '''
    Abstract class for implementing a reduced matrix element.
    
    <Bra | V(r,th) [lambda, mu]| Ket>
    
    don't care M, Mt, Ml or Ms in further implementations
    '''
    
    PARAMS_FORCE = {}
    PARAMS_SHO   = {}
    
    COUPLING = None
    _BREAK_ISOSPIN = None
    
    DEBUG_MODE = False
    
    NULL_TOLERANCE = 1.e-14
    
    def __init__(self, bra, ket, run_it=True):
        
        self.bra : _WaveFunction = None
        self.ket : _WaveFunction = None
        
        raise MatrixElementException("Abstract method, implement me!")
    
    def __checkInputArguments(self, *args, **kwargs):
        raise MatrixElementException("Abstract method, implement me!")
    
    @staticmethod
    def _automaticParseInteractionParameters(map_, kwargs_dict):
        """ 
        perform the parsing for standard arguments, map must have the Attribute
        scheme to import from any incoming dictionary.
        i.e:
        :map_ <dict>:{k: <tuple>} (internal attr_ key, type_class to convert it)
        
        _map = {
            CentralMEParameters.potential : (AttributeArgs.name, str),
            CentralMEParameters.constant  : (AttributeArgs.value, float),
            CentralMEParameters.mu_length : (AttributeArgs.value, float),
            CentralMEParameters.n_power   : (AttributeArgs.value, int)
            ...
            }
        :kwargs
        """
        for arg, value in kwargs_dict.items():
            if arg in map_:
                attr_parser = map_[arg]
                attr_, parser_ = attr_parser
                kwargs_dict[arg] = parser_(kwargs_dict[arg].get(attr_))
            elif isinstance(value, str):
                kwargs_dict[arg] = float(value) if '.' in value else int(value)
            
        return kwargs_dict
    
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
    
    def __call__(self):
        return self.value
        
    
    def _run(self):
        """ 
        Calculate the final numerical value, must get a totally antisymmetrized_
        and normalized matrix element evaluation. 
        """
        raise MatrixElementException("Abstract method, implement me!")
    
    @property
    def value(self):
        """ 
        If a matrix element has not been evaluated or discarded, evaluates it. 
        """
        if not hasattr(self, '_value'):
            self._run()
        return self._value
    
    @property
    def isNullMatrixElement(self):
        """ 
        If a matrix element is null it means it does not fulfill the necessary 
        symmetry requirements or it has been evaluated with 0.0 result.
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
        return abs(value) < self.NULL_TOLERANCE
    
    @classmethod
    def getMatrixElementClassLogs(cls):
        """
        Default log of the matrix element class (individual MatrixElements class
        cannot share a common attribute to store it own details) specify in each
        class.
        """
        return f'No after-run details.  ::  <{cls.__name__}>'
        
    @classmethod
    def setMatrixElementClassLogs(cls):
        raise MatrixElementException("Abstract Method, implement me!")

class _TwoBodyMatrixElement(_OneBodyMatrixElement):
    '''
    Abstract class to be implemented according to reduced matrix element
    
    <Bra(1,2) | V(1,2) [lambda, mu]| Ket(1,2)>
    
    don't care M, Mt, Ml or Ms in further implementations
    '''
    
    PARAMS_FORCE = {}
    PARAMS_SHO   = {}
    
    COUPLING = None
    _BREAK_ISOSPIN = None
    ## Explicit antisymmetrization_ of the matrix element, <ab V cd> - <ab V dc>
    ## set False if the matrix element do it implicitly.
    EXPLICIT_ANTISYMM = True
    
    DEBUG_MODE = False
    
    NULL_TOLERANCE = 1.e-14
    
    ONEBODY_MATRIXELEMENT = None ## 1-B Interaction associate with the force.
    
    ## Abstract class, the setting methods remain abstract from 1B m.e template
    
    def getExchangedME(self):
        """ Permute single particle functions """
        raise MatrixElementException("Abstract method, implement me!")
    
    @property
    def parity_bra(self):
        if not hasattr(self, '_parity_bra'):
            self._parity_bra = (self.bra.l1 + self.bra.l2) % 2
        return self._parity_bra
    
    @property
    def parity_ket(self):
        if not hasattr(self, '_parity_ket'):
            self._parity_ket = (self.ket.l1 + self.ket.l2) % 2
        return self._parity_ket
    
    def calculate_1BME(self, bra=None, ket=None):
        """
        Calculate the 1-body component of the interaction (if present):
        
        If bra and ket given <_1Body_WaveFunction>, evaluate them for the
        associated force.
        
        If not, the matrix element will be evaluated with the first states of 
        the matrix element instance. (Raise error if the m.e. is not instanciated)
        """
        
        if self.ONEBODY_MATRIXELEMENT != None:
            return
        
        if (bra != None) and (ket != None):
            
            assert isinstance(bra, _1Body_WaveFunction), "Invalid type [{}]".format(bra.__class__)
            assert isinstance(ket, _1Body_WaveFunction), "Invalid type [{}]".format(ket.__class__)
            
            return self.ONEBODY_MATRIXELEMENT(bra, ket, run_it=True)
        else:
            
            return self.ONEBODY_MATRIXELEMENT(self.bra.sp_state_1, 
                                              self.ket.sp_state_1, run_it=True)
    
        
#===============================================================================
# ONE BODY MATRIX ELEMENT TEMPLATE
#===============================================================================

class _OneBodyMatrixElement_jjscheme(_OneBodyMatrixElement):
    
    COUPLING = CouplingSchemeEnum.JJ
    _BREAK_ISOSPIN = True
    
    def __init__(self, bra, ket, run_it=True):
        
        self.__checkInputArguments(bra, ket)
        
        self.bra = bra
        self.ket = ket
        
        if not self.isNullMatrixElement and run_it:
            # evaluate the normal and antisymmetrized me
            self._run()
    
    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_1body_jj):
            raise MatrixElementException("<bra| is not <QN_1body_jj>")
        if not isinstance(ket, QN_1body_jj):
            raise MatrixElementException("|ket> is not <QN_1body_jj>")
        
        ## i.e. Matrix elements from neutron cannot connect with proton 
        if bra.m_t != ket.m_t:
            self._value = 0.0
            self._isNullMatrixElement = True
    
    ## NOTE: If the interaction will be called from the runner, remember to call 
    ##  the parameter setting (or define it) from the 2-Body matrix element.!!!
    
#===============================================================================
# TWO BODY MATRIX ELEMENT TEMPLATE
#===============================================================================
class _TwoBodyMatrixElement_JCoupled(_TwoBodyMatrixElement):
    """
    Implementation of non antisymmetrized_ (explicitly) matrix elements particle
    labeled, states in the  jj scheme coupled to total J angular momentum
    """
    
    _BREAK_ISOSPIN = True
    COUPLING = CouplingSchemeEnum.JJ
    EXPLICIT_ANTISYMM = False
    RECOUPLES_LS      = True
    ## Set RECOUPLES_LS as False if the interaction don't require LS re_coupling
    
    SYMMETRICAL_PNPN  = True ## set to False in case pnpn!=npnp (and pnnp!=nppn)
    
    def __init__(self, bra, ket, run_it=True):
        
        self.__checkInputArguments(bra, ket)
        
        self.bra : QN_2body_jj_J_Coupling = bra
        self.ket : QN_2body_jj_J_Coupling = ket
        
        self.J = bra.J
        self.exchange_phase = None
        self.exch_2bme = None
        ## Implicit MT to distinguish pp/pn/nn
        self.MT = self.bra.MT
        
        if (bra.J != ket.J):
            print("Bra J [{}]doesn't match with ket's J [{}]".format(bra.J, ket.J))
            self._value = 0.0
        else:
            self._nullConditionForSameOrbit()
            self._nullConditionsOnParticleLabelStates()
        
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
        if ket.MT != bra.MT:
            raise MatrixElementException(f"bra-ket states change the particle label <MT={bra.MT}|{ket.MT}>")
        
        ## Wave functions do not change the number of protons or neutrons_
        if bra.isospin_3rdComponent != ket.isospin_3rdComponent:
            self._value = 0.0
            self._isNullMatrixElement = True
    
    def getExchangedME(self):
        if self.exch_2bme == None:
            _, exch_ket = self.ket.exchange()
            self.exch_2bme = self.__class__(self.bra, exch_ket, run_it=False)
        
        return self.exch_2bme
        
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
        total J must obey angular momentum coupling restriction. 
        """
        if (self.bra.nucleonsAreInThesameOrbit() or 
            self.ket.nucleonsAreInThesameOrbit()):
            if self.J % 2 == 1:
                self._value = 0.0
                self._isNullMatrixElement = True
    
    def _nullConditionsOnParticleLabelStates(self):
        """ 
        This function checks if the label states are valid to skip at __init__
        IMPLEMENTATION !! set here self.value = 0 and _isnullMatrixElement =True
        (i.e. Coulomb only on <pp | pp>)"""
        
        raise MatrixElementException("abstract method, implement me!!")
    
    def _run(self):
        """ 
        Calculates the NON antisymmetrized_ matrix element value. 
        This overwriting allow to implement the antysimmetrization in the inner
        process to avoid the explicit calculation of the whole m.e.
        """
        if self.isNullMatrixElement:
            return
    
        if self.DEBUG_MODE: 
            XLog.write('nas_me', ket=self.ket.shellStatesNotation)
    
        # antisymmetrization_ taken in the inner evaluation
        self._value = self._LS_recoupling_ME()
    
        if self.DEBUG_MODE:
            XLog.write('nas_me', value=self._value, norms=self.bra.norm()*self.ket.norm())
    
        # value is always M=0, M_T=0
        self._value *= self.bra.norm() * self.ket.norm()
    
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
            self.L_bra, self.S_bra, self.bra.J)        

        if not self.isNullValue(w9j_bra):
            recoupling = np.sqrt((self.bra.j1 + 1)*(self.bra.j2 + 1)) * w9j_bra
            if self.DEBUG_MODE:
                re1 = ((self.bra.j1 + 1)*(self.bra.j2 + 1)*(2*self.S_bra + 1)
                       *(2*self.L_bra + 1))**.5 * w9j_bra 
                XLog.write('recoup', Lb=self.L_bra, Sb=self.S_bra, re_b=re1)
            
            w9j_ket = safe_wigner_9j(
                *self.ket.getAngularSPQuantumNumbers(1, j_over2=True), 
                *self.ket.getAngularSPQuantumNumbers(2, j_over2=True),
                self.L_ket, self.S_ket, self.ket.J)
            
            if not self.isNullValue(w9j_ket):
                recoupling *= w9j_ket
                recoupling *= np.sqrt((self.ket.j1 + 1)*(self.ket.j2 + 1))
                recoupling *= np.sqrt((2*self.S_bra + 1)*(2*self.L_bra + 1)
                                      *(2*self.S_ket + 1)*(2*self.L_ket + 1))
                
                if self.DEBUG_MODE:
                    re2 = ((self.ket.j1 + 1)*(self.ket.j2 + 1)*(2*self.S_ket + 1)
                           *(2*self.L_ket + 1))**.5 * w9j_ket
                    XLog.write('recoup', Lk=self.L_ket, Sk=self.S_ket, cts=re1*re2)
                
                return (False, recoupling)
        return (True, 0.0)
    
    def _antisymmetrized_LS_element(self):
        """
        Mediator function to evaluate the direct and exchange LS m.e, 
        in the case of non explicitly antysimetrized_ m.e is just the _LScoupledMatrixElement()
        """
        return self._LScoupled_MatrixElement()
    
    def _LS_recoupling_ME(self):
    #def _non_antisymmetrized_ME(self):
        """ 
        Obtains the non antisymmetrized matrix elements by recoupling to total
        L and S and call the Inner Interaction recoupled to LS - T scheme.
        """
        sum_ = 0.
        
        L_max = self.bra.l1+self.bra.l2
        L_min = abs(self.bra.l1-self.bra.l2)
        
        for S in (0, 1):
            self.S_bra = S
            
            for S_ket in self._validKetTotalSpins():
                self.S_ket = S_ket
                
                for L in range(L_min, L_max +1):
                    self.L_bra = L
                    
                    for L_ket in self._validKetTotalAngularMomentums():
                        self.L_ket = L_ket
                        
                        null, coupling = self._angularRecouplingCoefficiens()
                        if null:
                            continue
                        
                        val = self._antisymmetrized_LS_element()
                        if self.DEBUG_MODE:
                            XLog.write('recoup', antsym_val=val)
                        
                        sum_ += coupling * val
                
        return sum_
    
class _TwoBodyMatrixElement_Antisym_JCoupled(_TwoBodyMatrixElement_JCoupled):
    
    """ 
    Overwriting of the explicit implementation of the antysymmetrization_
    """
    EXPLICIT_ANTISYMM = True
    
    def _antisymmetrized_LS_element(self):
        """
        Mediator function to evaluate the direct and exchange LS m.e, 
        This is an explicitly antysimetrized_ m.e, it evaluates the both direct
        and exchange matrix elements in the LS scheme.
        """
        direct = self._LScoupled_MatrixElement()
        
        phase_9j  = (self.ket.j1 + self.ket.j2)// 2 + self.J
        phase_9j += 1 + self.S_ket + self.ket.l1 + self.ket.l2 + self.L_ket
        
        self.exch_2bme.S_bra = self.S_bra
        self.exch_2bme.S_ket = self.S_ket
        self.exch_2bme.L_bra = self.L_bra
        self.exch_2bme.L_ket = self.L_ket
        
        exch = self.exch_2bme._LScoupled_MatrixElement()
        return direct - (((-1)**(phase_9j)) * self.exchange_phase * exch)  
    
    def _run(self):
        """ Calculate the explicit antisymmetric matrix element value. """
        if self.isNullMatrixElement:
            return
        
        # construct the exchange ket
        phase, exchanged_ket = self.ket.exchange()
        self.exchange_phase = phase
        self.exch_2bme = self.__class__(self.bra, exchanged_ket, run_it=False)
        
        if self.DEBUG_MODE: 
            XLog.write('na_me', p='DIRECT', ket=self.ket.shellStatesNotation)
        
        # value is always M=0, M_T=0
        self._value =  self._LS_recoupling_ME()
        self._value *= self.bra.norm() * self.ket.norm()
        
        if self.DEBUG_MODE: 
            XLog.write('na_me', phs=phase)
            XLog.write('nas', norms=self.bra.norm()*self.ket.norm(), value=self._value)
    
    def _nullConditionForSameOrbit(self):
        _TwoBodyMatrixElement_JCoupled._nullConditionForSameOrbit(self)
        


class _TwoBodyMatrixElement_JTCoupled(_TwoBodyMatrixElement_JCoupled):
    
    """ 
        Base normalized & antisimetrized_ two body matrix element for isospin_
    symmetric interactions. 
        Based on general J interaction (perform the explicit exchange and the 
    LS decoupling). Overwrites the constructor for the T arguments and the JT 
    odd condition for the same orbit states.
    """
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    _BREAK_ISOSPIN    = False
    EXPLICIT_ANTISYMM = False
    
    def __init__(self, bra, ket, run_it=True):
        
        self.__checkInputArguments(bra, ket)
        
        self.bra : QN_2body_jj_JT_Coupling = bra
        self.ket : QN_2body_jj_JT_Coupling = ket
        
        self.J = bra.J
        self.T = bra.T
        self.MT= bra.MT
        
        self.exchange_phase = None
        self.exch_2bme = None
        
        if (bra.J != ket.J) or (bra.T != ket.T) or (bra.MT != ket.MT):
            print("Bra JT [{}]doesn't match with ket's JT [{}]"
                  .format(bra.J, bra.T, ket.J, ket.T))
            self._value = 0.0
        else:
            self._nullConditionForSameOrbit()
        
        if not self.isNullMatrixElement and run_it:
            # evaluate the normal and antisymmetrized me
            if self.DEBUG_MODE: 
                XLog.write('nas', bra=bra.shellStatesNotation, 
                           ket=ket.shellStatesNotation, J=self.J, T=self.T)
            self._run()
        
    #---------------------------------------------------------------------------
    
    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_2body_jj_JT_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_jj_JT_Coupling>")
        if not isinstance(ket, QN_2body_jj_JT_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_jj_JT_Coupling>")
        
    def _nullConditionForSameOrbit(self):
        """ When the  two nucleons of the bra or the ket are in the same orbit,
        total J and T must obey angular momentum coupling restriction. 
        """
        if (self.bra.nucleonsAreInThesameOrbit() or 
            self.ket.nucleonsAreInThesameOrbit()):
            if (self.J + self.T)%2 != 1:
                self._value = 0.0
                self._isNullMatrixElement = True
    


class _TwoBodyMatrixElement_Antisym_JTCoupled(_TwoBodyMatrixElement_JTCoupled):
    """ 
    Overwriting of the explicit implementation of the antysymmetrization_
    """
    EXPLICIT_ANTISYMM = True
    
    def _run(self):
        """ Calculate the explicit antisymmetric matrix element value. """
        if self.isNullMatrixElement:
            return
        
        # construct the exchange ket
        phase, exchanged_ket = self.ket.exchange()
        self.exchange_phase  = phase
        self.exch_2bme = self.__class__(self.bra, exchanged_ket, run_it=False)
        
        if self.DEBUG_MODE: 
            XLog.write('na_me', p='DIRECT', ket=self.ket.shellStatesNotation)
        
        # value is always M=0, M_T=0
        self._value = self._LS_recoupling_ME()
        self._value *= self.bra.norm() * self.ket.norm()
        
        if self.DEBUG_MODE: 
            XLog.write('na_me', phs=phase)
            XLog.write('nas', norms=self.bra.norm()*self.ket.norm(), value=self._value)
    
    def _antisymmetrized_LS_element(self):
        """
        Mediator function to evaluate the direct and exchange LS m.e, 
        This is an explicitly antysimetrized_ m.e, it evaluates the both direct
        and exchange matrix elements in the LS scheme.
        """
        direct = self._LScoupled_MatrixElement()
        
        phase_9j  = (self.ket.j1 + self.ket.j2)// 2 + self.J
        phase_9j += 1 + self.S_ket + self.ket.l1 + self.ket.l2 + self.L_ket
        
        self.exch_2bme.S_bra = self.S_bra
        self.exch_2bme.S_ket = self.S_ket
        self.exch_2bme.L_bra = self.L_bra
        self.exch_2bme.L_ket = self.L_ket
        
        exch = self.exch_2bme._LScoupled_MatrixElement()
        return direct - (((-1)**(phase_9j)) * self.exchange_phase * exch)
            


class _MatrixElementReader(_TwoBodyMatrixElement):
    
    """
    Dummy matrix element for reading from file, assume JT scheme for simplicity
    and could be settled when reading to J scheme accordingly to the file.
    """
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    _BREAK_ISOSPIN = False

#===============================================================================
#  COMMON METHODS AND FUNCTIONS FOR MATRIX ELEMENTS
#===============================================================================

def _standardSetUpForCentralWithExchangeOps(cls, 
                                            refresh_params=True, 
                                            generalizedCentral=False,
                                            **kwargs):
    """
    Process to set up Exchange operators and general central interactions.
    
    returns the class with the constant.
    :generalizedCentral: It sets up two sets of parameters for R and r integrals
                         Impoting is requires a different Enum for the setUp
    
    Protocol:
        0. Reset both sho parameters and force parameters of the class.
        1. set up b length.
        2. set up Force constants:
            2.1. Parts for the radial potential (n_power, mu_length, potential)
            2.2. Constants for the Exchange Operators
                Wigner(n.a.), Barlett(spin), Heisenberg(isospin), Majorana(spatial)
            2.3 set CentralMEParameters.constant = 1.0 for Moshinsky internal 
                transformation, in case of Value given in the kwargs, warning and
                assign to Wigner value if no BrinkBoekerParameters where given,
                ** ommit it otherwise.
    """
    
    
    # Refresh the Force parameters
    if cls.PARAMS_FORCE and refresh_params:
        cls.PARAMS_FORCE = {}
    
    _b = SHO_Parameters.b_length
    cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
        
    ## Exchange forms, if not pressent, all will be 0.0
    for param in BrinkBoekerParameters.members():
        exch_param = kwargs.get(param, {})
        cls.PARAMS_FORCE[param] = float(exch_param.get(AttributeArgs.value, 0.0))
    for param in filter(lambda x: x.startswith('opt_'), 
                        CentralMEParameters.members()):
        if not param in kwargs: continue
        cls.PARAMS_FORCE[param] = float(kwargs.get(param)[AttributeArgs.value])
    
    assert CentralMEParameters.potential in kwargs, "Argument [potential] is required."
    potential_ = kwargs[CentralMEParameters.potential][AttributeArgs.name]
    cls.PARAMS_FORCE[CentralMEParameters.potential] = potential_.lower()
    cls.PARAMS_FORCE[CentralMEParameters.constant]  = 1.0
    
    if CentralMEParameters.n_power in kwargs:
        n_pow = kwargs[CentralMEParameters.n_power][AttributeArgs.value]
        cls.PARAMS_FORCE[CentralMEParameters.n_power] = int(n_pow)
        
    ## Particular case for Wood-Saxon, A dependent
    if potential_ == PotentialForms.Wood_Saxon:
        A = float(kwargs.get(SHO_Parameters.A_Mass, 1))
        cls.PARAMS_FORCE[CentralMEParameters.opt_mu_2] *= (A**(1/3))
        
    if (CentralMEParameters.constant in kwargs.keys()):
        # print("[WARNING] Constant argument is not accepted for this matrix element, ",
        #       "if no permutation-operators was given, the constant value ",
        #       "will be assign to the Wigner term.")
        exchange_constants = (
            abs(cls.PARAMS_FORCE.get(BrinkBoekerParameters.Wigner)),
            abs(cls.PARAMS_FORCE.get(BrinkBoekerParameters.Bartlett)),
            abs(cls.PARAMS_FORCE.get(BrinkBoekerParameters.Heisenberg)),
            abs(cls.PARAMS_FORCE.get(BrinkBoekerParameters.Majorana)),
        )
        if sum(exchange_constants) > 1.0e-6:
            print("  ** Constants assigned, omitting given constant "
                  "(absolute value) W,B,H,M :", *exchange_constants )
        else:
            # print("  ** Constants not assigned, constant -> Wigner")
            value_ = float(kwargs[CentralMEParameters.constant][AttributeArgs.value])
            cls.PARAMS_FORCE[CentralMEParameters.constant] = value_
            cls.PARAMS_FORCE[BrinkBoekerParameters.Wigner] = 1.0
    #cls.plotRadialPotential()
    if generalizedCentral:
        cls = _standardSetUpForCentralGeneralizedWithExchangeOps(cls, **kwargs)
    
    return cls


def _standardSetUpForCentralGeneralizedWithExchangeOps(cls, **kwargs):
    """
    Process to set up Exchange operators and general central interactions.
    
    returns the class with the constant.
    
    Protocol:
        1. set up Force constants:
            2.1. Parts for the radial potential (n_power, mu_length, potential)
            2.2. Constants for the Exchange Operators ignored (set it in relative COM)
            2.3 set CentralMEParameters.constant = 1.0 if not 
    """
    
    _b = SHO_Parameters.b_length
    if not _b in cls.PARAMS_SHO:
        cls.PARAMS_SHO[_b] = float(kwargs.get(_b))
    
    GENMEparm = CentralGeneralizedMEParameters
    
    assert CentralMEParameters.potential in kwargs, "Argument [potential] is required."
    potential_ = kwargs[GENMEparm.potential_R][AttributeArgs.name]
    cls.PARAMS_FORCE[GENMEparm.potential_R] = potential_.lower()
    
    constant_  = kwargs.get(GENMEparm.constant_R)
    if not constant_:
        cls.PARAMS_FORCE[GENMEparm.constant_R] = 1
    else:
        cls.PARAMS_FORCE[GENMEparm.constant_R] = float(constant_[AttributeArgs.value])
    
    mu_val = kwargs[GENMEparm.mu_length_R][AttributeArgs.value]
    cls.PARAMS_FORCE[GENMEparm.mu_length_R] = float(mu_val)
    
    if GENMEparm.n_power_R in kwargs:
        n_pow = kwargs[GENMEparm.n_power_R][AttributeArgs.value]
        cls.PARAMS_FORCE[GENMEparm.n_power_R] = int(n_pow)
    
    return cls
