'''
Created on Feb 24, 2021

@author: Miguel
'''
import numpy as np
from helpers.Enums import CouplingSchemeEnum
from helpers.Helpers import shellSHO_Notation

class WaveFunctionException(BaseException):
    pass

class _1Body_WaveFunction:
    
    """
    General wave function for 1 body, must be define the particle label as
    attribute [m_e]"""
    
    _particleLabels = {
        -1   : 'p',
        0    : '',
        1    : 'n'}
    
    def __init__(self, *args):
        raise WaveFunctionException("Abstract method, implement me!")
    
    def __checkQNArguments(self, *args):
        raise WaveFunctionException("Abstract method, implement me!")
    
    def _checkProtonNeutronLabel(self, m_t):
        assert m_t in (1,0,-1), AttributeError("m_t label for the state must be"
            " 1 (+1/2 proton), -1 (-1/2 neutron) or 0 (undefined, might raise "
            "error if the matrix element is labeled)")
    
    @property
    def particleLabel(self):
        return self._particleLabels[self.m_t]

#===============================================================================
#     ONE BODY WAVE FUCNTIONS
#===============================================================================

class QN_1body_radial(_1Body_WaveFunction):
    """
    :N     <int> principal quantum number >=0
    :L     <int> orbital angular number >=0
    :m_l   <int> third component of L
    
    One body wave function for Radial SHO wave function.
        m_l = 0 unless specification.
        
        |n, l, m_l>
    """
    def __init__(self, n, l, m_l=0, mt=0):
        
        self.__checkQNArguments(n, l, m_l, mt)
        
        self.n = n
        self.l = l
        self.m_l = m_l
        
        self.m_t = mt
    
    def __checkQNArguments(self, n, l, m, m_t):
        
        _types = [isinstance(arg, int) for arg in (n, l, m)]
        assert not False in _types, AttributeError("Invalid argument types given"
            ": [(n,l,m)={}]. All parameters must be integers".format((n, l, m)))
        
        _sign  = [n >= 0, l >= 0] 
        assert not False in _sign, AttributeError("Negative argument/s given:"
                                                   " [(n,l)={}].".format((n, l)))
        
        assert abs(m) <= l, AttributeError("3rd component cannot exceed L number")
        self._checkProtonNeutronLabel(m_t)
        
    def __str__(self):
        lab_ = self._particleLabels[self.m_t]
        if self.m_l == 0:
            return "(n:{},l:{}){}".format(self.n, self.l, lab_)
        return "(n:{},l:{},m:{}){}".format(self.n, self.l, self.m_l, lab_)
    
    @property
    def shellState(self):
        return shellSHO_Notation(self.n, self.l)

class QN_1body_jj(_1Body_WaveFunction):
    """
    :n     <int> principal quantum number >=0
    :l     <int> orbital angular number >=0
    :j     <int> total angular momentum (its assumed to be half-integer)
    :m     <int> third component of j (its assumed to be half-integer)
    
    One body jj coupled wave function for nucleon (fermion) for SHO Radial 
    wave functions.
        m = 0 unless specification.
        
        |n, l, s=1/2, j, m>
    """
    def __init__(self, n, l, j, m=0, mt=0):
        
        self.__checkQNArguments(n, l, j, m, mt)
        
        self.n = n
        self.l = l
        self.j = j
        self.m = m
        
        self.m_t = mt
    
    def __checkQNArguments(self, n, l, j, m, m_t):
        
        _types = [isinstance(arg, int) for arg in (n, l, j, m)]
        assert not False in _types, AttributeError("Invalid argument types given"
            ": [(n,l,j,m)={}]. All parameters must be integers (also for j, m)"
            .format((n, l, j, m)))
        
        _sign  = [n >= 0, l >= 0, j > 0] # j = l + 1/2 so must be at least 1/2
        assert not False in _sign, AttributeError("Negative argument/s given:"
            " [(n,l,j>0)={}].".format((n, l, j)))
        
        assert abs(m) <= j, AttributeError("3rd component cannot exceed L number")
        
        assert j in (2*l + 1, 2*l - 1), AttributeError(
            "j[{}] given is invalid with l[{}] + 1/2".format(j, l))
        self._checkProtonNeutronLabel(m_t)
    
    @property
    def s(self):
        return 1
    
    @property
    def AntoineStrIndex(self):
        """ Classic Antoine_ shell model index for l<10 """
        aux = str(1000*self.n + 100*self.l + self.j)
        if aux == '1':
            return '001'
        return aux
    
    @property
    def AntoineStrIndex_l_greatThan10(self):
        """ 
        Same property but for matrix elements that involve s.h.o states with
        l > 10 (n=1 =10000)
        """
        aux = str(10000*self.n + 100*self.l + self.j)
        if aux == '1':
            return '001'
        return aux
    
    @property
    def shellState(self):
        return shellSHO_Notation(self.n, self.l, self.j)
    
    @property
    def get_nl(self):
        return self.n, self.l

    def __str__(self):
        lab_ = self._particleLabels[self.m_t]
        return "(n:{},l:{},j:{}/2){}".format(self.n, self.l, self.j, lab_)
    
    
            
        

#===============================================================================
#  TWO BODY WAVE FUNCTIONS
#===============================================================================

# TODO: define an addition operation for wave functions __add__():
# |j1 m1> (+) |j2, m2> = [|j1+j2, m>, ... |abs(j2-j1), m>] 

class _WaveFunction:
    
    COUPLING = None
    _IGNORE_LABEL = True
    
    def __init__(self, *args):
        raise WaveFunctionException("Abstract method, implement me!")
    
    def __checkQNArguments(self, *args):
        raise WaveFunctionException("Abstract method, implement me!")

# TODO: Must check if the quantum numbers match (j1,j2=J) (T=0,1)

class _ParticleWaveFunction(_WaveFunction):
    
    """ Wave functions that are not in the isospin_ must be labeled as proton or
    neutron_ states, in two body w.functions, we cannot operate without knowing 
    if the particles are identical or not.
    
    _ParticleWaveFunctions are specially useful to treat isospin_ non-symmetric 
    interactions (like electrostatic m.e.)
    """
    _IGNORE_LABEL = False
    
    @classmethod
    def ignoreParticleLabels(cls, ignore=False):
        # TODO: Option might be problematic, it will skip checking it until 
        # using this setter again.
        
        cls._IGNORE_LABEL = ignore
    
    def _checkLabeledState(self, sp_1, sp_2):
        if self._IGNORE_LABEL:
            return
        assert sp_1.m_t in (1, -1), AttributeError("Sp state 1 not labeled as p or n")
        assert sp_2.m_t in (1, -1), AttributeError("Sp state 2 not labeled as p or n")
        
    @property
    def identicalParticles(self):
        """ 
        Return True if the particles are identical (pp or nn), False for pn 
        """
        if not hasattr(self, '_identical'):
            
            self._identical = False
            if self.sp_state_1.m_t == self.sp_state_2.m_t:
                self._identical = True
            
        return self._identical
    
    @property
    def isospin_3rdComponent(self):
        """ 
        Return the isospin_ third component of the state: +1(pp), -1(nn), 0(pn)
        """
        if not hasattr(self, 'MT'):
            self.MT = (self.sp_state_1.m_t + self.sp_state_2.m_t) // 2
        return self.MT
    

class QN_2body_L_Coupling(_ParticleWaveFunction):
    
    """
    :sp_state_1    <QN_1body_radial> 
    :sp_state_2    <QN_1body_radial>
    :L             <int>
    :ML=0          <int>
    
    Quantum Numbers for a 2 body Wave Function  with coupling to
    total orbital angular momentum L:
        |QN_1body_radial(n1, l1), 
         QN_1body_radial(n2, l2), 
         (L, ML=0)>
    """
    
    COUPLING = CouplingSchemeEnum.L
    
    def __init__(self, sp_state_1, sp_state_2, L, ML=0):
        
        self.__checkQNArguments(sp_state_1, sp_state_2, L, ML)
        
        self.sp_state_1 = sp_state_1
        self.sp_state_2 = sp_state_2
        
        self.n1 = sp_state_1.n
        self.l1 = sp_state_1.l
        
        self.n2 = sp_state_2.n
        self.l2 = sp_state_2.l
        
        self.L  = L
        self.S  = 0
        
        self.ML = ML
    
    
    def __checkQNArguments(self, sp_1, sp_2, L, ML):
        
        _types =  [isinstance(arg, int) for arg in (L, ML)]
        _types += [isinstance(sp_1, QN_1body_radial), 
                   isinstance(sp_2, QN_1body_radial)]
        
        if False in _types:
            raise AttributeError("Invalid argument types given"
            ": [|sp_1>={}, |sp_1>={}, LML={},{}]. \nSingle particle w.f. must be"
            "1body SHO radial objects; L,ML must be integers."
            .format(str(sp_1), str(sp_2), L, ML))
        
        assert L >= 0, AttributeError("Negative argument/s given: L={}".format(L))
        
        self._checkLabeledState(sp_1, sp_2)
    
    def exchange(self):
        """ 
        Returns <tuple>:
            [0] Phase shift on L due exchange of the elements of the ket.
            [1] Exchanged wave function.
        """
        # CG coefficients: when permutation of two elements:
        #     <j1,m1,j2,m2|j,m>=(-)^(j1+j2-j)<j2,m2, j1,m1|j,m>
        return (
            (-1)**(self.l1+ self.l2 - (self.L)) ,
            QN_2body_L_Coupling(self.sp_state_2, self.sp_state_1, self.L, self.ML)
            )
    
    def __str__(self):
        return "[{}, {}, (L:{})]".format(self.sp_state_1.__str__(), 
                                           self.sp_state_2.__str__(), 
                                           self.L)
    @property
    def shellStatesNotation(self):
        return self.sp_state_1.shellState + ' ' + self.sp_state_2.shellState
    
class QN_2body_LS_Coupling(QN_2body_L_Coupling):
    
    """
    :sp_state_2    <QN_1body_radial> 
    :sp_state_2    <QN_1body_radial>
    :L             <int>
    :S             <int>
    
    Quantum Numbers for a 2 body Wave Function  with coupling to
    total orbital angular momentum L and total spin S:
        |QN_1body_radial(n1, l1), 
         QN_1body_radial(n2, l2), 
         (L, S)>
    """
    
    COUPLING = (CouplingSchemeEnum.L, CouplingSchemeEnum.S)
            
    def __init__(self, sp_state_1, sp_state_2, L, S):
        
        self.__checkQNArguments(sp_state_1, sp_state_2, L, S)
        
        # define the n,l, sp_states attributes
        QN_2body_L_Coupling.__init__(self, sp_state_1, sp_state_2, L)
        
        self.L  = L
        self.S  = S
    
    
    def __checkQNArguments(self, sp_1, sp_2, L, S):
        
        _types =  [isinstance(arg, int) for arg in (L, S)]
        _types += [isinstance(sp_1, QN_1body_radial), 
                   isinstance(sp_2, QN_1body_radial)]
        
        if False in _types:
            raise AttributeError("Invalid argument types given"
            ": [|sp_1>={}, |sp_1>={}, LS={},{}]. \nSingle particle w.f. must be"
            "1body SHO radial objects; L,ML must be integers."
            .format(str(sp_1), str(sp_2), L, S))
        
        assert L >= 0, AttributeError("Negative argument/s given: L={}".format(L))
        assert S >= 0, AttributeError("Negative argument/s given: S={}".format(S))
        
        self._checkLabeledState(sp_1, sp_2)
    
    def exchange(self):
        """ 
        Returns <tuple>:
            [0] Phase shift on LS due exchange of the elements of the ket.
            [1] Exchanged wave function.
        """
        # CG coefficients: when permutation of two elements:
        #     <j1,m1,j2,m2|j,m>=(-)^(j1+j2-j)<j2,m2, j1,m1|j,m>
        # +1 come from s1,s2=1/2 to S coupling
        return (
            (-1)**(1 + self.l1+ self.l2 - (self.L + self.S)) ,
            QN_2body_LS_Coupling(self.sp_state_2, self.sp_state_1,
                                 self.L, self.S)
            )
    
    def __str__(self):
        return "[{}, {}, (L:{}S:{})]".format(self.sp_state_1.__str__(), 
                                           self.sp_state_2.__str__(), 
                                           self.L, self.S)
    
class QN_2body_jj_JT_Coupling(_WaveFunction):
    
    """ 
    :sp_state_jj_1 <QN_1body_jj> 
    :sp_state_jj_2 <QN_1body_jj>
    :J             <int>
    :T             <int>
    :M = 0         <int>
    :MT = 0        <int>
    
    Quantum Numbers for a 2 body Wave Function in jj coupling with Coupling to
    total angular momentum J and total isospin T:
        M_J = 0, M_T = 0
        |QN_1body_jj(n1, l1, s1=1/2, j1) QN_1body_jj(n2, l2, s1=1/2, j2), J, T>
    """
    
    COUPLING = (CouplingSchemeEnum.JJ,
                CouplingSchemeEnum.T)
    
    # j1 and j2 are fractions and they are defined as 2*j
    def __init__(self, sp_state_jj_1, sp_state_jj_2, J,T, M=0, MT=0):
        
        self._checkQNArguments(sp_state_jj_1, sp_state_jj_2, J,T, M, MT)
        
        self.sp_state_1 = sp_state_jj_1
        self.sp_state_2 = sp_state_jj_2
        
        self.n1 = sp_state_jj_1.n
        self.l1 = sp_state_jj_1.l
        self.j1 = sp_state_jj_1.j
        
        self.n2 = sp_state_jj_2.n
        self.l2 = sp_state_jj_2.l
        self.j2 = sp_state_jj_2.j
        
        self.J  = J
        self.T  = T
        
        self.M  = M
        self.MT = MT
    
    def _checkQNArguments(self, sp_1, sp_2, J,T, M, MT):
        ## protected to overwrite in the J scheme
        _types =  [isinstance(arg, int) for arg in (J,T, M, MT)]
        _types += [isinstance(arg, QN_1body_jj) for arg in (sp_1, sp_2)]
        if False in _types:
            raise AttributeError("Invalid argument types given"
            ": [|sp_1>={}, |sp_1>={}, JTMMt={}]. \nSingle particle w.f. must be"
            "1body jj objects, JT, M,Mt must be integers."
            .format(str(sp_1), str(sp_2), (J,T, M, MT)))
        
        assert (abs(sp_1.j - sp_2.j) <= 2*J) and ((sp_1.j + sp_2.j) >= 2*J), \
            AttributeError("Single particle j1, j2 states don't sum J")
        assert abs(M)  <= J, AttributeError("M (J) must be <= J")
        assert T in (0, 1), AttributeError("T must be 1 or 0")
        assert abs(MT) <= T, AttributeError("MT must be <= T")
        
        ## Not necessary check m_t labels for the single particle states and total T
        if T == 1:
            if abs(sp_1.m_t + sp_2.m_t) != 2 or (sp_1.m_t + sp_2.m_t)//2 == MT:
                pass ## invalid, pp or nn state without T=1,MT valid component
        
        _sign  = [J >= 0, T >= 0]
        assert not False in _sign, AttributeError("Negative argument/s given: "
                                                  "[(J,T)={}].".format((J, T)))
        
         
    def norm(self):
        """ Norm of a 2 body antisymmetric wave function """
        #return np.sqrt(1 - delta*((-1)**(self.T + self.J))) / (1 + delta)
        delta = 0
        if self.nucleonsAreInThesameOrbit():
            if (self.J + self.T)%2 == 0:
                return 0
            delta = 1
        
        return 1 /np.sqrt(1 + delta)
    
    def exchange(self):
        """ 
        Returns <tuple>:
            [0] Phase change on JT due exchange of the elements.
            [1] Exchanged wave function.
        """
        # CG coefficients: when permutation of two elements:
        #     <j1,m1,j2,m2|j,m>=(-)^(j1+j2-j)<j2,m2, j1,m1|j,m>
        # +1 comes from t1,t2 to T coupling
        return (
            (-1)**(1 + (self.j1+ self.j2)//2 - (self.J + self.T)) ,
            QN_2body_jj_JT_Coupling(self.sp_state_2, self.sp_state_1,
                                    self.J, self.T, self.M, self.MT)
            )
    
    
    def nucleonsAreInThesameOrbit(self):
        """ property to assert J+T = odd coupling condition when |n(1)n(2)>. """
        
        if (self.j1==self.j2) and (self.l1==self.l2) and (self.n1==self.n2):
            return True
        return False
    
    def getSPQuantumNumbers(self, particle, j_over2=False):
        """ Get (n, l, j) tuple for the 'particle' state
        :particle <int> = 1, 2.  Which particle single states.
        :j_over2  <bool>         Get the value of j over 2 (True) or as 2*j (False)
        """
        #assert particle in (1, 2), WaveFunctionException("particle must be 1 or 2")
        aux = 1 if j_over2 else 0.5
        
        return (getattr(self, 'n{}'.format(particle)),
                getattr(self, 'l{}'.format(particle)),
                aux * getattr(self, 'j{}'.format(particle)))
    
    def getAngularSPQuantumNumbers(self, particle, j_over2=False):
        """ Get angular quantum numbers (l, s, j) tuple for the 'particle' state
        :particle <int> = 1, 2.  Which particle single states.
        :j_over2  <bool>         Get the value of j over 2 (True) or as 2*j (False)
        """
        #assert particle in (1, 2), WaveFunctionException("particle must be 1 or 2")
        #aux = 1 if j_over2 else 0.5
        aux = 0.5 if j_over2 else 1
        
        return (getattr(self, 'l{}'.format(particle)),
                aux * 1, 
                aux * getattr(self, 'j{}'.format(particle)))
    
    
    
    def __str__(self):
        return "[{}, {},(J:{}T:{})]".format(self.sp_state_1.shellState, 
                                            self.sp_state_2.shellState, 
                                            self.J, self.T)
        # return "[{}, {},(J:{}T:{})]".format(self.sp_state_1.__str__(), 
        #                                      self.sp_state_2.__str__(), 
        #                                      self.J, self.T)
    
    @property
    def shellStatesNotation(self):
        return self.sp_state_1.shellState + ' ' + self.sp_state_2.shellState
    
    def getRadialQuantumNumbers(self):
        return (QN_1body_radial(*self.sp_state_1.get_nl), 
                QN_1body_radial(*self.sp_state_2.get_nl))
    
    def __eq__(self, other, also_3rd_components=False):
        """ 
        Compare this wave function with other, true if quantum numbers match
        """
        if other.__class__ != self.__class__:
            raise WaveFunctionException("Cannot compare {} with this object {}, "
                "only same objects".format(other.__class__, self.__class__))
        if self.n1 == other.n1 and self.n2 == other.n2:
            if self.l1 == other.l1 and self.l2 == other.l2:
                if self.j1 == other.j1 and self.j2 == other.j2:
                    if also_3rd_components:
                        if self.M == other.M and self.MT == other.MT:
                            return True
                    else:
                        return True
                    
        return False

class QN_2body_jj_J_Coupling(_ParticleWaveFunction, QN_2body_jj_JT_Coupling):
    
    """ 
    Scheme without the Isospin Formalism (particle labeled wave functions)
    
    :sp_state_jj_1 <QN_1body_jj> 
    :sp_state_jj_2 <QN_1body_jj>
    :J             <int>
    :M = 0         <int>
    
    Quantum Numbers for a 2 body Wave Function in jj coupling with Coupling to
    total angular momentum J and total isospin T:
        M_J = 0, M_T = 0
        |QN_1body_jj(n1, l1, s1=1/2, j1) QN_1body_jj(n2, l2, s1=1/2, j2), J, T>
    """
    
    COUPLING = CouplingSchemeEnum.JJ
    
    # j1 and j2 are fractions and they are defined as 2*j
    def __init__(self, sp_state_jj_1, sp_state_jj_2, J, M=0):
        
        self._checkQNArguments(sp_state_jj_1, sp_state_jj_2, J, M)
        
        self.sp_state_1 = sp_state_jj_1
        self.sp_state_2 = sp_state_jj_2
        
        self.n1 = sp_state_jj_1.n
        self.l1 = sp_state_jj_1.l
        self.j1 = sp_state_jj_1.j
        
        self.n2 = sp_state_jj_2.n
        self.l2 = sp_state_jj_2.l
        self.j2 = sp_state_jj_2.j
        
        self.J  = J        
        self.M  = M
    
    def _checkQNArguments(self, sp_1, sp_2, J, M):
        
        self._checkLabeledState(sp_1, sp_2)
        #QN_2body_jj_JT_Coupling._checkQNArguments(sp_1, sp_2, J, 0, M, 0)
        super(QN_2body_jj_J_Coupling, self)._checkQNArguments(sp_1, sp_2, J, 0, M, 0)
         
    def norm(self):
        """ Norm of a 2 body antisymmetric wave function """
        delta = 0
        if self.nucleonsAreInThesameOrbit():
            if self.J % 2 == 1:
                return 0
            delta = 1
        
        return 1 /np.sqrt(1 + delta)
    
    def exchange(self):
        """ 
        Returns <tuple>:
            [0] Phase change on JT due exchange of the elements.
            [1] Exchanged wave function.
        """
        return (
            (-1)**((self.j1+ self.j2)//2 - self.J) ,
            QN_2body_jj_J_Coupling(self.sp_state_2, self.sp_state_1, self.J, self.M)
            )
    
    
    def nucleonsAreInThesameOrbit(self):
        """ property to assert J+T = odd coupling condition when |n(1)n(2)>. """
        if not self.identicalParticles:
            return False
        
        if (self.j1==self.j2) and (self.l1==self.l2) and (self.n1==self.n2):
            return True
        return False
    
    def __str__(self):
        lab_ = "{}{}".format(self.sp_state_1.particleLabel, 
                             self.sp_state_2.particleLabel)
        return "[{}, {},(J:{}){}]".format(self.sp_state_1.shellState, 
                                            self.sp_state_2.shellState, 
                                            self.J, lab_)
        
        # return "[{}, {},(J:{})]".format(self.sp_state_1.__str__(), 
        #                                      self.sp_state_2.__str__(), 
        #                                      self.J, self.T)
    
    def __eq__(self, other, also_3rd_components=False):
        """ 
        Compare this wave function with other, true if quantum numbers match
        """
        if other.__class__ != self.__class__:
            raise WaveFunctionException("Cannot compare {} with this object {}, "
                "only same objects".format(other.__class__, self.__class__))
        if not self.identicalParticles():
            return False
        
        if self.n1 == other.n1 and self.n2 == other.n2:
            if self.l1 == other.l1 and self.l2 == other.l2:
                if self.j1 == other.j1 and self.j2 == other.j2:
                    if also_3rd_components:
                        if self.M == other.M:
                            return True
                    else:
                        return True
                    
        return False
    
    

#==============================================================================
# HELPERS with W.F
#==============================================================================
def getMinMaxAngularMomentum(j1, j2, j3, j4):
    
    if isinstance(j1, int):
        Jmin = max(abs(j1 - j2), abs(j3 - j4))
        Jmax = min(j1 + j2, j3 + j4)
    elif isinstance(j1, QN_1body_jj):
        Jmin = max(abs(j1.j - j2.j), abs(j3.j - j3.j))
        Jmax = min(j1.j + j2.j, j3.j + j4.j)
    elif isinstance(j1, QN_1body_radial):
        Jmin = max(abs(j1.j - j2.j), abs(j3.j - j3.j))
        Jmax = min(j1.j + j2.j, j3.j + j4.j)
    else:
        raise BaseException("Invalid type for angular momentum")
    return Jmin, Jmax