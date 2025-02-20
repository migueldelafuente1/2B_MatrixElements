'''
Created on 4 feb 2025

@author: delafuente

Multi-evaluated interactions evaluates the Talmi method for very expensive 
matrix elements, specially the different components between Tz channels and 
interaction mechanisms in the first-principles interactions. Such as Argone-18

The new Moshinsky expansion requires the generalization of some "final" methods

    _LScoupled_MatrixElement:
        centerOfMassMatrixElement
            _deltaConditionsGlobalQN
            _totalSpinTensorMatrixElements (for )
            _BrodyMoshinskyTransformation:
                (XXX) _talmiIntegral (moved to the end cause it is not shared for 
                                    all the integrals)
                _interactionSeries:
                    (XXX) _valid_ket_relativeAngularMomentum
                    (XXX) _deltaconditionsForCOMIteration
                    (XXX) _interactionConstantsforCOM_iteration
                    
                    ** introduce term for individual term-integral evaluation
                        (overwrite explicitly)::
                        matrixElementCompoundEvaluation()
                        
'''
import numpy as np

from matrix_elements.transformations import TalmiMultiInteractionTransformation
from matrix_elements.MatrixElement import MatrixElementException,\
    _TwoBodyMatrixElement_JTCoupled, _TwoBodyMatrixElement_JCoupled
from helpers.Enums import SHO_Parameters, CentralMEParameters, Enum,\
    PotentialForms, CouplingSchemeEnum
from helpers.Helpers import ConstantsV18, safe_wigner_6j, safe_3j_symbols,\
    polynomialProduct, getGeneralizedLaguerreRootsWeights, gamma_half_int,\
    Constants, safe_clebsch_gordan, _LINE_2, prettyPrintDictionary
from helpers.integrals import talmiIntegral

class _BaseTalmiMultinteraction_JScheme(_TwoBodyMatrixElement_JCoupled,
                                        TalmiMultiInteractionTransformation):
    '''
    Abstract implementation of the methods for the evaluation of the center of 
    mass and angular integrals to be defined for the argone interaction
    '''
    COUPLING = CouplingSchemeEnum.JJ
    _BREAK_ISOSPIN    = True
    EXPLICIT_ANTISYMM = True
    
    _TEST_QUADRATURES = True
    
    _talmi_integrals = {} ## TODO: define
    
    _integrals_p_max = {}
    
    _tensor_ang_me = {} ## (j, l, l'): value
    _ls2_ang_me    = {} ## (j, l)    : value 
    
    def __init__(self, bra, ket, run_it=True):        
        _TwoBodyMatrixElement_JCoupled.__init__(self, bra, ket, run_it=run_it)
    
    def _deltaConditionsForCOM_Iteration(self):
        """ For the antisymmetrization_ of the wave functions. """
        if (((self.S_bra + 1 + self._l) % 2 == 1) and 
            ((self.S_ket + 1 + self._l_q) % 2 == 1)):
                return True
        return False
     
    def _validKet_relativeAngularMomentums(self):
        """ 
        Evaluate the l relative for all possible central-spinOrbit-tensor,
        with maximum delta-l range = 2 from tensor interactions.
        
        Select the valid relative l' case for each interaction.
        """
        return (l for l in range(max(0, self._l-2), self._l+2 +1, 2))
    
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin S = S'.
        """
        return (self.S_bra, )
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        
        L.ket from central (L.bra) to +2 moment for tensor
        """
        _L_min = max(0, self.L_bra - 2)
        _L_max =        self.L_bra + 2
        gen_ = (l_q for l_q in range(_L_min, _L_max +1))
        return tuple(gen_)
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        return self.centerOfMassMatrixElementEvaluation()
    
    def centerOfMassMatrixElementEvaluation(self):
        """
        There are non common (of COM dependent) constants/operators for all the
        channels. Calling the COM decomposition.
        """
        return self._BrodyMoshinskyTransformation()
    
    @classmethod
    def _calculateIntegrals(cls, key_,  n_integrals =1):
        """
        Select from each the interaction the Talmi integrals
        """
        for p in range(cls._integrals_p_max[key_] + 1, 
                       cls._integrals_p_max[key_] + n_integrals +1):
            
            ## Call here the different integral methods, which must save the 
            ## p-th integral (by _key_) in cls._talmi_integrals attribute.
            raise MatrixElementException("Abstract method, implement integrals!")
            cls._integrals_p_max[key_] += 1
    
    def talmiIntegral(self, key_):
        """ 
        Get or update Talmi integrals for the calculations 
           NOTE, not a final method, can be overwriten for other key_-parameter
           dependent integrals, customize _talmi_integrals accordingly
        """
        if self._p > self._integrals_p_max[key_]:
            n_integrals = max(self.rho_bra, self.rho_ket, 1)
            self._calculateIntegrals(key_, n_integrals)
        return self._talmi_integrals[key_][self._p]
    
    ## Methods
    ## -------------------------------------------------------------------------
    
    def _matrixElementCompoundEvaluation(self):
        """
        Combination of the matrix element for the different modes and channels
        
        i.e. compound decomposition in relative j 
        """
        raise MatrixElementException("Abstract method in abstract class."
                                     " Implement me!")
    

class NucleonAv14TermsInteraction_JTScheme(_TwoBodyMatrixElement_JTCoupled,
                                           _BaseTalmiMultinteraction_JScheme):
    """
    Nuclear intermediate-short range interactions from Argone 14 potential:
        R.B.Wiringa, R.A. Smith, T.L.Ainsworth
        Nucleon-nucleon potentials with and without Delta(1232) degrees of freedom
        Phys. Rev. C 29.4 (4-1984)
    
    Inclusion of all the terms (central, l2, lS, lS^2, S12) from the terms
        v_pi = Cs * (Yuk*s1s2 + TenYuk*S12)
        V_R  = v_c + v_l2*l2 + v_ls*lS + v_ls2*(lS)^2 + v_s12*S12
            v_i = I_i*YukTen2 + (R_i + rm*S_i + rm^2*P_i)*WoodSaxon(r)
    
    The relative l' terms include up to the tensor transference l+2
    The valid total ket.L is extended also to this bra.L + 2
    The valid ket.S are 1 and 0
    
    Every term has a condition to evaluate the l',S',ket.L and antisymmetry 
    condition individually and the l+S+T=even globally.
    
    Constants based on channel (c,l2,lS,lS2,S12, S12_pi, s1s2) by the different 
    projections of spin/isospin (charge independent/symmetric)
    """
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    _BREAK_ISOSPIN    = False
    
    _talmi_integrals = {
        ## keys_: T (for OPE),  for NN, the ST channels are shared
        'c_OPE'      : [],
        't_OPE'      : [],
        'c_NN_T2'    : [],
        'c_NN_WS0'   : [],
        'l2_NN_T2'   : [],
        'l2_NN_WS0'  : [],
        't_NN_T2'    : [],
        't_NN_WS0'   : [],
        'lS_NN_T2'   : [],
        'lS_NN_WS0'  : [],
        'lS2_NN_T2'  : [],
        'lS2_NN_WS0' : [],
    }
    
    _integrals_p_max = {}
    
    _tensor_ang_me = {} ## (j, l, l'): value
    _ls2_ang_me    = {} ## (j, l)    : value 
    
    #===========================================================================
    # CONSTANTS
    #===========================================================================
    
    class PartEnum(Enum):
        OPE = '_OPE'
        NN  = '_NN_'
    
    class TermEnum(Enum):
        c   = 'c'
        t   = 't'
        l2  = 'l2'
        lS  = 'lS'
        lS2 = 'lS2'
    
    class SubNNTermNNEnum(Enum):
        T2  = 'T2'
        WS0 = 'WS0'
    
    CONSTANTS_NN = {
        'c_NN_T2'    : {(0,0): -9.12,   (0,1): -8.1188,
                        (1,0): -6.5572, (1,1): -2.63,},
        'c_NN_WS0'   : {(0,0):  5874,   (0,1):  2800,
                        (1,0):  2700,   (1,1):  1179,},
        'l2_NN_T2'   : {(0,0):  0.62,   (0,1):  0.05, 
                        (1,0):  0.48,   (1,1): -0.12, },
        'l2_NN_WS0'  : {(0,0): -363,    (0,1):  63, 
                        (1,0):  110,    (1,1): -2, },
        't_NN_T2'    : {(1,0):  2.10,   (1,1): -0.91, },
        't_NN_WS0'   : {(1,0): -783,    (1,1):  406, },
        'lS_NN_T2'   : {(1,0):  0.42,   (1,1):  0.61, },
        'lS_NN_WS0'  : {(1,0): -242,    (1,1): -879, },
        'lS2_NN_T2'  : {(1,0): -0.55,   (1,1): -0.54, },
        'lS2_NN_WS0' : {(1,0): -44,     (1,1):  536, },
    }
    
    PARAMS_FORCE_OPE = {} # defined in classmethod.setInteractionParameters()
    
    FACTOR_Fpp =  0.284604989   # sqrt(4*pi*0.081)
    FACTOR_Fnn = -0.284604989
    FACTOR_Fc  =  0.284604989
    
    ## TESTING COMPONENTS  ---------------------------- 
    _TEST_QUADRATURES     = True
    _SWITCH_OFF_CONSTANTS = False
    _SWITCH_OFF_TERMS     = {
        'c_OPE'      : False,   't_OPE'      : False,
        'c_NN_T2'    : False,   'c_NN_WS0'   : False,
        'l2_NN_T2'   : False,   'l2_NN_WS0'  : False,
        't_NN_T2'    : False,   't_NN_WS0'   : False,
        'lS_NN_T2'   : False,   'lS_NN_WS0'  : False,
        'lS2_NN_T2'  : False,   'lS2_NN_WS0' : False,
    }
    ##  -----------------------------------------------
    
    @classmethod
    def getInteractionKeyword(cls, term, part, integral_nn=''):
        """
        Access the dictionaries for Talmi integrals and interaction constants
        :term = from cls.TermEnum
        :part = from cls.PartEnum
        :integral_nn = from cls.SubNNTermNNEnum (does not apply for OPE part)
        
        """
        return ''.join([term, part, integral_nn])
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        
        cls.PARAMS_SHO[SHO_Parameters.b_length] = kwargs[SHO_Parameters.b_length]
        
        e_v18 = ConstantsV18
        const = (cls.FACTOR_Fc**2)  * (e_v18.M_PION_0 + 2*e_v18.M_PION_pm) / 3
        
        mNN = 3 * e_v18.HBAR_C / (e_v18.M_PION_0 + 2*e_v18.M_PION_pm) 
        
        cls.PARAMS_FORCE_OPE = {
                CentralMEParameters.constant : const, 
                CentralMEParameters.mu_length: mNN,
        }
        
        cls.PARAMS_FORCE = {
            CentralMEParameters.mu_length  : mNN,  # for NN terms (shared)
            CentralMEParameters.opt_mu_2   : 0.2,  # a  diffuse length
            CentralMEParameters.opt_mu_3   : 0.5,  # r0 length
            CentralMEParameters.opt_cutoff : np.sqrt(1/2), #1 / (2.1**0.5),
        }
        
        print(_LINE_2,"  Default constants for [Argone 14 - Nuclear] interaction:")
        prettyPrintDictionary({'PARAMS_FORCE    ' : cls.PARAMS_FORCE, })
        prettyPrintDictionary({'PARAMS_FORCE_OPE' : cls.PARAMS_FORCE_OPE, })
        prettyPrintDictionary({'CONSTANTS_NN    ' : cls.CONSTANTS_NN, })
        print(f"     * Set interaction constants to 1 : [{cls._SWITCH_OFF_CONSTANTS}]")
        print(_LINE_2)
        
        ## testing ----------------------------------------------------------- 
        if cls._SWITCH_OFF_CONSTANTS: 
            for k in cls.CONSTANTS_NN:
                for k2 in cls.CONSTANTS_NN[k]:
                    cls.CONSTANTS_NN[k][k2] = 1
            for k in cls.PARAMS_FORCE_OPE:
                cls.CONSTANTS_NN[k][CentralMEParameters.constant] = 1
        
        cls._integrals_p_max = dict([(k,-1) for k in cls._talmi_integrals.keys()])
        cls._talmi_integrals = dict([(k, list()) for k in cls._talmi_integrals.keys()])
    
    def __init__(self, bra, ket, run_it=True):        
        _TwoBodyMatrixElement_JTCoupled.__init__(self, bra, ket, run_it=run_it)
    
    def _deltaConditionsForCOM_Iteration(self):
        """ For the antisymmetrization_ of the wave functions. """
        if (((self.S_bra + self.T + self._l) % 2 == 1) and 
            ((self.S_ket + self.T + self._l_q) % 2 == 1)):
                return True
        return False
    
    #===========================================================================
    # INDIVIDUAL MATRIX ELEMETNTS AND TALMI INTEGRALS
    #=========================================================================== 
    @classmethod
    def _integralCentralOPE(cls, p, key_):
        """
        It has analytical expression, fill the different isospin-channels
        TODO: Test with quadratures
        """
        b      = cls.PARAMS_SHO.get(SHO_Parameters.b_length)
        mu     = cls.PARAMS_FORCE_OPE.get(CentralMEParameters.mu_length)
        cutoff = cls.PARAMS_FORCE.get(CentralMEParameters.opt_cutoff)
        
        Y1 = talmiIntegral(p, PotentialForms.Yukawa, b, mu)
        Y2 = talmiIntegral(p, PotentialForms.YukawaGauss_power, b, mu, 
                           n_power=0, opt_mu_2=cutoff)
        I = Y1 - Y2
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integralTensorOPE(cls, p, key_):
        """
        Do not have analytical expression, evaluates quadratures here
        """
        poly   = [1, 3, 3]
        x, w   = getGeneralizedLaguerreRootsWeights(2*p + 1)
        b      = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu     = cls.PARAMS_FORCE_OPE.get(CentralMEParameters.mu_length)
        cutoff = cls.PARAMS_FORCE[CentralMEParameters.opt_cutoff]
        
        A  = np.sqrt(2) * b / mu
        B  = cutoff * (mu**2)
        def __funct(x, A, B, n):
            return ((1 - np.exp(-B*np.power(x, 2))) 
                    / (np.exp(np.power(x/A,2)) * np.power(x, n)) )
        I = 0.0
        for n, c_i in enumerate(poly):
            I += c_i * (2**n) * np.dot(w, __funct(x, A, B, n))
        
        aux = (2*p + 2)*np.log(A) + gamma_half_int(2*p + 3)
        I  *= np.sqrt(2) * b**2 / (mu * np.exp(aux))
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integralTensor2NN(cls, p, key_):
        """
        Do not have analytical expression, evaluates quadratures here
        """
        poly2 = polynomialProduct([1, 3, 3], [1, 3, 3])
        x, w  = getGeneralizedLaguerreRootsWeights(2*p)
        
        b  = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        # p  = self._p 
        
        A  = 2 * np.sqrt(2) * b / mu
        B  = cls.PARAMS_FORCE[CentralMEParameters.opt_cutoff] * (mu / 2)**2
        def __funct(x, A, B, n):
            return (np.power(1 - np.exp(-B*np.power(x, 2)), 2) 
                    / (np.exp(np.power(x/A,2)) * np.power(x, n)) )
        I = 0.0
        for n, c_i in enumerate(poly2):
            I += c_i * (2**n) * np.dot(w, __funct(x, A, B, n))
        
        aux = (2*p + 2)*np.log(A) + gamma_half_int(2*p + 3)
        I  *= 2 * np.sqrt(2) * b**2 / (mu * np.exp(aux))
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integralWoodSaxonNN(cls, p, key_):
        
        b  = cls.PARAMS_SHO.get(SHO_Parameters.b_length)
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        a  = cls.PARAMS_FORCE[CentralMEParameters.opt_mu_2]
        r0 = cls.PARAMS_FORCE[CentralMEParameters.opt_mu_3]
        
        N = int(key_[-1])
        
        I = talmiIntegral(p, PotentialForms.Wood_Saxon, b, mu, 
                          n_power=N, opt_mu_2=a, opt_mu_3=r0)
        cls._talmi_integrals[key_].append(I)
    
    
    @classmethod
    def _calculateIntegrals(cls, key_,  n_integrals =1):
        """
        Select from each the interaction the Talmi integrals
        """
        for p in range(cls._integrals_p_max[key_] + 1, 
                       cls._integrals_p_max[key_] + n_integrals +1):
            
            if cls.PartEnum.OPE in key_:
                if   key_.startswith(cls.TermEnum.c):
                    cls._integralCentralOPE(p, key_)
                elif key_.startswith(cls.TermEnum.t):
                    cls._integralTensorOPE(p, key_)
                else:
                    raise MatrixElementException(f"invalid OPE interaction integral, got [{key_}]")
            elif cls.PartEnum.NN in key_: ## NN integrals
                if   key_.endswith(cls.SubNNTermNNEnum.T2):
                    cls._integralTensor2NN(p, key_)
                elif key_[:-1].endswith('WS'):
                    cls._integralWoodSaxonNN(p, key_)
                else:
                    raise MatrixElementException(f"invalid NN interaction integral, got [{key_}]")
            else:
                raise MatrixElementException(f"not OPE or NN, got [{key_}]")
                        
            cls._integrals_p_max[key_] += 1
    
    def _get_tensor_angular_me(self):
        """ C_T*<ljS| Y_2*X_2 |l'jS> with S=S'=1 """
        tupl = (self._j_rel, self._l, self._l_q)
        if not tupl in self._tensor_ang_me:
            me_ang = safe_wigner_6j(self._l,    self.S_bra, self._j_rel, 
                                    self.S_ket, self._l_q,  2)
            
            me_ang *= safe_3j_symbols(self._l, 2, self._l_q, 0, 0, 0)
            me_ang *= (10.954451150103322 * 
                      np.sqrt((2*self._l + 1)*(2*self._l_q + 1)*(2*self._j_rel + 1)) * 
                      ((-1)**(self.S_bra + self._j_rel)))
            ## 2*sqrt(30) = 10.9545
            
            self._tensor_ang_me[tupl] = me_ang
        return self._tensor_ang_me.get(tupl)
    
    def _get_spinOrbitSquared_angular_me(self):
        """ C_T*<ljS| (l*S)^2 = (j^2 - l^2 - S^2)^2 |l'jS> with S=S'=1 & l=l'"""
        tupl = (self._j_rel, self._l)
        if not tupl in self._ls2_ang_me:
            jjp1 = self._j_rel * (self._j_rel + 1)
            llp1 = self._l * (self._l + 1)
            ssp1 = self.S_bra * (self.S_bra + 1)
            
            aux = llp1*(llp1 + 2*ssp1) + jjp1*(jjp1 - 2*llp1) + ssp1*(ssp1 - 2*jjp1)
                        
            self._ls2_ang_me[tupl] = 0.25 * aux
        return self._ls2_ang_me.get(tupl)
    
    ## ------------------------------------------------------------------------
    ## COMPONENTS
    
    def _component_OPE_central(self):
        """
        Central terms from the OPE term, the np states (MT=0) has two terms.
        """
        me_ang = 1
        
        key_ = self.getInteractionKeyword(self.TermEnum.c, self.PartEnum.OPE)
        if self._SWITCH_OFF_TERMS[key_]: return 0
        
        c1   = self.PARAMS_FORCE_OPE[CentralMEParameters.constant]
        me_rad  = c1 * self.talmiIntegral(key_)
        
        return me_ang * me_rad
        
    def _component_OPE_tensor(self):
        """
        Tensor S12 terms from the OPE term, the np states (MT=0) has two terms.
        """
        me_ang = self._get_tensor_angular_me()
        
        key_   = self.getInteractionKeyword( self.TermEnum.t, self.PartEnum.OPE)
        if self._SWITCH_OFF_TERMS[key_]: return 0
        
        c1     = self.PARAMS_FORCE_OPE[CentralMEParameters.constant]
        me_rad = c1 * self.talmiIntegral(key_)
        
        return me_ang * me_rad
    
    def _component_NN_central(self):
        """ 
        central term for NN short-intermediate phenomenological interaction
            <v(ang/spin)>=1 * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
        """
        me_ang = 1
        me_rad = 0.0
        for x in self.SubNNTermNNEnum.members():
            int_ = self.getInteractionKeyword(self.TermEnum.c, self.PartEnum.NN, x)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            key_ = (self.S_bra, self.T)
            
            constant = self.CONSTANTS_NN[int_][key_]
            integral = self.talmiIntegral(int_)
            me_rad  += constant * integral 
        
        return me_ang * me_rad
    
    def _component_NN_L2(self):
        """ 
        L^2 term for NN short-intermediate phenomenological interaction
            <v(ang/spin)> * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
        """
        me_ang = self._l*(self._l + 1)
        me_rad = 0.0
        for x in self.SubNNTermNNEnum.members():
            int_ = self.getInteractionKeyword(self.TermEnum.l2, self.PartEnum.NN, x)
            key_ = (self.S_bra, self.T)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            constant = self.CONSTANTS_NN[int_][key_]
            integral = self.talmiIntegral(int_)
            me_rad  += constant * integral 
        
        return me_ang * me_rad
    
    def _component_NN_tensor(self):
        """ 
        S12 term for NN short-intermediate phenomenological interaction
            <v(ang/spin)> * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
        """
        me_ang = self._get_tensor_angular_me()
        me_rad = 0.0
        for x in self.SubNNTermNNEnum.members():
            int_ = self.getInteractionKeyword(self.TermEnum.t, self.PartEnum.NN, x)
            key_ = (self.S_bra, self.T)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            constant = self.CONSTANTS_NN[int_][key_]
            integral = self.talmiIntegral(int_)
            me_rad  += constant * integral 
        
        return me_ang * me_rad
    
    def _component_NN_LS(self):
        """
        L.S term for NN short-intermediate phenomenological interaction
            <v(ang/spin)> * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
        """
        j = self._j_rel
        me_ang = 0.5*( (j*(j + 1)) - (self._l*(self._l + 1)) - 2)   ## (S=1)^2=1*2
        me_rad = 0.0
        for x in self.SubNNTermNNEnum.members():
            int_ = self.getInteractionKeyword(self.TermEnum.lS, self.PartEnum.NN, x)
            key_ = (self.S_bra, self.T)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            constant = self.CONSTANTS_NN[int_][key_]
            integral = self.talmiIntegral(int_)
            me_rad  += constant * integral 
        
        return me_ang * me_rad
    
    def _component_NN_LS2(self):
        """ 
        (L.S)^2 term for NN short-intermediate phenomenological interaction
            <v(ang/spin)> * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
        """
        me_ang = self._get_spinOrbitSquared_angular_me()
        me_rad = 0.0
        for x in self.SubNNTermNNEnum.members():
            int_ = self.getInteractionKeyword(self.TermEnum.lS2, self.PartEnum.NN, x)
            key_ = (self.S_bra, self.T)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            constant = self.CONSTANTS_NN[int_][key_]
            integral = self.talmiIntegral(int_)
            me_rad  += constant * integral
        
        return me_ang * me_rad
    
    def _matrixElementCompoundEvaluation(self):
        """
        Combination of the matrix element for the different modes and channels
        
        i.e. compound decompossition in relative j 
        """
        ## central CD-ope & NN + l2 terms
        sum_central = 0.0
        
        if (self._l == self._l_q) and (self.L_bra == self.L_ket):
            # spin_factor = (4*self.T - 3) * (4*self.S_bra - 3)
            sum_central += self._component_OPE_central() #* spin_factor
            sum_central += self._component_NN_central()
            sum_central += self._component_NN_L2() ## TODO: THIS WHAT???
        
        if self.S_bra != 1:      ## tensor/ls dependent have only S=1 components
            return sum_central        
        
        ## decomposition in j-relative system
        j_min = max( abs(self.S_bra - self._l), abs(self.J - self._L))
        j_max = min(     self.S_bra + self._l,      self.J + self._L)
    
        sum_noncentral = 0.0
        
        assert self.S_bra == self.S_ket and (self._l+self._l_q)%2 == 0, "Invalid phase conditions"
        for j in range(j_min, j_max +1):
            self._j_rel = j
            fac_1 = safe_wigner_6j(self._l, self.L_bra,  self._L, 
                                   self.J , self._j_rel, self.S_bra)
            if self.isNullValue(fac_1): continue
            fac_2 = safe_wigner_6j(self._l_q, self.L_ket,  self._L, 
                                   self.J ,   self._j_rel, self.S_ket)
            if self.isNullValue(fac_2): continue
            
            components    = [0,]*4
            #spin_factor   = (4*self.T - 3)
            components[0] = self._component_OPE_tensor()# * spin_factor
            components[1] = self._component_NN_tensor()
            if self._l == self._l_q:
                components[2] = self._component_NN_LS()
                components[3] = self._component_NN_LS2()
            
            factor = (2*j + 1) * fac_1 * fac_2
            sum_noncentral += factor * sum(components)
        
        phase   = (-1)**(self.L_bra + self.L_ket)
        factor_jdecomp  = phase * np.sqrt((2*self.L_bra + 1)*(2*self.L_ket + 1))
        
        sum_noncentral *= factor_jdecomp
        
        return sum_central + sum_noncentral

class NucleonAv18TermsInteraction_JTScheme(NucleonAv14TermsInteraction_JTScheme):
    
    '''
    Nuclear intermediate-short range interactions from Argone 18 potential:
        R.B.Wiringa, V.G.J. Stoks, R.Schiavilla
        Accurate nucleon-nucleon potential with charge-independence breaking
        Phys. Rev. C 51.1 (1-1994)
        
    Inclusion of all the terms (central, l2, lS, lS^2, S12) from the terms
        v_pi = Cs * (Yuk*s1s2 + TenYuk*S12)
        V_R  = v_c + v_l2*l2 + v_ls*lS + v_ls2*(lS)^2 + v_s12*S12
            v_i = I_i*YukTen2 + (R_i + rm*S_i + rm^2*P_i)*WoodSaxon(r)
    
    The relative l' terms include up to the tensor transference l+2
    The valid total ket.L is extended also to this bra.L + 2
    The valid ket.S are 1 and 0
    
    Every term has a condition to evaluate the l',S',ket.L and antisymmetry 
    condition individually and the l+S+T=even globally.
    
    Constants based on channel (c,l2,lS,lS2,S12, S12_pi, s1s2) and the interaction
    type (pp, pn, nn) with the different (S,T) central channels.
    
    It implements the isospin-breaking components with respect to the Av-14:
        * Central / isotensor /Charge - assymetric term
    
    '''
    _talmi_integrals = { 
        ## keys_: T (for OPE),  for NN, the ST channels are shared
        'c_OPE'      : {
            (1,-1): [],    # pp -> Y(m_pi 0)
            (1, 0): [],    # pn -> Y(m_pi +-)
            (1, 1): [],    # nn -> Y(m_pi 0)
            (0, 0): [],    # pn -> Y(m_pi +-)
        },
        't_OPE'      : {
            (1,-1): [],    # pp -> T(m_pi 0)
            (1, 0): [],    # pn -> T(m_pi +-)
            (1, 1): [],    # nn -> T(m_pi 0)
            (0, 0): [],    # pn -> T(m_pi +-)
        },
        'c_NN_T2'    : [],
        'c_NN_WS0'   : [],
        'c_NN_WS1'   : [],
        'c_NN_WS2'   : [],
        'l2_NN_T2'   : [],
        'l2_NN_WS0'  : [],
        'l2_NN_WS1'  : [],
        'l2_NN_WS2'  : [],
        't_NN_T2'    : [],
        't_NN_WS0'   : [],
        't_NN_WS1'   : [],
        't_NN_WS2'   : [],
        'lS_NN_T2'   : [],
        'lS_NN_WS0'  : [],
        'lS_NN_WS1'  : [],
        'lS_NN_WS2'  : [],
        'lS2_NN_T2'  : [],
        'lS2_NN_WS0' : [],
        'lS2_NN_WS1' : [],
        'lS2_NN_WS2' : [],
    }
    
    class SubNNTermNNEnum(Enum):
        T2  = 'T2'
        WS0 = 'WS0'
        WS1 = 'WS1'
        WS2 = 'WS2'
    
    ## NOTE_ Remember, In this code, m_t(protons) = +1 (in taurus is -1)
    CONSTANTS_NN = {
        'c_NN_T2'    : {
            (0,0)   : -2.09971, 
            (0,1, 1): -11.27028, (0,1, 0): -10.66788, (0,1,-1): -11.27028, 
            (1,0)   : -8.62770, 
            (1,1, 1): -7.62701,  (1,1, 0): -7.62701,  (1,1,-1): -7.62701,},
        'c_NN_WS0'   : {
            (0,0)   : 1204.4301, 
            (0,1, 1): 3346.6874, (0,1, 0): 3126.5542, (0,1,-1): 3342.7664, 
            (1,0)   : 2605.2682, 
            (1,1, 1): 1815.4920, (1,1, 0): 1813.5315, (1,1,-1): 1811.5710, },
        'c_NN_WS1'   : {
            (0,0)   : 511.9380, 
            (0,1, 1): 1859.5627, (0,1, 0): 1746.4298, (0,1,-1): 1857.4367, 
            (1,0)   : 1459.6345, 
            (1,1, 1): 969.3863,  (1,1, 0): 966.2483,  (1,1,-1): 967.2603, },
        'c_NN_WS2'   : {
            (0,0)   : 0, 
            (0,1, 1): 0,         (0,1, 0): 0,         (0,1,-1): 0, 
            (1,0)   : 441.9733, 
            (1,1, 1): 1847.8059, (1,1, 0): 1847.8059, (1,1,-1): 1847.8059, },
        'l2_NN_T2'   : {(0,0):   -0.31452, (0,1):  0.12472, 
                        (1,0):   -0.13201, (1,1):  0.06709, },
        'l2_NN_WS0'  : {(0,0):   217.4559, (0,1):  16.7780, 
                        (1,0):   253.4350, (1,1): 342.0669, },
        'l2_NN_WS1'  : {(0,0):   117.9063, (0,1):   9.0972, 
                        (1,0):   137.4144, (1,1): 185.4713, },
        'l2_NN_WS2'  : {(0,0):          0, (0,1): 0, 
                        (1,0):    -1.0076, (1,1): -615.2339, },
        't_NN_T2'    : {(1,0):   1.485601, (1,1): 1.07985, },
        't_NN_WS0'   : {(1,0):          0, (1,1): 0, },
        't_NN_WS1'   : {(1,0): -1126.8359, (1,1): -190.0949, },
        't_NN_WS2'   : {(1,0):   370.1324, (1,1): -811.2040, },
        'lS_NN_T2'   : {(1,0):    0.10180, (1,1): -0.62697, },
        'lS_NN_WS0'  : {(1,0):    86.0658, (1,1): -570.5571, },
        'lS_NN_WS1'  : {(1,0):    46.6655, (1,1): -309.3605, },
        'lS_NN_WS2'  : {(1,0):  -356.5175, (1,1): 819.1222, },
        'lS2_NN_T2'  : {(1,0):    0.07357, (1,1): 0.74129, },
        'lS2_NN_WS0' : {(1,0):  -217.5791, (1,1): 9.3418, },
        'lS2_NN_WS1' : {(1,0):  -117.9731, (1,1): 5.0652, },
        'lS2_NN_WS2' : {(1,0):    18.3935, (1,1): -376.4384, },
    }
    
    PARAMS_FORCE_OPE = {} # defined in classmethod.setInteractionParameters()
    
    FACTOR_Fpp =  0.27386127875   # sqrt(0.075)
    FACTOR_Fnn = -0.27386127875
    FACTOR_Fc  =  0.27386127875
    
    ## TESTING COMPONENTS  ---------------------------------------------------
    _TEST_QUADRATURES     = True
    _SWITCH_OFF_CONSTANTS = False  ## Set constants to 1
    _SWITCH_OFF_TERMS     = {
        ## keys_: T (for OPE),  for NN, the ST channels are shared
        'c_OPE'      : False,  't_OPE'      : False,
        'c_NN_T2'    : False,
        'c_NN_WS0'   : False,  'c_NN_WS1'   : False,  'c_NN_WS2'   : False,
        'l2_NN_T2'   : False,
        'l2_NN_WS0'  : False,  'l2_NN_WS1'  : False,  'l2_NN_WS2'  : False,
        't_NN_T2'    : False,
        't_NN_WS0'   : False,  't_NN_WS1'   : False,  't_NN_WS2'   : False,
        'lS_NN_T2'   : False,
        'lS_NN_WS0'  : False,  'lS_NN_WS1'  : False,  'lS_NN_WS2'  : False,
        'lS2_NN_T2'  : False,
        'lS2_NN_WS0' : False,  'lS2_NN_WS1' : False,  'lS2_NN_WS2' : False,
    }
    ## -----------------------------------------------------------------------
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        
        cls.PARAMS_SHO[SHO_Parameters.b_length] = kwargs[SHO_Parameters.b_length]
        
        e_v18 = ConstantsV18
        scaling_0pm = (e_v18.M_PION_0 / e_v18.M_PION_pm)**2
        
        const_pp = (cls.FACTOR_Fpp**2) * scaling_0pm * e_v18.M_PION_0 / 3
        const_nn = (cls.FACTOR_Fnn**2) * scaling_0pm * e_v18.M_PION_0 / 3
        const_pn1= 2 * (cls.FACTOR_Fc**2) * e_v18.M_PION_pm / 3
        const_pn0= cls.FACTOR_Fpp * cls.FACTOR_Fnn * scaling_0pm * e_v18.M_PION_0 / 3        
        
        ## lengths for the code has to be for r/mu, not as in the article
        mu_pp = (e_v18.HBAR_C / e_v18.M_PION_0)
        mu_pn = (e_v18.HBAR_C / e_v18.M_PION_pm)
        
        ## NOTE_ Remember, In this code, m_t(protons) = +1 (in taurus is -1)
        cls.PARAMS_FORCE_OPE = {
            (1, 1): { # 'pp'
                CentralMEParameters.constant : const_pp, 
                CentralMEParameters.mu_length: mu_pp,
                },
            (1, 0): { # 'pn'
                CentralMEParameters.constant : const_pn0,
                CentralMEParameters.mu_length: mu_pp,
                },
            (1,-1): { # 'nn'
                CentralMEParameters.constant : const_nn, 
                CentralMEParameters.mu_length: mu_pp,
                },
            (0, 0): {
                CentralMEParameters.constant : -const_pn1, 
                CentralMEParameters.mu_length: mu_pn,
                },
        }
        
        mNN = 3 * e_v18.HBAR_C / (e_v18.M_PION_0 + 2*e_v18.M_PION_pm) 
        
        cls.PARAMS_FORCE = {
            CentralMEParameters.mu_length  : mNN,  # for NN terms (shared)
            CentralMEParameters.opt_mu_2   : 0.2,  # a  diffuse length
            CentralMEParameters.opt_mu_3   : 0.5,  # r0 length
            CentralMEParameters.opt_cutoff : 0.6900655593423541, #1 / (2.1**0.5),
        }
        
        print(_LINE_2,"  Default constants for [Argone 14 - Nuclear] interaction:")
        prettyPrintDictionary({'PARAMS_FORCE    ' : cls.PARAMS_FORCE, })
        prettyPrintDictionary({'PARAMS_FORCE_OPE' : cls.PARAMS_FORCE_OPE})
        prettyPrintDictionary({'CONSTANTS_NN    ' : cls.CONSTANTS_NN, })
        print(f"     * Set interaction constants to 1 : [{cls._SWITCH_OFF_CONSTANTS}]")
        print(_LINE_2)
        
        ## testing ----------------------------------------------------------- 
        if cls._SWITCH_OFF_CONSTANTS: 
            for k in cls.CONSTANTS_NN:
                for k2 in cls.CONSTANTS_NN[k]:
                    cls.CONSTANTS_NN[k][k2] = 1
            for k in cls.PARAMS_FORCE_OPE:
                cls.CONSTANTS_NN[k][CentralMEParameters.constant] = 1
        
        cls._integrals_p_max = dict([(k,-1) for k in cls._talmi_integrals.keys()])
        cls._talmi_integrals = dict([(k, list()) for k in cls._talmi_integrals.keys()])
    
    #===========================================================================
    # INDIVIDUAL MATRIX ELEMETNTS AND TALMI INTEGRALS
    #=========================================================================== 
    @classmethod
    def _integralCentralOPE(cls, p, key_):
        """
        It has analytical expression, fill the different isospin-channels
        TODO: Test with quadratures
        """
        b      = cls.PARAMS_SHO.get(SHO_Parameters.b_length)
        cutoff = cls.PARAMS_FORCE.get(CentralMEParameters.opt_cutoff)
        
        for T in (0, 1):
            for MT in range(-T, T +1):
                mu = cls.PARAMS_FORCE_OPE[(T, MT)].get(CentralMEParameters.mu_length)
                
                Y1 = talmiIntegral(p, PotentialForms.Yukawa, b, mu)
                Y2 = talmiIntegral(p, PotentialForms.YukawaGauss_power, b, mu, 
                                   n_power=0, opt_mu_2=cutoff)
                I = Y1 - Y2
                
                cls._talmi_integrals[key_][(T, MT)].append(I)
    
    @classmethod
    def _integralTensorOPE(cls, p, key_):
        """
        Do not have analytical expression, evaluates quadratures here
        """
        poly   = [1, 3, 3]
        x, w   = getGeneralizedLaguerreRootsWeights(2*p + 1)
        b      = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        cutoff = cls.PARAMS_FORCE[CentralMEParameters.opt_cutoff]
        
        for T in (0, 1):
            for MT in range(-T, T +1):
                # p  = self._p 
                mu = cls.PARAMS_FORCE_OPE[(T, MT)].get(CentralMEParameters.mu_length)
                A  = np.sqrt(2) * b / mu
                B  = cutoff * (mu**2)
                def __funct(x, A, B, n):
                    return ((1 - np.exp(-B*np.power(x, 2))) 
                            / (np.exp(np.power(x/A,2)) * np.power(x, n)) )
                I = 0.0
                for n, c_i in enumerate(poly):
                    I += c_i * (2**n) * np.dot(w, __funct(x, A, B, n))
                
                aux = (2*p + 2)*np.log(A) + gamma_half_int(2*p + 3)
                I  *= np.sqrt(2) * b**2 / (mu * np.exp(aux))
                
                cls._talmi_integrals[key_][(T, MT)].append(I)
    
    @classmethod
    def _calculateIntegrals(cls, key_,  n_integrals =1):
        """
        Select from each the interaction the Talmi integrals
        """
        for p in range(cls._integrals_p_max[key_] + 1, 
                       cls._integrals_p_max[key_] + n_integrals +1):
            
            if cls.PartEnum.OPE in key_:
                if   key_.startswith(cls.TermEnum.c):
                    cls._integralCentralOPE(p, key_)
                elif key_.startswith(cls.TermEnum.t):
                    cls._integralTensorOPE(p, key_)
                else:
                    raise MatrixElementException(f"invalid OPE interaction integral, got [{key_}]")
            elif cls.PartEnum.NN in key_: ## NN integrals
                if   key_.endswith(cls.SubNNTermNNEnum.T2):
                    cls._integralTensor2NN(p, key_)
                elif key_[:-1].endswith('WS'):
                    cls._integralWoodSaxonNN(p, key_)
                else:
                    raise MatrixElementException(f"invalid NN interaction integral, got [{key_}]")
            else:
                raise MatrixElementException(f"not OPE or NN, got [{key_}]")
                        
            cls._integrals_p_max[key_] += 1
    
    def talmiIntegral(self, key_, MT=None):
        """ 
        Get or update Talmi integrals for the calculations
        """
        if key_.endswith(self.PartEnum.OPE):
            assert isinstance(MT, int), "MT is required for OPE part!"
            if self._p > self._integrals_p_max[key_]:
                n_integrals = max(self.rho_bra, self.rho_ket, 1)
                ## calculate integrals for OPE terms calculates all the MT terms
                self._calculateIntegrals(key_, n_integrals)
            return self._talmi_integrals[key_][(self.T, MT)][self._p]
        else:
            if self._p > self._integrals_p_max[key_]:
                n_integrals = max(self.rho_bra, self.rho_ket, 1)
                self._calculateIntegrals(key_, n_integrals)
            return self._talmi_integrals[key_][self._p]
    
    ## ------------------------------------------------------------------------
    ## COMPONENTS
    
    def _component_OPE_central(self):
        """ 
        central term for NN short-intermediate phenomenological interaction
            <v(ang/spin)>=1 * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
            
            # must be included all 
                CI = c, tau, sig, tau*sig
                CD = T, sig*T               (isotensor-dependent)
                CA = tauZ, sig*tauZ
        """
        me_ang = 1
        me_rad = 0.0
        T      =  self.T
        idx_   = 3 - (2*self.S_bra + self.T)
        kkk    = CentralMEParameters.constant
        ##             ST  11 10 01  00  
        _CIDA_cst = {'s'  : [3, 1, -3, -1], 'Ts':  [1,  0, -1, 0],  
                     'tzs': [1, 0, -1,  0], }
        
        for x in self.SubNNTermNNEnum.members():
            integrals_CI, integrals_CD, integrals_CA = 0.0, 0.0, 0.0
            
            int_ = self.getInteractionKeyword(self.TermEnum.c, self.PartEnum.OPE)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            if (T == 0):
                integral = self.talmiIntegral(int_, MT=0)                
                integrals_CI = self.PARAMS_FORCE_OPE[(T, 0)][kkk] * integral
            else:
                # for MT in (-1,0,1):
                #     key_ = (self.S_bra, self.T, MT)
                integral_pp = self.talmiIntegral(int_, MT= 1)
                integral_pn = self.talmiIntegral(int_, MT= 0)
                integral_nn = self.talmiIntegral(int_, MT=-1)
                integrals_CI = (
                    self.PARAMS_FORCE_OPE[(T, 1)][kkk] * integral_pp +
                    self.PARAMS_FORCE_OPE[(T, 0)][kkk] * integral_pn +
                    self.PARAMS_FORCE_OPE[(T,-1)][kkk] * integral_nn) / 3
                integrals_CD = ((
                    self.PARAMS_FORCE_OPE[(T, 1)][kkk] * integral_pp +
                    self.PARAMS_FORCE_OPE[(T,-1)][kkk] * integral_nn) / 2
                    - self.PARAMS_FORCE_OPE[(T, 0)][kkk] * integral_pn) / 6
                integrals_CA = (
                    self.PARAMS_FORCE_OPE[(T, 1)][kkk] * integral_pp -
                    self.PARAMS_FORCE_OPE[(T,-1)][kkk] * integral_nn)  / 4
            
            sum_ = 0
            sum_ += _CIDA_cst['s'][idx_] * integrals_CI / 16
            if self.T == 1:
                sum_ += _CIDA_cst[ 'Ts'][idx_] * integrals_CD / 4
                sum_ += _CIDA_cst['tzs'][idx_] * integrals_CA / 4
            
            me_rad  += sum_
        return me_ang * me_rad
    
    def _component_OPE_tensor(self):
        """
        Tensor S12 terms from the OPE term, the np states (MT=0) has two terms.
        """
        if self.S_bra == 0: return 0
        
        me_ang = self._get_tensor_angular_me()
        me_rad = 0.0
        T      =  self.T
        idx_   = 3 - (2*self.S_bra + self.T)
        kkk    = CentralMEParameters.constant
        ##             ST  11 10 01  00  
        _CIDA_cst = {'c': [3, 1, 0, 0], 'T':  [1, 0, 0, 0]}
        
        for x in self.SubNNTermNNEnum.members():
            integrals_CI, integrals_CD = 0.0, 0.0
            
            int_ = self.getInteractionKeyword(self.TermEnum.t, self.PartEnum.OPE)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            if (T == 0):
                integral = self.talmiIntegral(int_, MT=0)
                integrals_CI = self.PARAMS_FORCE_OPE[(T, 0)][kkk] * integral
            else:
                integral_pp = self.talmiIntegral(int_, MT= 1)
                integral_pn = self.talmiIntegral(int_, MT= 0)
                integral_nn = self.talmiIntegral(int_, MT=-1)
                integrals_CI = (
                    self.PARAMS_FORCE_OPE[(T, 1)][kkk] * integral_pp +
                    self.PARAMS_FORCE_OPE[(T, 0)][kkk] * integral_pn +
                    self.PARAMS_FORCE_OPE[(T,-1)][kkk] * integral_nn ) / 3
                integrals_CD = ((
                    self.PARAMS_FORCE_OPE[(T, 1)][kkk] * integral_pp +
                    self.PARAMS_FORCE_OPE[(T,-1)][kkk] * integral_nn ) / 2
                    - self.PARAMS_FORCE_OPE[(T, 0)][kkk] * integral_pn) / 6
            ## integrals_CA:: specified in the article: 
            ## ""we neglect the possibility of a charge-asymmetric tensor term""
            
            sum_ = 0
            sum_ += _CIDA_cst['c'][idx_] * integrals_CI / 4
            if self.T == 1:
                sum_ += _CIDA_cst['T'][idx_] * integrals_CD
            
            me_rad  += sum_
        return me_ang * me_rad
    
    def _component_NN_central(self):
        """ 
        central term for NN short-intermediate phenomenological interaction
            <v(ang/spin)>=1 * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
            
            # must be included all 
                CI = c, tau, sig, tau*sig
                CD = T, sig*T               (isotensor-dependent)
                CA = tauZ, sig*tauZ
        """
        me_ang = 1
        me_rad = 0.0
        S, T   = self.S_bra, self.T
        idx_   = 3 - (2*self.S_bra + self.T)
        ##             ST  11 10 01  00  
        _CIDA_cst = {'c': [9, 3,  3,  1], 't':   [3, -3, 1, -1], 
                     's': [3, 1, -3, -1], 'ts':  [1, -1, -1, 1], 
                     'T': [3, 0, 1, 0],   'Ts':  [1,  0, -1, 0], 
                     'tz':[3, 0, 1, 0],   'tzs': [1,  0, -1, 0],}
        
        for x in self.SubNNTermNNEnum.members():
            integrals_CI, integrals_CD, integrals_CA = 0.0, 0.0, 0.0
            
            int_ = self.getInteractionKeyword(self.TermEnum.c, self.PartEnum.NN, x)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            integral = self.talmiIntegral(int_)
            
            if self.T==0:
                integrals_CI = self.CONSTANTS_NN[int_][(S, T)] * integral
            else:
                integrals_CI = (
                    self.CONSTANTS_NN[int_][(S, T,  1)] +
                    self.CONSTANTS_NN[int_][(S, T,  0)] +
                    self.CONSTANTS_NN[int_][(S, T, -1)] ) * integral / 3
                integrals_CD = ((
                    self.CONSTANTS_NN[int_][(S, T,  1)] +
                    self.CONSTANTS_NN[int_][(S, T, -1)] ) / 2
                    - self.CONSTANTS_NN[int_][(S, T, 0)]) * integral / 6
                integrals_CA = (
                    self.CONSTANTS_NN[int_][(S, T,  1)] -
                    self.CONSTANTS_NN[int_][(S, T, -1)]) * integral / 4
            
            sum_ = 0
            for prt in ('c', 't', 's', 'ts'):
                sum_ += _CIDA_cst[prt][idx_] * integrals_CI / 16
            if self.T == 1:
                for prt in ('T', 'Ts', 'tz'):
                    sum_ += _CIDA_cst[prt][idx_] * integrals_CD / 4
                sum_ += _CIDA_cst['tzs'][idx_] * integrals_CA / 4
            
            me_rad  += sum_
        return me_ang * me_rad
    
    def _component_NN_L2(self):
        """ 
        L^2 term for NN short-intermediate phenomenological interaction
            <v(ang/spin)> * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
            
            # must be included all 
                CI = c, tau, sig, tau*sig
                CD = T, sig*T               (isotensor-dependent)
                CA = tauZ, sig*tauZ
        """
        me_ang = self._l*(self._l + 1)
        me_rad = 0.0
        S, T   = self.S_bra, self.T
        idx_   = 3 - (2*self.S_bra + self.T)
        ##             ST  11 10 01  00  
        _CIDA_cst = {'c': [9, 3,  3,  1], 't':   [3, -3,  1, -1], 
                     's': [3, 1, -3, -1], 'ts':  [1, -1, -1,  1],}
        
        for x in self.SubNNTermNNEnum.members():
            integrals_CI = 0.0
            
            int_ = self.getInteractionKeyword(self.TermEnum.l2, self.PartEnum.NN, x)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            integral = self.talmiIntegral(int_)
            integrals_CI = self.CONSTANTS_NN[int_][(S, T)] * integral
            
            sum_ = 0
            for prt in ('c', 't', 's', 'ts'):
                sum_ += _CIDA_cst[prt][idx_] * integrals_CI / 16
            
            me_rad  += sum_
        return me_ang * me_rad
    
    def _component_NN_tensor(self):
        """ 
        S12 term for NN short-intermediate phenomenological interaction
            <v(ang/spin)> * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
        """
        if self.S_bra == 0: return 0
        me_ang = self._get_tensor_angular_me()
        me_rad = 0.0
        S, T   = self.S_bra, self.T
        idx_   = 3 - (2*self.S_bra + self.T)
        ##             ST  11 10 01  00  
        _CIDA_cst = {'c': [3, 1,  0,  0], 't':   [1, -1, 0, 0], }
        
        for x in self.SubNNTermNNEnum.members():
            int_ = self.getInteractionKeyword(self.TermEnum.t, self.PartEnum.NN, x)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            integrals_CI = 0.0
            integral = self.talmiIntegral(int_)
            integrals_CI = self.CONSTANTS_NN[int_][(S, T)] * integral
            
            sum_  = 0
            sum_ += _CIDA_cst['c'][idx_] * integrals_CI / 4
            sum_ += _CIDA_cst['t'][idx_] * integrals_CI / 4
            me_rad  += sum_
        
        return me_ang * me_rad
    
    def _component_NN_LS(self):
        """
        L.S term for NN short-intermediate phenomenological interaction
            <v(ang/spin)> * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
        """
        if self.S_bra == 0: return 0
        j = self._j_rel
        me_ang = 0.5*( (j*(j + 1)) - (self._l*(self._l + 1)) - 2)   ## (S=1)^2=1*2
        
        me_rad = 0.0
        S, T   = self.S_bra, self.T
        idx_   = 3 - (2*self.S_bra + self.T)
        ##             ST  11 10 01  00  
        _CIDA_cst = {'c': [3, 1,  0,  0], 't':   [1, -1, 0, 0], }
        
        for x in self.SubNNTermNNEnum.members():
            int_ = self.getInteractionKeyword(self.TermEnum.lS, self.PartEnum.NN, x)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            integrals_CI = 0.0
            integral = self.talmiIntegral(int_)
            integrals_CI = self.CONSTANTS_NN[int_][(S, T)] * integral
            
            sum_  = 0
            sum_ += _CIDA_cst['c'][idx_] * integrals_CI / 4
            sum_ += _CIDA_cst['t'][idx_] * integrals_CI / 4
            me_rad  += sum_
        
        return me_ang * me_rad
    
    def _component_NN_LS2(self):
        """ 
        (L.S)^2 term for NN short-intermediate phenomenological interaction
            <v(ang/spin)> * (I_st*T^2 + (P_st + Q_st*(m*r) + T_st*(m*r)^2)*W(r))
        """
        if self.S_bra == 0: return 0
        me_ang = self._get_spinOrbitSquared_angular_me()
        me_rad = 0.0
        S, T   = self.S_bra, self.T
        idx_   = 3 - (2*self.S_bra + self.T)
        ##             ST  11 10 01  00  
        _CIDA_cst = {'c': [3, 1,  0,  0], 't':   [1, -1, 0, 0], }
        
        for x in self.SubNNTermNNEnum.members():
            int_ = self.getInteractionKeyword(self.TermEnum.lS2, self.PartEnum.NN, x)
            if self._SWITCH_OFF_TERMS[int_]: continue
            
            integrals_CI = 0.0
            integral = self.talmiIntegral(int_)
            integrals_CI = self.CONSTANTS_NN[int_][(S, T)] * integral
            
            sum_  = 0
            sum_ += _CIDA_cst['c'][idx_] * integrals_CI / 4
            sum_ += _CIDA_cst['t'][idx_] * integrals_CI / 4
            me_rad  += sum_
        
        return me_ang * me_rad
    
    # def _matrixElementCompoundEvaluation(self): Not needed to be overwriten

class ElectromagneticAv18TermsInteraction_JScheme(_BaseTalmiMultinteraction_JScheme):
    """
    Electromagnetic terms for the Argone 18 interaction:
        R.B.Wiringa, V.G.J. Stoks, R.Schiavilla
        Accurate nucleon-nucleon potential with charge-independence breaking
        Phys. Rev. C 51.1 (1-1994)
    
    Register the different pp/nn/pn channels for the :
    - pure Central 
        1-photon exchange, 2-photon exchange, Darwin-Foldy, Vacuum polarization
    - tensor - spin-orbit - spin - spin-antisymmetric
        Magnetic moment factor
    """
    _BREAK_ISOSPIN = True
    COUPLING = (CouplingSchemeEnum.JJ, )
    USE_EXACT_VACUUM_POLARIZATION = False
    
    _talmi_integrals = { 
        ## keys_: T (for OPE),  for NN, the ST channels are shared
        'c1' : [],
        'cpn': [],
        'c2' : [],
        'df' : [],
        'vp' : [],
        't'  : [],
        'lS' : [],
    }
    
    _tensor_ang_me = {} ## (lambda, lambda', L, l, l'): value
    _ls_ang_me     = {} ## (lambda, lambda', L, l, l'): value
    # _lsAnt_ang_me  = {} ## (lambda, lambda', L, l, l'): value
    
    #===========================================================================
    # CONSTANTS
    #===========================================================================
    
    class TermEnum(Enum):
        c1  = 'c1'
        c2  = 'c2'
        df  = 'df'
        vp  = 'vp'
        s   = 's'
        t   = 't'
        lS  = 'lS'
        cpn = 'cpn'
    
    CONSTANTS_ELECTRO = {
        'c1' : None, 'cpn': None, 'c2' : None, 'df': None, 'vp': None, 
        's': {},     't': {},     'lS': {}
        ## To be defined in the setInteractionParameters()
    }
    
    ## TESTING COMPONENTS  -------------------------
    _TEST_QUADRATURES     = True
    _SWITCH_OFF_CONSTANTS = False
    _SWITCH_OFF_TERMS     = {
        'c1' : False,  'cpn': False,  'c2' : False,  'df' : False,
        's'  : False,  'vp' : False,  't'  : False,  'lS' : False,
    }
    ##  --------------------------------------------
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
                
        cls.PARAMS_SHO[SHO_Parameters.b_length] = kwargs[SHO_Parameters.b_length]
        
        v18     = ConstantsV18
        e_cst   = Constants
        
        hbc     = v18.HBAR_C
        T_kin   = 10   ## Approx. 10 MeV in the nucleus. Change for other cases.
        alpha   = hbc * v18.ALPHA 
        alpha_q = alpha * np.sqrt((T_kin + 2*v18.M_PROTON) / (2*v18.M_PROTON) )
        
        
        ## ALPHA' is not has to be selected from a k fixed, whic
        cls.CONSTANTS_ELECTRO[cls.TermEnum.c1] = alpha_q
        cls.CONSTANTS_ELECTRO[cls.TermEnum.cpn]= alpha * 0.0189       ## article
        cls.CONSTANTS_ELECTRO[cls.TermEnum.c2] = - hbc * alpha_q * v18.ALPHA / v18.M_PROTON
        cls.CONSTANTS_ELECTRO[cls.TermEnum.df] = - hbc * alpha / ((2*v18.M_PROTON)**2)
        cls.CONSTANTS_ELECTRO[cls.TermEnum.vp] = 2 * alpha_q * v18.ALPHA / (3 * np.pi)
        
        ## NOTE_ Remember, In this code, m_t(protons) = +1 (in taurus is -1)
        ## MM constants pp
        aux = - hbc * alpha * (v18.MAGMOM_PROTON**2) / ((2 * v18.M_PROTON)**2)
        cls.CONSTANTS_ELECTRO[cls.TermEnum.s] [ 1] = aux * 2 / 3
        cls.CONSTANTS_ELECTRO[cls.TermEnum.t] [ 1] = aux
        aux = - alpha * (4*v18.MAGMOM_PROTON - 1) / (2 * v18.M_PROTON**2)
        cls.CONSTANTS_ELECTRO[cls.TermEnum.lS][ 1] = aux
        ## MM constants pn
        aux = - hbc * alpha * ((v18.MAGMOM_PROTON * v18.MAGMOM_NEUTRON) / 
                               (4 * v18.M_PROTON  * v18.M_NEUTRON))
        cls.CONSTANTS_ELECTRO[cls.TermEnum.s] [ 0] = aux * 2 / 3
        cls.CONSTANTS_ELECTRO[cls.TermEnum.t] [ 0] = aux
        m_nuc_red = 1 / ((1/v18.M_PROTON) + (1/v18.M_NEUTRON))
        aux = - hbc * alpha * v18.MAGMOM_NEUTRON / (2 * v18.M_PROTON * m_nuc_red)
        cls.CONSTANTS_ELECTRO[cls.TermEnum.lS][ 0] = aux
        ## MM constants nn
        aux = - hbc * alpha * (v18.MAGMOM_NEUTRON**2) / ((2 * v18.M_NEUTRON)**2)
        cls.CONSTANTS_ELECTRO[cls.TermEnum.s] [-1] = aux * 2 / 3
        cls.CONSTANTS_ELECTRO[cls.TermEnum.t] [-1] = aux
        cls.CONSTANTS_ELECTRO[cls.TermEnum.lS][-1] = 0
        
        print(_LINE_2,"  Default constants for [ElectromagneticAv18] interaction:")
        prettyPrintDictionary({'CONSTANTS_ELECTRO' : cls.CONSTANTS_ELECTRO, })
        print(f"     * Set interaction constants to 1 : [{cls._SWITCH_OFF_CONSTANTS}]")
        print(_LINE_2)
        
        ## testing ----------------------------------------------------------- 
        if cls._SWITCH_OFF_CONSTANTS: 
            for k in cls.CONSTANTS_ELECTRO:
                if isinstance(cls.CONSTANTS_ELECTRO[k], dict):
                    for k2 in cls.CONSTANTS_ELECTRO[k]:
                        cls.CONSTANTS_ELECTRO[k][k2] = 1
                else:
                    cls.CONSTANTS_ELECTRO[k] = 1
        
        dipole_len = 1 / 4.27     ## b parameter from the Form Factor length
        vp_length  = v18.HBAR_C / e_cst.M_ELECTRON
        
        cls.PARAMS_FORCE = {
            CentralMEParameters.mu_length  : dipole_len,  # for all the Form factors (shared)
            CentralMEParameters.opt_mu_2   : vp_length,   # a  diffuse length
        }
        
        cls._integrals_p_max = dict([(k,-1) for k in cls._talmi_integrals.keys()])
        cls._talmi_integrals = dict([(k, list()) for k in cls._talmi_integrals.keys()])
    
    def _nullConditionsOnParticleLabelStates(self):
        return True
    
    def _validKetTotalSpins(self):
        """
        Due the antisymmetric spin Orbit operator, there are <1|A|0>=-<0|A|1> != 0
        """
        return (0, 1) 
    #==========================================================================
    # # integrals
    #==========================================================================
    
    @classmethod
    def _integral_C1(cls, p, key_):
        """
        Do not have analytical expression, evaluates quadratures here
        """
        # poly2 = polynomialProduct([1, 3, 3], [1, 3, 3])
        poly2 = [1, 11/16, 3/16, 1/48]
        x, w  = getGeneralizedLaguerreRootsWeights(2*p)
        
        b  = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        
        A  = np.sqrt(2) * b / mu
        def __funct(x, A, n):
            return (np.exp(-A*np.sqrt(x)) * np.power(x, n/2) )
        I0 = np.exp(gamma_half_int(2*p + 2) - gamma_half_int(2*p + 3))
        I = 0
        for n, c_i in enumerate(poly2):
            I -= c_i * (A**(n)) * np.dot(w, __funct(x, A, n))
        I /= np.exp(gamma_half_int(2*p + 3))
        I  = (b**2 / np.sqrt(2)) * (I0 - I) 
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integral_C1_pn(cls, p, key_):
        poly2 = [0, 15/384, 15/384, 6/384, 4/384]
        x, w  = getGeneralizedLaguerreRootsWeights(2*p)
        
        b  = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        
        A  = np.sqrt(2) * b / mu
        def __funct(x, A, n):
            return (np.exp(-A*np.sqrt(x)) * np.power(x, n/2) )
        I = 0
        for n, c_i in enumerate(poly2):
            if n == 0: continue
            I += c_i * (A**(n)) * np.dot(w, __funct(x, A, n))
        I *= ( (b / mu)**2 / (np.sqrt(2) * np.exp(gamma_half_int(2*p + 3))) )
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integral_C2(cls, p, key_):
        # poly2 = polynomialProduct([1, 3, 3], [1, 3, 3])
        poly1 = [1, 11/16, 3/16, 1/48]
        poly2 = polynomialProduct(poly1, poly1)
        x, w  = getGeneralizedLaguerreRootsWeights(2*p)
        
        b  = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        
        A  = np.sqrt(2) * b / mu
        def __funct(x, A, n, c=1):
            return (np.exp(-A*c*np.sqrt(x)) * np.power(x, (n-1)/2) )
        
        I0 = 0.5 * (b**2) * np.exp(gamma_half_int(2*p + 1) - 
                                   gamma_half_int(2*p + 3))
        I1, I2 = 0, 0
        for n, c_i in enumerate(poly1):
            I1 += c_i * (A**(n)) * np.dot(w, __funct(x, A, n))
        for n, c_i in enumerate(poly2):
            I2 += c_i * (A**(n)) * np.dot(w, __funct(x, A, n, c=2))
        I = I0 + ((I2 - 2*I1) * (b**2) / (2 * np.exp(gamma_half_int(2*p + 3))) ) 
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integral_DarwinFoldy(cls, p, key_):
        poly2 = [0, 1/16, 1/16, 1/48]
        x, w  = getGeneralizedLaguerreRootsWeights(2*p + 1)
        
        b  = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        
        A  = np.sqrt(2) * b / mu
        def __funct(x, A, n):
            return (np.exp(-A*np.sqrt(x)) * np.power(x, n/2) )
        I = 0
        for n, c_i in enumerate(poly2):
            if n == 0: continue
            I += c_i * (A**n) * np.dot(w, __funct(x, A, n))
        I *= ( (b / mu)**3 / (np.exp(gamma_half_int(2*p + 3))) )
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integral_exact_VacuumPolarization(cls, p, key_):
        raise MatrixElementException("Not implemented, TODO!")
        return 0
    
    @classmethod
    def _integral_approx_VacuumPolarization(cls, p, key_):
        """
        Approximation of the vacuum polarization integral from:        
        [Auerbach, Hufner, Kerman, Shakin] A Theory of Isobaric Analog Resonances (1972)
        
        I ~ (-EurlerConst +5/6) + |ln(kr)| + (6pi/8)*kr + O(kr^2)
        
        with 2kr << 1. 
        """
        poly2 = [1, 11/16, 3/16, 1/48]
        x, w  = getGeneralizedLaguerreRootsWeights(2*p)
        
        b  = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        mu2= cls.PARAMS_FORCE[CentralMEParameters.opt_mu_2]
        assert mu2 > 200, "[WARNING] This approximation requires to be 2r/mu << 1!"
        
        A  = np.sqrt(2) * b / mu
        B  = np.sqrt(2) * b / mu2
        def __funct(x, A, n):
            return (np.exp(-A*np.sqrt(x)) * np.power(x, n/2) )
        def __funct2(x, A, B, n):
            _sqx = np.sqrt(x)
            return np.exp(-A*_sqx) * np.power(x, n/2) * np.abs(np.log(B*_sqx))
        
        D1, D2 = -0.5772 + (5/6), (6 * np.pi * B) / 8 
        
        ## Term constant
        I00 = np.exp(gamma_half_int(2*p + 2) - gamma_half_int(2*p + 3))
        I01 = 0
        for n, c_i in enumerate(poly2):
            I01 += c_i * (A**(n)) * np.dot(w, __funct(x, A, n))
        I01 /= np.exp(gamma_half_int(2*p + 3))
        I0   = D1 * (I00 - I01)
        
        ## Term logarithm
        I10 = np.dot(w, np.abs(np.log(B*np.sqrt(x))))
        I11 = 0
        for n, c_i in enumerate(poly2):
            I11 += c_i * (A**(n)) * np.dot(w, __funct2(x, A, B, n))
        I1  = (I10 - I11) / np.exp(gamma_half_int(2*p + 3))
        
        ## Term r
        x, w  = getGeneralizedLaguerreRootsWeights(2*p + 1)
        
        I20 = 1 ## (It is gamma_(p+3/2))
        I21 = 0
        for n, c_i in enumerate(poly2):
            I21 += c_i * (A**(n)) * np.dot(w, __funct(x, A, n))
        I2 =  D2 * (I20 - I21) / np.exp(gamma_half_int(2*p + 3))
        
        I = (b**2 / np.sqrt(2)) * (I0 + I1 + I2)
        return I
    
    @classmethod
    def _integral_VacuumPolarization(cls, p, key_):
        if cls.USE_EXACT_VACUUM_POLARIZATION:
            I = cls._integral_exact_VacuumPolarization (p, key_)
        else:
            I = cls._integral_approx_VacuumPolarization(p, key_)
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integral_Tensor(cls, p, key_):
        poly2 = [1, 1, 1/2, 1/6, 1/24, 1/144]
        x, w  = getGeneralizedLaguerreRootsWeights(2*p)
        
        b  = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        # p  = self._p 
        
        A  = np.sqrt(2) * b / mu
        def __funct(x, A, n):
            return (np.exp(-A*np.sqrt(x)) * np.power(x, n/2 - 1) ) 
        I = np.dot(w, 1/x)
        for n, c_i in enumerate(poly2):
            I -= c_i * (A**n) * np.dot(w, __funct(x, A, n))
        
        I  *= mu**3 / (2 * np.sqrt(2) * (np.exp( gamma_half_int(2*p + 3))) )
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _integral_SpinOrbit(cls, p, key_):
        """
        Do not have analytical expression, evaluates quadratures here 
        (same integral of the tensor term)
        """
        poly2 = [1, 1, 1/2, 7/48, 1/48]
        x, w  = getGeneralizedLaguerreRootsWeights(2*p)
        
        b  = cls.PARAMS_SHO  [SHO_Parameters.b_length]
        mu = cls.PARAMS_FORCE[CentralMEParameters.mu_length]
        # p  = self._p 
        
        A  = np.sqrt(2) * b / mu
        def __funct(x, A, n):
            return (np.exp(-A*np.sqrt(x)) * np.power(x, n/2 - 1) ) 
        I = np.dot(w, 1/x)
        for n, c_i in enumerate(poly2):
            I -= c_i * (A**n) * np.dot(w, __funct(x, A, n))
        
        I  *= mu**3 / (2 * np.sqrt(2) * (np.exp( gamma_half_int(2*p + 3))) )
        
        cls._talmi_integrals[key_].append(I)
    
    @classmethod
    def _calculateIntegrals(cls, key_,  n_integrals =1):
        """
        Select from each the interaction the Talmi integrals
        """
        for p in range(cls._integrals_p_max[key_] + 1, 
                       cls._integrals_p_max[key_] + n_integrals +1):
            
            if   (key_ == cls.TermEnum.c1):
                cls._integral_C1(p, key_)
            elif (key_ == cls.TermEnum.cpn):
                cls._integral_C1_pn(p, key_)
            elif (key_ == cls.TermEnum.c2):
                cls._integral_C2(p, key_)
            elif (key_ == cls.TermEnum.df):
                cls._integral_DarwinFoldy(p, key_)
            elif (key_ == cls.TermEnum.vp):
                cls._integral_VacuumPolarization(p, key_)
            elif (key_ == cls.TermEnum.t):
                cls._integral_Tensor(p, key_)
            elif (key_ == cls.TermEnum.lS):
                cls._integral_SpinOrbit(p, key_)
            
            else:
                raise MatrixElementException(f"not OPE or NN, got [{key_}]")
                        
            cls._integrals_p_max[key_] += 1
            
            
    
    #===========================================================================
    # ## _components
    #===========================================================================
    ## Angular - spin dependent common matrix elements
    def _get_key_angular_momentums(self, *args):
        return '_'.join([f"{x}" for x in args])
    
    def _get_tensor_angular_me(self):
        """ 
        Compute the angular factors in the COM scheme for Tensor interaction.
        """
        tupl = self._get_key_angular_momentums(self.L_bra, self.L_ket,
                                               self._L,    self._l,   self._l_q)
        if tupl in self._tensor_ang_me: 
            return self._tensor_ang_me.get(tupl)
        
        ## Breakpoint waring the dimension of the tensor_me dictionary
        assert self._tensor_ang_me.__len__() < 1e7, "I might be exploding. Fix me!"
        
        aux    = safe_wigner_6j(self._l,    self.L_bra, self._L, 
                                self.L_ket, self._l_q,  2)
        if self.isNullValue(aux):
            self._tensor_ang_me[tupl] = 0
            return 0
        
        me_ang  = aux * safe_clebsch_gordan(self._l, 2, self._l_q, 0, 0, 0)
        me_ang *= (
            10.954451150103322 * 
            np.sqrt((2*self.L_ket + 1)*(2*self.L_bra + 1)*(2*self._l + 1)) * 
            ((-1)**(self._l + self._L))
        )
        ## [2]*sqrt(24) = 2*sqrt(30) = 10.9545
        
        self._tensor_ang_me[tupl] = me_ang
        return self._tensor_ang_me.get(tupl)
    
    def _get_spinOrbit_angular_me(self):
        """
        Compute the angular factors in the COM scheme for LS/LA. 
        """
        tupl = self._get_key_angular_momentums(self.L_bra, self.L_ket,
                                               self._L,    self._l,    self._l_q)
        if tupl in self._ls_ang_me: 
            return self._ls_ang_me.get(tupl)
        ## Breakpoint to waring the dimension of the spinOrbit_me dictionary
        assert self._ls_ang_me.__len__() < 1e7, "I might be exploding. Fix me!"
        
        aux    = safe_wigner_6j(self._l,    self.L_bra, self._L, 
                                self.L_ket, self._l_q,  1)
        if self.isNullValue(aux):
            self._ls_ang_me[tupl] = 0
            return 0
        
        me_ang  = aux * np.sqrt(self._l * (self._l + 1) * (2*self._l + 1))
        me_ang *= (-1)**(self._l + self._L)
        
        self._ls_ang_me[tupl] = me_ang
        return self._ls_ang_me.get(tupl)
    
    def _component_central_1Photon(self):
        """ Contributes both for pp and nn channels, with different integrals """
        if   self.MT == 1:
            if self._SWITCH_OFF_TERMS[self.TermEnum.c1]:  return 0
            integral = self.talmiIntegral(self.TermEnum.c1)
            return self.CONSTANTS_ELECTRO[self.TermEnum.c1] * integral
        elif self.MT == 0:
            if self._SWITCH_OFF_TERMS[self.TermEnum.cpn]: return 0
            integral = self.talmiIntegral(self.TermEnum.cpn)
            return self.CONSTANTS_ELECTRO[self.TermEnum.cpn] * integral
        else:
            return 0
    
    def _component_central_2Photon(self):
        """ pp channel only. """
        if self._SWITCH_OFF_TERMS[self.TermEnum.c2]: return 0
        if self.MT != 1: 
            return 0
        integral = self.talmiIntegral(self.TermEnum.c2)
        return self.CONSTANTS_ELECTRO[self.TermEnum.c2] * integral
    
    def _component_central_DarwinFoldy(self):
        """ pp channel only. """
        if self._SWITCH_OFF_TERMS[self.TermEnum.df]: return 0
        if self.MT != 1: 
            return 0
        integral = self.talmiIntegral(self.TermEnum.df)
        return self.CONSTANTS_ELECTRO[self.TermEnum.df] * integral
    
    def _component_central_VacuumPolarization(self):
        """ pp channel only. 
            The evaluated integral can be evaluated exactly """
        if self._SWITCH_OFF_TERMS[self.TermEnum.vp]: return 0
        if self.MT != 1: 
            return 0
        if self.USE_EXACT_VACUUM_POLARIZATION:
            raise MatrixElementException("I'm not implemented yet!")
        
        integral = self.talmiIntegral(self.TermEnum.vp)
        return self.CONSTANTS_ELECTRO[self.TermEnum.vp] * integral
    
    def _component_spin_MagneticMoment(self):
        """ channel dependent. sigma(1)*sigma(2) """
        if self._SWITCH_OFF_TERMS[self.TermEnum.s]: return 0
        
        factor = (4*self.S_bra - 3)
        integral = self.talmiIntegral(self.TermEnum.df)  ## F_s == F_df
        return self.CONSTANTS_ELECTRO[self.TermEnum.s][self.MT] * integral * factor
    
    def _component_tensor_MagneticMoment(self):
        """ channel dependent. S12(1,2) """
        if self._SWITCH_OFF_TERMS[self.TermEnum.t]: return 0
        
        ang_me = self._get_tensor_angular_me()
        if self.isNullValue(ang_me): return 0
        
        aux = safe_wigner_6j(self.L_bra, self.S_bra, self.J, 
                             self.S_ket, self.L_ket, 2)
        if self.isNullValue(aux):
            return 0
        ang_me *= aux * (-1)**(self.S_ket + self.J + self.L_bra + self.L_ket)
        ang_me *= np.sqrt(2*self.J + 1)
        
        integral = self.talmiIntegral(self.TermEnum.t)
        return self.CONSTANTS_ELECTRO[self.TermEnum.t][self.MT] * integral * ang_me
        
    def _component_spinOrbit_MagneticMoment(self, antisymmetric_LS=False):
        """
        Contributes for the pp and pn channel. same integral. LS or LS+LA
        :antisymmetric_LS = False to evaluate the LS, True for LA (A=(s1-s2)/2)
        """
        if self._SWITCH_OFF_TERMS[self.TermEnum.lS]: return 0
        if self.MT == -1: return 0
        
        if antisymmetric_LS:
            spin_me = (-1)**(self.S_ket) / np.sqrt(2)
        else:
            spin_me = np.sqrt(6)
        
        ang_me = self._get_spinOrbit_angular_me()
        if self.isNullValue(ang_me): return 0
        
        aux = safe_wigner_6j(self.L_bra, self.S_bra, self.J, 
                             self.S_ket, self.L_ket, 1)
        if self.isNullValue(aux):
            return 0
        
        ang_me *= aux * spin_me
        ang_me *= np.sqrt((2*self.J + 1) * (2*self.L_bra + 1) * (2*self.L_ket + 1))
        ang_me *= (-1)**(self.S_bra + self.J + 1)
        
        integral = self.talmiIntegral(self.TermEnum.lS)
        return self.CONSTANTS_ELECTRO[self.TermEnum.lS][self.MT] * integral * ang_me
    
    
    def _matrixElementCompoundEvaluation(self):
        """
        Combination of the matrix element for the different modes and channels
        
        i.e. compound decompossition in relative j 
        """
        ## central CD-ope & NN + l2 terms
        sum_central = [0,]*5
        if (self.S_bra != self.S_ket):
            ## only case for the class 4 spin orbit tensor.
            return self._component_spinOrbit_MagneticMoment(antisymmetric_LS=True)
        
        if (self._l == self._l_q) and (self.L_bra == self.L_ket):
            sum_central[0] = self._component_central_1Photon()
            sum_central[1] = self._component_central_2Photon()
            sum_central[2] = self._component_central_DarwinFoldy()
            sum_central[3] = self._component_central_VacuumPolarization()
            
            sum_central[4] = self._component_spin_MagneticMoment()
        
        if self.S_bra != 1:      ## tensor/ls dependent have only S=1 components
            return sum(sum_central)        
        
        sum_noncentral = [0,]*2
        sum_noncentral[0]  = self._component_tensor_MagneticMoment()
        sum_noncentral[1]  = self._component_spinOrbit_MagneticMoment()
        
        return sum(sum_central) + sum(sum_noncentral)
    
