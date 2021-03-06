'''
Created on Mar 8, 2021

@author: Miguel
'''
import numpy as np
import pandas as pd

from matrix_elements.MatrixElement import MatrixElementException,\
    _TwoBodyMatrixElement
from helpers.WaveFunctions import QN_2body_L_Coupling, QN_1body_radial

from helpers.Enums import CouplingSchemeEnum, SHO_Parameters,\
    CentralMEParameters, PotentialForms, OUTPUT_FOLDER

from helpers.Helpers import angular_condition, fact
from helpers.integrals import talmiIntegral
from matrix_elements.BM_brackets import BM_Bracket
from helpers.Log import XLog

#===============================================================================
# moved there due circular import (after angular_condition method)
from helpers.Helpers import _B_coeff_memo_accessor

# B coefficients stored in Memory are not b_length normalized (divide by b**3)
_B_Coefficient_Memo = {}

#===============================================================================

class _TalmiTransformationBase(_TwoBodyMatrixElement):
    
    COUPLING = CouplingSchemeEnum.L
    DEBUG_MODE = False
    _BREAK_ISOSPIN = False
    
    def __init__(self, bra, ket, run_it=True):
        """
        This method allow instance for radial wave functions when calculating
        in this basis alone (No JJ coupling).
        
            <(n1,l1) (n2,l2) (LS)| V | (n1,l1)' (n2,l2)' (L'S')>
        
        When using JJ coupling, implement this class in 2nd place hierarchy after 
        _TwoBodyMatrixElement_JTCoupled (that defines J, T and sp jj objects)
        
        :bra    <QN_2body_L_Coupling>
        :ket    <QN_2body_L_Coupling>
        """
        
        self.__checkInputArguments(bra, ket)
        
        self.bra = bra
        self.ket = ket
        
        # set L state as protected to fix with 2BME_JT coupled attributes
        self._L_bra = bra.L
        self._L_ket = ket.L
        
        ## protected internal variables for the iterations
        self._n = None
        self._l = None
        self._N = None
        self._L = None
        
        # special ket quantum numbers
        self._n_q = None 
        self._l_q = None 
        
        self._p = None
        
        if not self.isNullMatrixElement and run_it:
            # evaluate the normal and anti_symmetrized m.e.
            self._run()
    
    
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
        """
        # Refresh the Force parameters
        if cls.PARAMS_FORCE:
            cls.PARAMS_FORCE = {}
        
        params_and_defaults = {
            SHO_Parameters.b_length       : 1,
            SHO_Parameters.hbar_omega     : 1,
            CentralMEParameters.potential : None,
            CentralMEParameters.mu_length : None,
            CentralMEParameters.constant  : 1,
            CentralMEParameters.n_power   : 0
        }
        
        for param, default in params_and_defaults.items():

            value = kwargs.get(param)
            if value is None:
                if default is None:
                    raise MatrixElementException(
                        "Required parameter [{}]: got None".format(param))
                value = default
            
            if (param == CentralMEParameters.potential):
                value = value.lower()
                if (value not in PotentialForms.members()):
                    raise MatrixElementException("Potential name is not defined "
                        "in PotentialForms Enumeration, got: [{}]".format(value))

            if param in SHO_Parameters.members():
                # if param == SHO_Parameters.b_length:
                #     # In the center of mass system, b_length = b_length / sqrt_(2)
                #     value *= np.sqrt(2) 
                #     ## NO b_rel = sqrt_(2)*b !!, transformation taken into account
                #     ## for the Talmi integrals and the B_nlp coefficients
                cls.PARAMS_SHO[param] = value
            else:
                cls.PARAMS_FORCE[param] = value
        
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = []
        
    
    #===========================================================================
    # % PROPERTIES
    #===========================================================================
    
    @property
    def rho_bra(self):
        if not hasattr(self, '_rho_bra'):
            self._rho_bra = 2*(self.bra.n1 + self.bra.n2) + self.bra.l1 + self.bra.l2
        return self._rho_bra
        
    @property
    def rho_ket(self):
        if not hasattr(self, '_rho_ket'):
            self._rho_ket = 2*(self.ket.n1 + self.ket.n2) + self.ket.l1 + self.ket.l2
        return self._rho_ket
    
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
        
    #===========================================================================
    # % SHARED METHODS
    #===========================================================================
    
    def __checkInputArguments(self, bra, ket):
        if not isinstance(bra, QN_2body_L_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_L_Coupling>")
        if not isinstance(ket, QN_2body_L_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_L_Coupling>")
    
    def _run(self):
        
        if len(self.__class__.__bases__) > 1:
            raise MatrixElementException(
                "You are making use of Talmi transformation from another matrix"
                " element definition (multiple inheritance): {}, \n" 
                "<TalmiTransformation>._run() is only valid to calculate m.e. "
                "directly <(n1l1)(n2l2)(L) |V[overwritable] |(n1l1)'(n2l2)'(L')>."
                "Otherwise, this class is helper for the main matrix element.\n"
                "Then, change the order of the parent classes for <TalmiTransformation>"
                " to be a helper ".format(self.__class__.__bases__))
        
        if not self.isNullMatrixElement:
            
            self._value = self.centerOfMassMatrixElementEvaluation()
    
    @classmethod
    def _calculateIntegrals(cls, n_integrals =1):
        
        args = [
            cls.PARAMS_FORCE.get(CentralMEParameters.potential),
            cls.PARAMS_SHO.get(SHO_Parameters.b_length),
            cls.PARAMS_FORCE.get(CentralMEParameters.mu_length),
            cls.PARAMS_FORCE.get(CentralMEParameters.n_power),
        ]
        
        for p in range(cls._integrals_p_max + 1, cls._integrals_p_max + n_integrals +1):
            
            cls._talmiIntegrals.append(talmiIntegral(p, *args))
            
            cls._integrals_p_max += 1
            
    
    def talmiIntegral(self):
        """ 
        Get or update Talmi integrals for the calculations
        """
        if self._p > self._integrals_p_max:
            self._calculateIntegrals(n_integrals = max(self.rho_bra, self.rho_ket, 1))
        return self._talmiIntegrals.__getitem__(self._p)
    
    @staticmethod
    def BCoefficient(n, l, n_q, l_q, p, b_param=1):
        """ 
        @static method to access B(n,l, n', l', p) coefficients. 
        Testing purposes, Do not use it inner calculations
        """
        
        _dummy_me = _TalmiTransformationBase(
            QN_2body_L_Coupling(QN_1body_radial(0, 0, mt=1), 
                                QN_1body_radial(0, 0, mt=1), 0),
            QN_2body_L_Coupling(QN_1body_radial(0, 0, mt=1), 
                                QN_1body_radial(0, 0, mt=1), 0),
            run_it = False)
        
        _dummy_me._n   = n
        _dummy_me._l   = l
        _dummy_me._n_q = n_q
        _dummy_me._l_q = l_q
        _dummy_me._p   = p
        
        return _dummy_me._B_coefficient_evaluation() / (b_param**3)
    
    
    def _B_coefficient(self, b_param=None):
        """ _Memorization Pattern for B coefficients. """
        
        if not b_param:
            b_param = self.PARAMS_SHO[SHO_Parameters.b_length]
        
        tpl   = (self._n, self._l, self._n_q, self._l_q, self._p)
        tpl = _B_coeff_memo_accessor(*tpl)
        
        global _B_Coefficient_Memo
        
        if not tpl in _B_Coefficient_Memo:
            _B_Coefficient_Memo[tpl] = self._B_coefficient_evaluation()

        return _B_Coefficient_Memo[tpl] / (b_param**3)
    
    def _B_coefficient_evaluation(self):
        """ SHO normalization coefficients for WF, not b_length dependent """
        
        # parity condition
        if(((self._l + self._l_q)%2)!=0):
            return 0
        
        const = 0.5*((fact(self._n) + fact(self._n_q)
                      + fact(2*(self._n + self._l) + 1)
                      + fact(2*(self._n_q + self._l_q) + 1))
                      - 
                      (fact(self._n + self._l)
                       + fact(self._n_q + self._l_q))
                    )
        
        const += fact(2*self._p + 1) - fact(self._p)
        
        const = (-1)**(self._p - (self._l + self._l_q)//2) * np.exp(const)
        const /= (2**(self._n + self._n_q))
        
        aux_sum = 0.
        max_ = min(self._n, self._p - (self._l + self._l_q)//2)
        min_ = max(0, self._p - (self._l + self._l_q)//2 - self._n_q)
        
        for k in range(min_, max_ +1):
            const_k = ((fact(self._l + k) 
                      + fact(self._p - k - (self._l - self._l_q)//2))
                       -
                     (fact(k) + fact(self._n - k) + fact(2*(self._l + k) + 1)
                      + fact(self._p - (self._l + self._l_q)//2 - k) 
                      + fact(self._n_q - self._p + k + (self._l + self._l_q)//2)
                      + fact(2*(self._p - k) + self._l_q - self._l + 1))
                    )
            aux_sum += np.exp(const_k)
        
        return const * aux_sum
    
    #===========================================================================
    # % ABSTRACT METHODS
    #===========================================================================
    
    def _interactionConstantsForCOM_Iteration(self):
        """ 
        Coefficients n,l, N, L dependent for the intern sum and the interaction.
        """
        raise MatrixElementException("Abstract Method, Implement me!")
        return
    
    def _globalInteractionCoefficient(self):
        """ 
        Coefficient non dependent on the COM qqnn series ex:(L,L', S,S', JT ...)
        """
        raise MatrixElementException("Abstract Method, Implement me!")
        return 
    
    def _deltaConditionsForCOM_Iteration(self):
        """
        Define if non null requirements on LS coupled J(and T) Matrix Element, 
        while doing the center of mass decomposition (n,l,n',l', N, L indexes 
        involved). 
        
        TODO: Check parity depending on the total wave function (J,T dependence)
        
        return True when conditions are fulfilled.
        """
        raise MatrixElementException("Abstract Method, Implement me!")
    
    def _interactionSeries(self):
        """
        Final Method.!!
        C_[X](n1,n2, l1,l2, (n1,n2, l1,l2)' L, L', ..., p)
        carry valid values, call delta and common constants
        """
        raise MatrixElementException("Series to be implemented here from "
                                     "secureIter or minimalIter classes")
        return
        
    
    def _BrodyMoshinskyTransformation(self):
        """
        ##  WARNING: This method is final. Do not overwrite!!
        
        Sum over valid p-Talmi integrals range (Energy + Ang. momentum + parity 
        conservation laws). Call implemented Talmi Coefficients for 
        Brody-Moshinsky transformation
        """
        
        sum_ = 0.0
        if self.DEBUG_MODE:
            XLog.write('talmi')
        for p in range(max(self.rho_bra, self.rho_ket) +1):
            if self.DEBUG_MODE:
                XLog.write('talmi', p=p)
            self._p = p
            # 2* from the antisymmetrization_ (_deltaConditionsForCOM_Iteration)
            series = 2 * self._interactionSeries()
            Ip =     self.talmiIntegral()
            product = series * Ip
            sum_ += product
            # sum_ += self._interactionSeries() * self.talmiIntegral()
            
            if self.DEBUG_MODE:
                XLog.write('talmi', series = series, Ip=Ip, val=product)
        return self._globalInteractionCoefficient() * sum_
    
    def centerOfMassMatrixElementEvaluation(self):
        """
        Matrix Element to be called for a defined interaction.
            <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        Use:
            1. Define deltas or factors over global conditions
            2. Define variables for the Moshinski transformation.
            3. Instance the Talmi Integrals in the limits/interaction
        """
        raise MatrixElementException("Abstract Method, require implementation")
    
    def deltaConditionsForGlobalQN(self):
        """
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        return True when conditions are fulfilled.
        
        Acts on (n1,l1)(n2,l2) (L_t,S) (n1,l1)'(n2,l2)'(L_t', S') (JT) qu. Numbers
        """
        raise MatrixElementException("Abstract Method, require implementation")


    def _debbugingTable(self, bmb_bra, bmb_ket, com_const, b_coeff):
        """
        Fill a table with calculated elements. (implement for other forces)
        """
        if not hasattr(self, '_debbugingTableDF'):
            columns = ['com_qqnn', 'com_constant', 'bmb_bra', 'bmb_ket', 'bmbs', 
                       *['p'+str(p) for p in range(max(self.rho_bra,self.rho_ket)+1)]]
            
            self._debbugingTableDF = pd.DataFrame(columns=columns)
        
        p_value =  b_coeff* bmb_bra*bmb_ket*com_const
        key = ' '.join([str(i) for i in (self._n, self._l, self._N, self._L)])
        
        row = {
            'com_qqnn' : key + str((self._n_q, self._l_q)),
            'com_constant' : com_const,
            'bmb_bra' : bmb_bra, 
            'bmb_ket': bmb_ket, 
            'bmbs': bmb_bra*bmb_ket,
            'p'+str(self._p): p_value
        }
        
        if row['com_qqnn'] in self._debbugingTableDF.com_qqnn.values:
            i = self._debbugingTableDF[self._debbugingTableDF['com_qqnn']==row['com_qqnn']]
            i = i.index.values
            
            self._debbugingTableDF.at[i, 'p'+str(self._p)] = p_value
        else:
            self._debbugingTableDF = self._debbugingTableDF.append(row, 
                                                                   ignore_index=True)
            
    
    def getDebuggingTable(self, filename_export=None):
        """ save debugging table for analysis"""
        self._debbugingTableDF = self._debbugingTableDF.fillna('-')
        
        if filename_export:
            self._debbugingTableDF.to_csv(OUTPUT_FOLDER + filename_export)
        return self._debbugingTableDF
        
        

class _TalmiTransformation_SecureIter(_TalmiTransformationBase):
    
    """
    Explicit run over all possible p, with the evaluation of every compatible
    N,L, n,n', l,l' under verifications on parity and angular momentum
    """
    
    def _validCOM_Bra_qqnn(self):
                
        rho = self.rho_bra
        lambda_ = self._L_bra
        ## Notation: big_Lambda = L + l, big_N = N + n
        
        
        rho_lambda_dont_pair = (rho - lambda_) % 2
        
        valid_qqnn = []
        big_Lambda_min = lambda_
        
        if rho_lambda_dont_pair:
            if (lambda_ == 0):         #lambda_ = 0 case for parity
                return valid_qqnn
            else:
                big_Lambda_min += 1
        
        # the order is fixed to be similar to BMosh. book benchmark
        for n in range((rho//2) +1):
            for N in range((rho//2) - n, -1, -1):
                big_N = N + n
                big_Lambda = rho - (2*big_N)
                
                if big_Lambda < lambda_:
                    # ensure angular condition on L+l >= lambda_
                    continue
                
                l_min = lambda_ + ((big_Lambda - big_Lambda_min)//2)
                for l in range(big_Lambda - l_min, l_min +1, +1):
                    L = rho - l - (2*big_N)
                    
                    if not angular_condition(l, L, lambda_):
                        _ = 0
                        continue 
                    elif self.parity_bra + (L + l)% 2 == 1:
                        _ = 1
                        continue
                    valid_qqnn.append((n, l, N, L))
        
        # TODO: Comment when not debugging
        valid_qqnn = sorted(valid_qqnn, 
                            key=lambda x: 1000*x[0]+100*x[1]+10*x[2]+x[3])
        
        return valid_qqnn
    
    def _validKet_relativeAngularMomentums(self):
        """ 
        get valid l' qqnns (as tuple) in the c.o.m for the bra wave function
        """
        raise MatrixElementException("Abstract Method, implement according to force.")
        return
    
    def _interactionSeries(self):
        
        sum_ = 0.0
        if self.DEBUG_MODE:
            XLog.write('intSer')
        
        for qqnn_bra in self._validCOM_Bra_qqnn():
            
            self._n, self._l, self._N, self._L = qqnn_bra
            
            bmb_bra = BM_Bracket(self._n, self._l, self._N, self._L, 
                                 self.bra.n1, self.bra.l1, self.bra.n2, self.bra.l2, 
                                 self._L_bra)
            if self.isNullValue(bmb_bra):
                continue
            
            for l_q in self._validKet_relativeAngularMomentums():
                self._l_q = l_q
                
                self._n_q  = self._n
                self._n_q += (self.rho_ket - self.rho_bra + self._l  - self._l_q) // 2
                
                if self._n_q < 0 or not self._deltaConditionsForCOM_Iteration():
                    continue
                
                bmb_ket = BM_Bracket(self._n_q, self._l_q, self._N, self._L, 
                                     self.ket.n1, self.ket.l1, self.ket.n2, self.ket.l2, 
                                     self._L_ket)
                if self.isNullValue(bmb_ket):
                    continue
                
                _b_coeff = self._B_coefficient(self.PARAMS_SHO.get(SHO_Parameters.b_length))
                _com_coeff = self._interactionConstantsForCOM_Iteration()
                
                aux =  _com_coeff * bmb_bra * bmb_ket * _b_coeff
                sum_ += aux
                
                # TODO: comment when not debugging
                if self.DEBUG_MODE:
                    XLog.write('intSer', bmbs=bmb_bra * bmb_ket, B=_b_coeff,
                               comCoeff=_com_coeff, aux=aux)
                #     self._debbugingTable(bmb_bra, bmb_ket, _com_coeff, _b_coeff)
        if self.DEBUG_MODE:
            XLog.write('intSer', value=sum_)
        return sum_
        
        
    

class _TalmiTransformation_MinimalIter(_TalmiTransformationBase):
    
    """
    Algotithm to run over only valid N,L, n,n', l,l' and p obtained directly, 
    more efficient with less loops and conditionals.
    """
    
    def _interactionSeries(self):
        """
        Final Method.!!
        C_[X](n1,n2, l1,l2, (n1,n2, l1,l2)' L, L', ..., p)
        carry valid values, call delta and common constants
        """
        bmb_bra = BM_Bracket(self._n, self._l, 
                             self._N, self._L, 
                             self.bra.n1, self.bra.l1, 
                             self.bra.n2, self.bra.l2, 
                             self._L_bra)
        if self.isNullValue(bmb_bra):
            return 0.0
        
        bmb_ket = BM_Bracket(self._n_q, self._l_q,
                             self._N, self._L,
                             self.ket.n1, self.ket.l1,
                             self.ket.n2, self.ket.l2,
                             self._L_ket)
        if self.isNullValue(bmb_ket):
            return 0.0
        
        _b_coeff    = self._B_coefficient()
        _com_coeff = self._interactionConstantsForCOM_Iteration()
        
        # TODO: comment when not debugging
        # if self.DEBUG_MODE:
        #     self._debbugingTable(bmb_bra, bmb_ket, _com_coeff, _b_coeff)
        
        return _com_coeff * bmb_bra * bmb_ket * _b_coeff
    
    def _getBigLambdaMinimum(self, lambda_, rho):
        
        big_Lambda_min = lambda_ 
             
        if (rho - lambda_) % 2:
            if (lambda_ == 0):
                return None
            else:
                big_Lambda_min += 1
                
        return big_Lambda_min
    
    def _BrodyMoshinskyTransformation(self):
        """
        ##  WARNING: This method is final. Do not overwrite!!
        
        Sum over valid p-Talmi integrals range (Energy + Ang. momentum + parity 
        conservation laws). Call implemented Talmi Coefficients for 
        Brody-Moshinsky transformation
        """
        ## TODO: Not Tested
        
        ## Notation: big_Lambda = L + l, big_N = N + n
        rho = self.rho_bra
        lambda_ = self._L_bra
        rho_q   = self.rho_ket
        lambda_q = self._L_ket
        
        big_Lambda_min = self._getBigLambdaMinimum(lambda_, rho)
        big_Lambda_q_min = self._getBigLambdaMinimum(lambda_q, rho_q)
        
        if (big_Lambda_min is None) or (big_Lambda_q_min is None):
            return 0.0
        
        sum_ = 0.0
        for big_N in range((rho - big_Lambda_min)//2 +1):
            for big_N_q in range((rho_q - big_Lambda_q_min)//2 +1):
                
                big_Lambda = rho - 2*big_N
                big_Lambda_q = rho_q - 2*big_N_q
                
                l_max = min(lambda_ + (big_Lambda - big_Lambda_min)//2,
                            lambda_q + (big_Lambda_q - big_Lambda_q_min)//2)
                l_min = big_Lambda - l_max 
                
                if (l_max - l_min) > lambda_ or (l_max < 0):
                    # Maybe unnecessary, add breakpoint
                    _ = 0
                    continue
                
                for N in range(big_N +1):
                    self._N = N
                    
                    for l in range(l_min, l_max +1):#big_Lambda + l_min +1):
                        
                        self._l = l
                        self._n = big_N - N
                        self._L = big_Lambda - l
                        self._l_q = big_Lambda_q - big_Lambda + l
                        self._n_q = self._n + (rho - rho_q + l - self._l_q)//2
                        
                        aux = (N, self._L, self._n, l, (self._n_q, self._l_q))
                        if ((self._n_q < 0) or (self._l_q < 0) 
                            or not self._deltaConditionsForCOM_Iteration()):
                            #  Maybe unnecessary, add breakpoint
                            _ = 0
                            continue
                        
                        p_min = (l + self._l_q)//2
                        p_max = self._n+self._n_q + (l+self._l_q)//2
                        for p in range(p_min, p_max +1):
                            self._p = p
            
                            sum_ += self.talmiIntegral() * self._interactionSeries()
                        
            
        return self._globalInteractionCoefficient() * sum_
    

# class TalmiTransformation(_TalmiTransformation_MinimalIter):
class TalmiTransformation(_TalmiTransformation_SecureIter):
    
    """ 
    Methods to implement the Brody Moshinsky transformation for different 
    force types, and the Talmi integral of the radial potential in the relative 
    mass coordinates
    """
    pass
    
