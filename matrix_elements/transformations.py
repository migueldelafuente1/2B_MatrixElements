'''
Created on Mar 8, 2021

@author: Miguel
'''
import numpy as np

from matrix_elements.MatrixElement import MatrixElementException,\
    _TwoBodyMatrixElement
from helpers.WaveFunctions import QN_2body_L_Coupling, QN_1body_radial

from helpers.Enums import CouplingSchemeEnum, SHO_Parameters,\
    CentralMEParameters, CentralGeneralizedMEParameters, PotentialForms,\
    OUTPUT_FOLDER

from helpers.Helpers import angular_condition, fact
from helpers.integrals import talmiIntegral
from matrix_elements.BM_brackets import BM_Bracket
from helpers.Log import XLog

#===============================================================================
# moved there due circular import (after angular_condition method)
from helpers.Helpers import _B_coeff_memo_accessor
from helpers import MATPLOTLIB_INSTALLED, PANDAS_INSTALLED
if PANDAS_INSTALLED:
    import pandas as pd

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
        self.L_bra = bra.L
        self.L_ket = ket.L
        
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
            CentralMEParameters.n_power   : 0,
            CentralMEParameters.opt_mu_2  : 1,
            CentralMEParameters.opt_mu_3  : 1,
            CentralMEParameters.opt_cutoff: 0,
        }
        
        for param, default in params_and_defaults.items():

            value = kwargs.get(param)
            if value is None:
                if param.startswith('opt_'):
                    continue
                    ## Parameters for the 
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
        
        ## Wood-Saxon fixing of mu_2 = r0 * A^-1/3
        if kwargs[CentralMEParameters.potential] == PotentialForms.Wood_Saxon:
            A = 1
            kwargs[CentralMEParameters.opt_mu_2] *= A**(1/3)
        
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
        
        arg_keys = [
            CentralMEParameters.potential, 
            SHO_Parameters     .b_length,
            CentralMEParameters.mu_length,
            CentralMEParameters.n_power
        ]
        
        args = [ 
            cls.PARAMS_FORCE.get(arg_keys[0]), 
            cls.PARAMS_SHO  .get(arg_keys[1]),
            cls.PARAMS_FORCE.get(arg_keys[2]), 
            cls.PARAMS_FORCE.get(arg_keys[3]),
        ]
        kwargs = map(lambda x: (x, cls.PARAMS_FORCE.get(x, None)), 
                     CentralMEParameters.members(but=arg_keys))
        kwargs = dict(filter(lambda x: x[1] != None, kwargs))
        
        for p in range(cls._integrals_p_max + 1, cls._integrals_p_max + n_integrals +1):
            
            cls._talmiIntegrals.append(talmiIntegral(p, *args, **kwargs))
            
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
    
    
    def _B_coefficient(self, b_param=None, com_qqnn=False):
        """ _Memorization Pattern for B coefficients. """
        
        if not b_param:
            b_param = self.PARAMS_SHO[SHO_Parameters.b_length]
        
        if com_qqnn:
            tpl_0   = (self._N, self._L, self._N_q, self._L_q, self._q)
        else: ## relative qqnn
            tpl_0   = (self._n, self._l, self._n_q, self._l_q, self._p)
        
        ## NOTE: the accessor sort the (n,l) > (n',l') to not evaluate it 
        ## twice the element, since B(n,l,n',l', p) = B(n',l',n,l, p)
        tpl = _B_coeff_memo_accessor(*tpl_0)
        
        global _B_Coefficient_Memo
        
        if not tpl in _B_Coefficient_Memo:
            _B_Coefficient_Memo[tpl] = self._B_coefficient_evaluation(*tpl_0)
        return _B_Coefficient_Memo[tpl] / (b_param**3)
    
    def _B_coefficient_evaluation(self, n, l, n_q, l_q, p):
        """ 
        SHO normalization coefficients for WF, not b_length dependent 
        """
        # parity condition
        if(((l + l_q)%2)!=0):
            return 0
        
        const = 0.5*((fact(n) + fact(n_q)
                      + fact(2*(n + l) + 1)
                      + fact(2*(n_q + l_q) + 1))
                      - 
                      (fact(n + l)
                       + fact(n_q + l_q))
                    )
        
        const += fact(2*p + 1) - fact(p)
        
        const = (-1)**(p - (l + l_q)//2) * np.exp(const)
        const /= (2**(n + n_q))
        
        aux_sum = 0.
        max_ = min(n, p - (l + l_q)//2)
        min_ = max(0, p - (l + l_q)//2 - n_q)
        
        for k in range(min_, max_ +1):
            const_k = ((fact(l + k) 
                      + fact(p - k - (l - l_q)//2))
                       -
                     (fact(k) + fact(n - k) + fact(2*(l + k) + 1)
                      + fact(p - (l + l_q)//2 - k) 
                      + fact(n_q - p + k + (l + l_q)//2)
                      + fact(2*(p - k) + l_q - l + 1))
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
        return
    
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
    
    def _validCOM_qqnn(self, elem):
        
        if elem == 'bra':  
            rho = self.rho_bra
            lambda_ = self.L_bra
            parity_ = self.parity_bra
        elif elem == 'ket':
            rho = self.rho_ket
            lambda_ = self.L_ket
            parity_ = self.parity_ket
        else:
            raise MatrixElementException("Set elem as 'bra' or 'ket', stopping")
        
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
                        continue 
                    elif parity_ + (L + l)% 2 == 1:
                        continue
                    valid_qqnn.append((n, l, N, L))
        
        # TODO: Comment when not debugging
        # valid_qqnn = sorted(valid_qqnn, 
        #                     key=lambda x: 1000*x[0]+100*x[1]+10*x[2]+x[3])
        
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
        
        for qqnn_bra in self._validCOM_qqnn('bra'):
            
            self._n, self._l, self._N, self._L = qqnn_bra
            
            bmb_bra = BM_Bracket(self._n, self._l, self._N, self._L, 
                                 self.bra.n1, self.bra.l1, self.bra.n2, self.bra.l2, 
                                 self.L_bra)
            if self.isNullValue(bmb_bra):
                continue
            if self.DEBUG_MODE:
                XLog.write('intSer', nlNL=qqnn_bra)
            
            for l_q in self._validKet_relativeAngularMomentums():
                self._l_q = l_q
                
                self._n_q  = self._n
                self._n_q += (self.rho_ket - self.rho_bra + self._l  - self._l_q) // 2
                
                if self._n_q < 0 or not self._deltaConditionsForCOM_Iteration():
                    continue
                
                bmb_ket = BM_Bracket(self._n_q, self._l_q, self._N, self._L, 
                                     self.ket.n1, self.ket.l1, self.ket.n2, self.ket.l2, 
                                     self.L_ket)
                if self.isNullValue(bmb_ket):
                    continue
                
                _b_coeff = self._B_coefficient(self.PARAMS_SHO.get(SHO_Parameters.b_length))
                _com_coeff = self._interactionConstantsForCOM_Iteration()
                
                aux =  _com_coeff * bmb_bra * bmb_ket * _b_coeff
                sum_ += aux
                
                # TODO: comment when not debugging
                if self.DEBUG_MODE:
                    XLog.write('intSer_ket', nq=self._n_q, lq=self._l_q, 
                               bmbs=bmb_bra * bmb_ket, 
                               B=_b_coeff, comCoeff=_com_coeff, aux=aux)
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
                             self.L_bra)
        if self.isNullValue(bmb_bra):
            return 0.0
        
        bmb_ket = BM_Bracket(self._n_q, self._l_q,
                             self._N, self._L,
                             self.ket.n1, self.ket.l1,
                             self.ket.n2, self.ket.l2,
                             self.L_ket)
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
        lambda_ = self.L_bra
        rho_q   = self.rho_ket
        lambda_q = self.L_ket
        
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
    @staticmethod
    def _potentialShapes(r, potential, mu, n_pow=0):
        """
        functions V(r) for the central potentials
        """
        x = r/mu
        if potential == PotentialForms.Gaussian:
            return np.exp(- np.power(x, 2))
        elif potential == PotentialForms.Gaussian_power:
            return np.exp(- np.power(x, 2)) / np.power(x, n_pow)
        elif potential == PotentialForms.Power:
            return np.power(x, n_pow)
        elif potential == PotentialForms.Coulomb:
            return 1 / x
        elif potential == PotentialForms.Exponential:
            return np.exp(- x)
        elif potential == PotentialForms.Yukawa:
            return np.exp(- x) / (x)
        else:
            raise MatrixElementException(
                "Not implemented Potential option ::" + potential)
    
    @classmethod
    def plotRadialPotential(cls, save_pdf=False):
        """ 
        Auxiliary function to plot the central parameters given.
        """
        if not MATPLOTLIB_INSTALLED:
            return
        import matplotlib.pyplot as plt
        dr = 0.001
        r = np.arange(dr, 5 + dr, 0.001) 
        
        fig = plt.figure(figsize=(7, 7))
        ax  = fig.add_subplot(1,1,1)
        
        plt.subplots_adjust(bottom=0.1, left=0.1)
        ax.set_title("Central Potential Shape V(r)")
        ax.set_ylabel("V(r) [MeV]")
        ax.set_xlabel("r    [fm]")
        
        y_total = np.zeros(len(r))
        c_min, c_max, r_max, r_max_aux = 0, 1, max(r), 0
        if isinstance(list(cls.PARAMS_FORCE.values())[0], dict):
            y = []
            for consts in cls.PARAMS_FORCE.values():
                args = [
                    consts.get(CentralMEParameters.potential),
                    consts.get(CentralMEParameters.mu_length),
                    consts.get(CentralMEParameters.n_power, '_')
                    ]
                V_0 = consts.get(CentralMEParameters.constant, 1)
                c_max, c_min = max(c_max, V_0), min(c_min, V_0)
                r_max_aux = max(r_max_aux, args[1])
                str_lab = '{0} (C:{3:.1f}, mu:{1:.2f}, N:{2})'.format(*args,V_0)
                
                y.append(TalmiTransformation._potentialShapes(r, *args))
                y[-1] *= V_0
                plt.plot(r, y[-1], '--', label=str_lab)
                
                y_total += y[-1]
            r_max = min(r_max, r_max_aux)
        else:
            args = [
                    cls.PARAMS_FORCE.get(CentralMEParameters.potential),
                    cls.PARAMS_FORCE.get(CentralMEParameters.mu_length),
                    cls.PARAMS_FORCE.get(CentralMEParameters.n_power, '_')
                    ]
            V_0 = cls.PARAMS_FORCE.get(CentralMEParameters.constant, 1)
            c_max, c_min = max(c_max, V_0), min(c_min, V_0)
            r_max = min(r_max, 2.5*args[1])
            str_lab = '{0}(C:{3}, mu:{2}, N:{1})'.format(*args, V_0)
            
            y_total  = TalmiTransformation._potentialShapes(r, *args)
            y_total *= V_0
            plt.plot(r, y_total, 'r-', label=str_lab)
        
        plt.plot(r, y_total, 'k-', label='Total V(r)')
        ax.axis([0, r_max, min(1.5*c_min, -0.5*c_max), 1.5*c_max])
        plt.legend()
        plt.grid()
        
        if  save_pdf:
            plt.savefig("central_potential.pdf")
        plt.show()
    

class TalmiGeneralizedTransformation(_TalmiTransformation_SecureIter):
    
    """
    Generalization of the Talmi interaction to emply R - dependent terms
    by extend the sumatory for N' != N and L' != L
    """
    
    def __init__(self, bra, ket, run_it=True):
        _TalmiTransformation_SecureIter.__init__(self, bra, ket, run_it=False)
        
        self._L_q = None
        self._N_q = None
        self._q   = None
        
        if not self.isNullMatrixElement and run_it:
            # evaluate the normal and anti_symmetrized m.e.
            self._run()
    
    def _validKet_totalAngularMomentums(self):
        """ 
        Valid orbital Angular momentum L to L' (system (r1 + r2)/2 )
        """
        raise MatrixElementException("Abstract Method, implement me!")
        return
    
    @classmethod
    def _calculateIntegrals(cls, n_integrals =1, rel_coordinates=True):
        """
        :n_integrals     = 1     : number of integrals for the different  
        :rel_coordinates = True  : relative or com coordinate integrals
        """
        arg_keys_rel = [
            CentralGeneralizedMEParameters.potential, 
            SHO_Parameters.b_length,
            CentralGeneralizedMEParameters.mu_length,
            CentralGeneralizedMEParameters.n_power,
        ]
        arg_keys_com = [
            CentralGeneralizedMEParameters.potential_R, 
            SHO_Parameters.b_length,
            CentralGeneralizedMEParameters.mu_length_R,
            CentralGeneralizedMEParameters.n_power_R
        ]
        args_to_use = arg_keys_rel if rel_coordinates else arg_keys_com
        _but_args = arg_keys_rel + arg_keys_com
        
        args = [ 
            cls.PARAMS_FORCE.get(args_to_use[0]), 
            cls.PARAMS_SHO  .get(args_to_use[1]),
            cls.PARAMS_FORCE.get(args_to_use[2]), 
            cls.PARAMS_FORCE.get(args_to_use[3]),
        ]
        
        if rel_coordinates:
            kwargs = map(lambda x: (x, cls.PARAMS_FORCE.get(x, None)),
                         CentralGeneralizedMEParameters.members(but=_but_args))
            kwargs = dict(filter(lambda x: x[1] != None, kwargs))
            
            for p in range(cls._integrals_p_max + 1, cls._integrals_p_max + n_integrals +1):                
                cls._talmiIntegrals.append(talmiIntegral(p, *args, **kwargs))
                cls._integrals_p_max += 1
        else:
            kwargs = map(lambda x: (x, cls.PARAMS_FORCE.get(x, None)), 
                         CentralGeneralizedMEParameters.members(but=_but_args))
            kwargs = dict(filter(lambda x: x[1] != None, kwargs))
            
            for q in range(cls._integrals_q_max + 1, cls._integrals_q_max + n_integrals +1):                
                cls._talmiIntegrals_R.append(talmiIntegral(q, *args, **kwargs))
                cls._integrals_q_max += 1
            
    def talmiIntegral(self):
        """ 
        Get or update Talmi integrals for the calculations
        """
        if self._p > self._integrals_p_max:
            self._calculateIntegrals(n_integrals = max(self.rho_bra, self.rho_ket, 1),
                                     rel_coordinates=True)
        return self._talmiIntegrals.__getitem__(self._p)
    
    def totalRCoordinateTalmiIntegral(self, **kwargs):
        """
        Define the integral for the total system in powers of r^q:
            I_q = sqrt(2)*b^2/Gamma(p+3/2) * 
                        integral ( dR (R/sqrt(2)b)^{2p+1} e^{-(R/b)^2 /2} U(R))
        being b the oscillator length.
        """
        if self._q > self._integrals_q_max:
            self._calculateIntegrals(n_integrals = max(self.rho_bra, self.rho_ket, 1),
                                     rel_coordinates=False)
        return self._talmiIntegrals_R.__getitem__(self._q)
    
    def _interactionSeries(self):
        
        sum_ = 0.0
        if self.DEBUG_MODE:
            XLog.write('intSer')
        B_LEN = self.PARAMS_SHO.get(SHO_Parameters.b_length)
        for qqnn_bra in self._validCOM_qqnn('bra'):
            
            self._n, self._l, self._N, self._L = qqnn_bra
            
            bmb_bra = BM_Bracket(self._n, self._l, self._N, self._L, 
                                 self.bra.n1, self.bra.l1, self.bra.n2, self.bra.l2, 
                                 self.L_bra)
            if self.isNullValue(bmb_bra):
                continue
            if self.DEBUG_MODE:
                XLog.write('intSer', nlNL=qqnn_bra)
            
            for qqnn_ket in self._validCOM_qqnn('ket'):
                
                self._n_q, self._l_q, self._N_q, self._L_q = qqnn_ket
                
                if not self._l_q in self._validKet_relativeAngularMomentums():
                    continue
                if not self._L_q in self._validKet_totalAngularMomentums():
                    continue
                skip  = (self._n_q + self._N_q)
                skip -= (self._n + self._N) + (self.rho_ket - self.rho_bra )//2
                if skip != 0: 
                    continue
                
                bmb_ket = BM_Bracket(self._n_q, self._l_q, self._N_q, self._L_q, 
                                     self.ket.n1, self.ket.l1, self.ket.n2, self.ket.l2, 
                                     self.L_ket)
                
                if not self._deltaConditionsForCOM_Iteration():
                    continue
                if self.isNullValue(bmb_ket):
                    continue
                
                _b_coeff_p = self._B_coefficient(B_LEN, com_qqnn=False)
                _b_coeff_q = self._B_coefficient(B_LEN, com_qqnn=True )
                _com_coeff = self._interactionConstantsForCOM_Iteration()
                
                aux =  _com_coeff * bmb_bra * bmb_ket * _b_coeff_p * _b_coeff_q
                sum_ += aux
                
                # TODO: comment when not debugging
                if self.DEBUG_MODE:
                    XLog.write('intSer_ket', nq=self._n_q, lq=self._l_q, 
                               Nq=self._N_q, Lq=self._L_q,
                               bmbs=bmb_bra * bmb_ket, 
                               Bp=_b_coeff_p, Bq=_b_coeff_q, comCoeff=_com_coeff, 
                               aux=aux)
                #     self._debbugingTable(bmb_bra, bmb_ket, _com_coeff, _b_coeff)
        if self.DEBUG_MODE:
            XLog.write('intSer', value=sum_)
        return sum_
    
    
    def _BrodyMoshinskyTransformation(self):
        """
        ##  WARNING: This method is final. Do not overwrite!!
        
        # This overwritting is for general R and r integrable interactions.
        
        Sum over valid p-Talmi integrals range (Energy + Ang. momentum + parity 
        conservation laws). Call implemented Talmi Coefficients for 
        Brody-Moshinsky transformation
        """
        
        sum_ = 0.0
        if self.DEBUG_MODE: XLog.write('talmi')
        for p in range(max(self.rho_bra, self.rho_ket) +1):            
            self._p = p
            for q in range(max(self.rho_bra, self.rho_ket) +1):
                if self.DEBUG_MODE: XLog.write('talmi', p=p, q=q)
                
                self._q = q
                
                # 2* from the antisymmetrization_ (_deltaConditionsForCOM_Iteration)
                series = 2 * self._interactionSeries()
                Ip =     self.talmiIntegral()
                Iq =     self.totalRCoordinateTalmiIntegral()
                product = series * Ip * Iq
                sum_ += product
                                
                if self.DEBUG_MODE:
                    XLog.write('talmi', series = series, Ip=Ip, Iq=Iq, val=product)
        
        if self.DEBUG_MODE: XLog.write('talmi', final_sum=sum_)
        return self._globalInteractionCoefficient() * sum_
    

