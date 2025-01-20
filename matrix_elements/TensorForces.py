'''
Created on Mar 8, 2021

@author: Miguel
'''
import numpy as np

from helpers.Enums import CouplingSchemeEnum, CentralMEParameters, \
    BrinkBoekerParameters, SHO_Parameters, PotentialForms, AttributeArgs
from helpers.Helpers import ConstantsV18, safe_clebsch_gordan

from matrix_elements.MatrixElement import _TwoBodyMatrixElement_JTCoupled,\
    MatrixElementException, _standardSetUpForCentralWithExchangeOps,\
    _TwoBodyMatrixElement_JCoupled
from matrix_elements.transformations import TalmiTransformation
from helpers.Helpers import safe_racah, safe_wigner_6j
from helpers.WaveFunctions import QN_2body_LS_Coupling
from helpers.Log import XLog
from helpers.integrals import talmiIntegral

class TensorForce(TalmiTransformation):#):
    
    COUPLING = (CouplingSchemeEnum.L, CouplingSchemeEnum.S)
    
    def __init__(self, bra, ket, J,  run_it=True):
        self.__checkInputArguments(bra, ket, J)
         
        # TODO: Might accept an LS coupled wave functions (when got that class)
        self.J = J
         
        self.S_bra = bra.S
        self.S_ket = ket.S
         
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
            for attr in (CentralMEParameters.opt_cutoff,
                         CentralMEParameters.opt_mu_2,
                         CentralMEParameters.opt_mu_3):
                if attr in kwargs:
                    _map[attr] = (AttributeArgs.value, float)
            
            kwargs = TensorForce._automaticParseInteractionParameters(_map, kwargs)
        
        super(TensorForce, cls).setInteractionParameters(*args, **kwargs)
    
    def __checkInputArguments(self, bra, ket, J):
        if not isinstance(J, int):
            raise MatrixElementException("J is not <int>")
        if not isinstance(bra, QN_2body_LS_Coupling):
            raise MatrixElementException("<bra| is not <QN_2body_LS_Coupling>")
        if not isinstance(ket, QN_2body_LS_Coupling):
            raise MatrixElementException("|ket> is not <QN_2body_LS_Coupling>")
    
    
    def _validKet_relativeAngularMomentums(self):
        """  Tensor interaction only allows l'=l and l +- 2 """
        if self._l > 1:
            return (self._l - 2, self._l, self._l + 2)
        else:
            return (self._l, self._l + 2)
    
    def deltaConditionsForGlobalQN(self):
        """ 
        Define if non null requirements on LS coupled J Matrix Element, 
        before doing the center of mass decomposition.
        
        NOTE: Redundant if run from JJ -> LS recoupling
        """
        if (abs(self.L_bra - self.L_ket) > 2):
            return False
        return True
    
    def _deltaConditionsForCOM_Iteration(self):
        """ For the antisymmetrization_ of the wave functions. """
        if (((self.S_bra + self.T + self._l) % 2 == 1) and 
            ((self.S_ket + self.T + self._l_q) % 2 == 1)):
            return True
        return False
    
    def _totalSpinTensorMatrixElement(self):
        """ <1/2 1/2 (S) | S^[1]| 1/2 1/2 (S)>, only non zero for S=S'=1 """
        if (self.S_bra != self.S_ket) or (self.S_bra == 0):
            return True, 0.0
        
        return False, 3.872983346207417 ## = np.sqrt(15)
    
    def centerOfMassMatrixElementEvaluation(self):
        """ 
        Radial Brody-Moshinsky transformation, implementation for a
        non central tensor force.
        """
        skip, spin_me = self._totalSpinTensorMatrixElement()
        if skip:
            return 0
        
        factor = safe_racah(self.L_bra, self.L_ket, 
                            self.S_bra, self.S_ket,
                            2, self.J)
        if self.isNullValue(factor) or not self.deltaConditionsForGlobalQN():
            return 0
        
        # no special interaction constant for the Central ME
        # phase = (-1)**(1 + self.rho_bra - self.J)
        phase   = (-1)**(self.S_bra + self.L_ket  - self.L_bra - self.J)
        ## TODO: last phase must be the one (commented is fine/ 1st version))
        factor *= np.sqrt(8*(2*self.L_bra + 1)*(2*self.L_ket + 1))
        
        return factor * spin_me * phase * self._BrodyMoshinskyTransformation()
    
    def _globalInteractionCoefficient(self):
        
        return self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    
    def _interactionConstantsForCOM_Iteration(self):
        # no special internal c.o.m interaction constants for the Central ME
        factor = safe_racah(self.L_bra, self.L_ket, 
                            self._l, self._l_q,
                            2, self._L)
        if self.isNullValue(factor):
            return 0
        
        factor *= safe_clebsch_gordan(self._l, 2, self._l_q, 0, 0, 0)
        phase = (-1)**(self._L - self._l_q)
        ## TODO: phase must be uncommented (commented is fine, 1st version)
        
        return factor * np.sqrt(2*self._l + 1) * phase 





class TensorForce_JTScheme(TensorForce, _TwoBodyMatrixElement_JTCoupled):
    
    COUPLING = (CouplingSchemeEnum.JJ, CouplingSchemeEnum.T)
    
    def __init__(self, bra, ket, run_it=True):
        _TwoBodyMatrixElement_JTCoupled.__init__(self, bra, ket, run_it=run_it)
    
    
    def _validKetTotalSpins(self):
        """ 
        Return ket states <tuple> of the total spin, for tensor force impose 
        S = S' = 1, return nothing to skip the bracket spin S=0
        """
        if self.S_bra == 0:
            return []
        return (1, )
    
    def _deltaConditionsForCOM_Iteration(self):
        """ For the antisymmetrization_ of the wave functions. """
        if (((self.S_bra + self.T + self._l) % 2 == 1) and 
            ((self.S_ket + self.T + self._l_q) % 2 == 1)):
                return True
        return False
    
    def _validKetTotalAngularMomentums(self):
        """ 
        Return ket states <tuple> of the total angular momentum, depending of 
        the Force.
        
        OJO: Moshinsky, lambda' = lambda, lambda +-1, lambda +-2!!! as condition
        in the C_Tensor, due rank 2 tensor coupling
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
    
    def _run(self):
        ## First method that runs antisymmetrization_ by exchange the quantum
        ## numbers (X2 time), change 2* _series_coefficient
        return _TwoBodyMatrixElement_JTCoupled._run(self)



class TensorS12_JTScheme(TensorForce_JTScheme):
    
    """ 
        Recalculated using the evaluation of [sigma(1) x sigma(2)]_2 
        (obtained by me (delafuen), differnces in the spin parts.
        test if match with Moshinsky's values)
    """
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        """ 
        Implement the parameters for the Tensor interaction calculation. 
        
        Modification to import Exchange operators in the Brink-Boeker form.
        """
        cls = _standardSetUpForCentralWithExchangeOps(cls, **kwargs) 
        
        cls._integrals_p_max = -1
        cls._talmiIntegrals  = []
    
    
    def _totalSpinTensorMatrixElement(self):
        """ <1/2 1/2 (S) | S^[1]| 1/2 1/2 (S)>, only non zero for S=S'=1 """
        if (self.S_bra == 0) or (self.S_ket == 0):
            return True, 0.0
        return False, 4.47213595499958 ## = np.sqrt(20)
    
    def centerOfMassMatrixElementEvaluation(self):
        """ 
        Radial Brody-Moshinsky transformation, implementation for a
        non central tensor force.
        """
        skip, spin_me = self._totalSpinTensorMatrixElement()
        if skip:
            return 0
        
        factor = safe_wigner_6j(self.L_bra, self.S_bra, self.J,
                                self.S_ket, self.L_ket, 2)
        if self.isNullValue(factor) or not self.deltaConditionsForGlobalQN():
            return 0
        phase   = (-1)**(self.S_bra + self.J + self.L_ket + self.L_bra)
        ## NOTE: the last L_bra should be from the ket since the W
        factor *= phase * 3.8832518251113983 #* np.sqrt(2*self.J + 1) 
        ## 3.8832 = _sqrt(24*pi / 5)
        factor *= ((2*self.L_bra + 1)*(2*self.L_ket + 1))**0.5
        
        return factor * spin_me * self._BrodyMoshinskyTransformation()
    
    def _globalInteractionCoefficient(self):
        # no special interaction constant for the Tensor
        return self.PARAMS_FORCE.get(CentralMEParameters.constant)
    
    def _interactionConstantsForCOM_Iteration(self):
        # # factors from transforming the <Y_2* V(r)> m.e.
        # factor = safe_racah(self._l, self._l_q, self.L_bra, self.L_ket, 
        #                     2, self._L)
        # if self.isNullValue(factor):
        #     return 0
        #
        # phase   = (-1)**(self._L + self.L_ket - self._l)
        # factor *= float(clebsch_gordan(self._l, 2, self._l_q, 0, 0, 0))
        # factor *= np.sqrt((2*self._l + 1)*(2*self.L_bra + 1)*(2*self.L_ket + 1))
        #
        # return phase * factor * 0.6307831305050401  # _sqrt(5 / 4*pi)
        
        factor = safe_wigner_6j(self._L, self._l,    self.L_bra,
                                2,       self.L_ket, self._l_q)
        if self.isNullValue(factor):
            return 0
        phase   = (-1)**(self._L + self._l)
        factor *= safe_clebsch_gordan(self._l, 2, self._l_q, 0, 0, 0)
        factor *= np.sqrt((2*self._l + 1))
        
        return phase * factor * 0.6307831305050401  # _sqrt(5 / 4*pi)
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        # return self.centerOfMassMatrixElementEvaluation()
    
        # Radial Part for Gaussian Integral
        radial_energy = self.centerOfMassMatrixElementEvaluation()
        
        # Exchange Part
        # W + P(S)* B - P(T)* H - P(T)*P(S)* M
        _S_aux = (-1)**(self.S_bra + 1)
        _T_aux = (-1)**(self.T)
        _L_aux = (-1)**(self.T + self.S_bra + 1)
        
        exchange_energy = (
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Wigner),
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Bartlett)   * _S_aux,
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Heisenberg) * _T_aux,
            self.PARAMS_FORCE.get(BrinkBoekerParameters.Majorana)   * _L_aux
        )
        
        # Add up
        prod_part = radial_energy * sum(exchange_energy)
        
        if self.DEBUG_MODE:
            XLog.write('BB', radial=radial_energy, exch=exchange_energy, 
                       exch_sum=sum(exchange_energy), val=prod_part)
        
        return prod_part


class ElectromagneticNonCentral_JScheme(TensorForce, _TwoBodyMatrixElement_JCoupled):
    """
    TODO: Terms from the Argone potential for the Magnetic-Moment.
    constants dependent on the pp - pn - nn chanels
    
    Not setteable
    
    Components non-central terms dependent on sigma*sigma, S_ij, L*S, L*A
    """
    
    pass

class OPE_TPE_Force_JTScheme(TensorForce_JTScheme):
    
    """
    Two-Pion-Exchange potential, from Nijmegen partial-wave analysis:
    
        C * S12 * (1 + 3(r/mu) + 3(r/mu)^2) * Y(r/b) * (cutoff)
    
    Includes an optional parameter for a cuttoff function 1 - exp(-cr^2)
    That requires to change the Talmi integrals
    """
    
    PARAMS_FORCE = {
        -1: { # 'pp'
            },
        0 : { # 'pn'
            },
        1 : { # 'nn'
            },
        CentralMEParameters.opt_cutoff : 0.6900655593423541, #1 / (2.1**0.5),
        }
    
    FACTOR_Fpp =  0.27386127875   # sqrt(0.075)
    FACTOR_Fnn = -0.27386127875
    FACTOR_Fc  =  0.27386127875
    
    @classmethod
    def setInteractionParameters(cls, *args, **kwargs):
        
        ## TODO. define the SHO and the constants (to avoid recalculating)
        cls.PARAMS_SHO[SHO_Parameters.b_length] = None
        
        scaling_0pm = (ConstantsV18.M_PION_0 / ConstantsV18.M_PION_pm)**2
        
        const_pp = (cls.FACTOR_Fpp**2) * scaling_0pm * ConstantsV18.M_PION_0 / 3
        const_nn = (cls.FACTOR_Fnn**2) * scaling_0pm * ConstantsV18.M_PION_0 / 3
        const_pn1= 2 * (cls.FACTOR_Fc**2) * ConstantsV18.M_PION_pm / 3
        const_pn0= cls.FACTOR_Fpp * cls.FACTOR_Fnn * (scaling_0pm * 
                                                      ConstantsV18.M_PION_0 / 3)
        
        ## lengths for the code has to be for r/mu, not as in the article
        mu_pp = (ConstantsV18.HBAR_C / ConstantsV18.M_PION_0)
        mu_pn = (ConstantsV18.HBAR_C / ConstantsV18.M_PION_pm)
        
        cls.PARAMS_FORCE = {
            -1: { # 'pp'
                 CentralMEParameters.constant : const_pp, 
                 CentralMEParameters.mu_length: mu_pp,
                },
            0 : { # 'pn'
                 CentralMEParameters.constant : (const_pn0, const_pn1), 
                 CentralMEParameters.mu_length: (mu_pp, mu_pn),
                },
            1 : { # 'nn'
                 CentralMEParameters.constant : const_nn, 
                 CentralMEParameters.mu_length: mu_pp,
                },
            CentralMEParameters.opt_cutoff : 0.6900655593423541, #1 / (2.1**0.5),
        }
        
    
    @classmethod
    def _calculateIntegrals(cls, n_integrals =1):
        
        arg_keys = [
            CentralMEParameters.potential, 
            SHO_Parameters.b_length,
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
            
            args[3] = 0
            
            I1 = talmiIntegral(p, *args, **kwargs)
            
            args[0] = PotentialForms.YukawaGauss_power
            args[3] = 1
            kwargs[CentralMEParameters.opt_mu_2] = 0.0
            I2 = talmiIntegral(p, *args, **kwargs)
            
            args[3] = 2
            I3 = talmiIntegral(p, *args, **kwargs)
            kwargs[CentralMEParameters.opt_mu_2] = 0.0
            cls._talmiIntegrals.append(talmiIntegral(p, *args, **kwargs))
            
            cls._integrals_p_max += 1
    
    
    def _deltaConditionsForCOM_Iteration(self):
        """ For the antisymmetrization_ of the wave functions. """
        if (((self.S_bra + 1 + self._l) % 2 == 1) and 
            ((self.S_ket + 1 + self._l_q) % 2 == 1)):
                return True
        return False
    
    def _LScoupled_MatrixElement(self):#, L, S, L_ket=None, S_ket=None):
        """ 
        <(n1,l1)(n2,l2) (LS)| V |(n1,l1)'(n2,l2)'(L'S') (T)>
        """
        return self.centerOfMassMatrixElementEvaluation()
    
    def _run(self):
        ## First method that runs antisymmetrization_ by exchange the quantum
        ## numbers (X2 time), change 2* _series_coefficient
        return _TwoBodyMatrixElement_JTCoupled._run(self)