'''
Created on Mar 12, 2021

@author: Miguel
'''
import numpy as np

from helpers.Enums import PotentialForms, CentralMEParameters
from helpers.Helpers import gamma_half_int, fact, angular_condition,\
    safe_clebsch_gordan, _B_coeff_memo_accessor,\
    getStatesAndOccupationUpToLastOccupied, shellSHO_Notation,\
    getStatesAndOccupationOfFullNucleus, getGeneralizedLaguerreRootsWeights
from helpers.Log import XLog
from . import SCIPY_INSTALLED
from copy import deepcopy
# SCIPY_INSTALLED = 1
if SCIPY_INSTALLED:
    from scipy.special import gammaincc
    from scipy.special import roots_laguerre, roots_genlaguerre, roots_legendre


class IntegralException(BaseException):
    pass

def talmiIntegral(p, potential, b_param, mu_param, n_power=0, **kwargs):
    """
    :p          index order
    :potential  form of the potential, from Poten
    :b_param    SHO length parameter
    :mu_param   force proportional coefficient (by interaction definition)
    :n_power    auxiliary parameter for power dependent potentials

    :kwargs are optional parameters defined in CentralParams
    """
    
    ## In case of CUTOFF :: The integral would have a filtering convolution.
    ##     Connect with the propper new function and fix the constants:
    if CentralMEParameters.opt_cutoff in kwargs:
        kwargs2 = deepcopy(kwargs)
        cutoff_len = kwargs[CentralMEParameters.opt_cutoff]
        del kwargs2[CentralMEParameters.opt_cutoff]
        
        return integralsWithCutoff(p, potential, cutoff_len,  
                                   b_param, mu_param, n_power, **kwargs2)
    
    # -------------------------------------------------------------------------
    
    if potential == PotentialForms.Gaussian:
        return (b_param**3) / (1 + 2*(b_param/mu_param)**2)**(p+1.5)
    
    elif potential == PotentialForms.Coulomb:
        return (b_param**2) * mu_param * np.exp(fact(p) - gamma_half_int(2*p + 3))\
                / (2**.5)
    elif potential == PotentialForms.Delta:
        if p != 0: return 0
        return np.sqrt(2 / np.pi)  ## it is independent of b length
    
    elif potential == PotentialForms.Gaussian_power:
        ## checked (16/1/25), compared with numerical integration. (Lag.ass.quadr.)
        ## NOTE: n_power = -3, -5 ,... lead to undefined Gamma
        if n_power < -2: assert n_power % 2 == 0, "n_power < 0 must be -1 or even!"
        
        aux = gamma_half_int(2*p + 3 + n_power) - gamma_half_int(2*p + 3)
        aux = (b_param**3) * np.exp(aux)
        
        x = (2**0.5) * b_param / mu_param
        return aux * (x**n_power) / ((1 + x**2)**(p + 1.5 + (n_power/2)))
    
    elif potential == PotentialForms.Power:
        if n_power == 0:
            return b_param**3
        aux =  gamma_half_int(2*p + 3 + n_power) - gamma_half_int(2*p + 3)
        aux += n_power * ((np.log(2)/2) - np.log(mu_param))
        return np.exp(aux + ((n_power + 3)*np.log(b_param)))
    
    elif potential == PotentialForms.Yukawa:
        sum_ = 0.       
        
        A    = (2**0.5) * b_param / mu_param
        x, w = getGeneralizedLaguerreRootsWeights(2*p + 1, order=max(10, 3*p//5))
        f    = np.exp(-np.power(x/A, 2))
        sum_ = sum(w * f)
        
        aux   = 2 * (b_param**3) 
        sum_ *= aux / np.exp(gamma_half_int(p + 3) + (2*p + 3)*np.log(A))
        
        return sum_
    
    elif potential == PotentialForms.Exponential:
        sum_ = 0.       
        
        A    = (2**0.5) * b_param / mu_param
        x, w = getGeneralizedLaguerreRootsWeights(2*p + 2, order=max(10, 3*p//5))
        f    = np.exp(-np.power(x/A, 2))
        sum_ = sum(w * f)
        
        aux   = 2 * (b_param**3) 
        sum_ *= aux / np.exp(gamma_half_int(p + 3) + (2*p + 3)*np.log(A))
        
        return sum_
    
    elif potential == PotentialForms.YukawaGauss_power:
        sum_ = 0.
        mu_2 = kwargs.get(CentralMEParameters.opt_mu_2, mu_param)
        mu_3 = kwargs.get(CentralMEParameters.opt_mu_3, mu_param)
        
        A = 1 + 2 * (b_param / mu_2)**2
        B = (2**0.5) * b_param / mu_param
        C = B / (2 * np.sqrt(A))
        
        N = 2*p + 1 + n_power
        for k in range(N +1):
            aux = gamma_half_int((N + 1 - k)) - fact(k) - fact(N - k)
            aux = np.exp(aux) * ((-C)**k) * gammaincc(p + 1 + (N - k)/2, C**2)
            
            sum_ += aux
        
        aux  = np.exp(fact(N) - gamma_half_int(2*p + 3) + C**2 - ((N+1)/2)*np.log(A))
        aux *= (b_param**3) * np.power(2**0.5 * b_param / mu_3 ,N) / B
        sum_*= aux
                
        return sum_
    
    elif potential == PotentialForms.Wood_Saxon:

        if not (CentralMEParameters.opt_mu_2   in kwargs and 
                CentralMEParameters.opt_mu_3   in kwargs):
            raise IntegralException("missing parameters in kwargs:", kwargs)
        
        aux  = b_param**3 / np.exp(gamma_half_int(2*p + 3))
        if n_power>0: 
            aux *= ((np.sqrt(2) * b_param/ mu_param)**n_power)
        
        ## Laguerre Integral for the potential
        a  = kwargs.get(CentralMEParameters.opt_mu_2) 
        r0 = kwargs.get(CentralMEParameters.opt_mu_3)
        A, B  = np.sqrt(2) * b_param / a, r0 / a 
        x, w  = getGeneralizedLaguerreRootsWeights(2*p + 1 + n_power)
        f     = 1.0 / (1.0 + np.exp((A*np.sqrt(x)) - B))
        sum_ = sum(w * f)
        
        return sum_
        
    else:
        raise IntegralException("Talmi integral [{}] is not defined, valid potentials: {}"
                        .format(potential, PotentialForms.members()))

#===============================================================================
## NOTE: Old expressions for the exponential-like potentials, not recommended, 
## this version fails for p>15 due the large sum additions in the series.
    # elif potential == PotentialForms.Yukawa:
    #     sum_ = 0.       
    #
    #     cte_bm = b_param / ((2**0.5) * mu_param)
    #     N = 2*p + 1
    #     for k in range(N +1):
    #         aux = gamma_half_int((N + 1 - k)) - fact(k) - fact(N - k)
    #         aux = np.exp(aux) * ((-cte_bm)**k) * gammaincc(p + 1 - k/2, cte_bm**2)
    #
    #         sum_ += aux
    #
    #     aux  = np.exp( fact(N) - gamma_half_int(2*p + 3) + cte_bm**2)
    #     aux *= mu_param * (b_param**2) / (2**0.5)
    #     sum_ *= aux
    #
    #     return sum_
    #
    # elif potential == PotentialForms.Exponential:
    #     sum_ = 0.       
    #
    #     cte_bm = b_param / ((2**0.5) * mu_param)
    #     N = 2*p + 2
    #     for k in range(N +1):
    #         aux = gamma_half_int((N + 1 - k)) - fact(k) - fact(N - k)
    #         aux = np.exp(aux) * ((-cte_bm)**k) * gammaincc(p + (3 - k)/2, cte_bm**2)
    #
    #         sum_ += aux
    #
    #     aux  = np.exp( fact(N) - gamma_half_int(2*p + 3) + cte_bm**2)
    #     aux *= (b_param**3)
    #     sum_ *= aux
    #
    #     return sum_
#===============================================================================

def integralsWithCutoff(p, potential, cutoff_len, b_param, mu_param, n_power, 
                        **kwargs):
    """
    This method connects the Talmi integrals of order p for a certain potential
    under a cuttoff function:
        V(r) * (1 - exp(- cutoff_len*r^2)) = V(r) - V(r)*exp(- cutoff_len*r^2)
    
    NOTE: kwargs must not contain central argument names and/or cutoff.
        in case of being called from talmiIntegral() will lead to a cycle.
    NOTE: Some integrals could be not implemented.
    """
    
    integral_1 = talmiIntegral(p, potential, b_param, mu_param, n_power, **kwargs)
    integral_2 = 0.0
    
    if potential == PotentialForms.Gaussian:
        raise IntegralException("TODO: implement me!")
    
    elif potential == PotentialForms.Exponential:
        raise IntegralException("TODO: implement me!")
    
    elif potential == PotentialForms.Coulomb:
        raise IntegralException("TODO: implement me!")
    
    elif potential == PotentialForms.Yukawa:
        raise IntegralException("TODO: implement me!")
    
    elif potential == PotentialForms.Power:
        raise IntegralException("TODO: implement me!")
    
    elif potential == PotentialForms.Gaussian_power:
        raise IntegralException("TODO: implement me!")
    
    elif potential == PotentialForms.YukawaGauss_power:
        raise IntegralException("TODO: implement me!")
    
    elif potential == PotentialForms.Wood_Saxon:
        raise IntegralException("TODO: implement me!")
    
    return integral_1 - integral_2

#if not 'roots_legendre' in globals():
#from scipy.special import roots_legendre, roots_laguerre

class GaussianQuadrature:
    
    """ 
    Class to evaluate Gaussian type integrations, save the roots and weights for
    speeding.
    """
    
    _roots_weights_Legendre = {}
    _roots_weights_Laguerre = {}
    
    @staticmethod
    def legendre(function, a, b, order, *args, **kwargs):
        """ 
        Evaluate integral of <function> object, of 1 variable and parameters:
            Args:
        :function  <function> f(x, *args, **kwargs)
        :a         <float> lower limit
        :b         <float> upper limit
        :order     <int>   order of the quadrature
        
        *args and **kwargs will be passed to the function (check argument introduction
            in case of several function management)
        """
        
        A = (b - a) / 2
        B = (b + a) / 2
        
        if not order in GaussianQuadrature._roots_weights_Legendre:
            x_i, w_i = roots_legendre(order, mu=False)
            GaussianQuadrature._roots_weights_Legendre[order] = (x_i, w_i)
        else:
            x_i, w_i = GaussianQuadrature._roots_weights_Legendre.get(order)
        x_i_aux = A*x_i + B
        
        #integral = A * sum(w_i * function(x_i, *args, **kwargs))
        #return integral
        
        integral = 0.0
        for i in range(len(x_i)):
            integral += w_i[i] * function(x_i_aux[i], *args, **kwargs)
        
        #print("order", order, "=",integral)
        return A * integral
        
    @staticmethod
    def laguerre(function, order, *args, **kwargs):
        """ 
        Evaluate integral of <function> object, of 1 variable and parameters:
            integral[0, +infinity] {dx exp(-x) function(x, **args)} 
            Args:
        :function  <function> f(x, *args, **kwargs)
        :order     <int>   order of the quadrature
                or <tuple> (order, alpha) where alpha is float for the associated 
                              Laguerre integration (dx x^alpha exp(-x) *f(x))
        
        *args and **kwargs will be passed to the function (check argument introduction
            in case of several function management)
        """
        if not order in GaussianQuadrature._roots_weights_Laguerre:
            if isinstance(order, tuple):
                x_i, w_i = roots_genlaguerre(order[0], alpha=order[1])
            else:
                x_i, w_i = roots_laguerre(order, mu=False)
            GaussianQuadrature._roots_weights_Laguerre[order] = (x_i, w_i)
        else:
            x_i, w_i = GaussianQuadrature._roots_weights_Laguerre.get(order)
        
        # integral = 0.0
        # for i in range(len(x_i)):
        #     integral += w_i[i] * function(x_i[i], *args, **kwargs)
        integral = np.dot(w_i, function(x_i, *args, **kwargs))
        
        #print("order", order, "=",integral)
        return integral


#===============================================================================
#     SPIN ORBIT INTEGRAL FOR GOGNY D1S
#===============================================================================


class _SpinOrbitPartialIntegral:
    
    """ 
    Evaluates the integral I_(q=0){ij}{kl}, from the Spin Orbit term that appear 
    in Gogny interaction (modified here to accept V(r) functions), for the 
    harmonic oscillator wave functions with form:
    
    I[ij][kl] ~ [(j*)k](grad(i,+1)* grad(k,-1) - grad(i,-1)* grad(k,+1))
        
    :qqnn_i  ket's <QN_1body_radial> in the gradient term
    :qqnn_j  ket's <QN_1body_radial>
    :qqnn_k  bra's <QN_1body_radial> in the gradient term
    :qqnn_l  bra's <QN_1body_radial>
    
    """
    __PARAMETERS_SETTED = False
    
    def __init__(self, qqnn_i, qqnn_j, qqnn_k, qqnn_l):
        
        self.wf_i = qqnn_i
        self.wf_j = qqnn_j
        self.wf_k = qqnn_k
        self.wf_l = qqnn_l
        
        self._testing_valid_qqnn_DpDp_positive()
    
    @classmethod
    def setInteractionParameters(cls, *args):
        """
        :args processed from spin orbit interaction (just need to call it once):
        same form/order as talmiIntegral() function arguments
            args[0]: CentralMEParameters.potential
            args[1]: SHO_Parameters.b_length
            args[2]: CentralMEParameters.mu_length (ignored)
            args[3]: CentralMEParameters.n_power
        
        Just implemented for Gogny (Constant), then potential == 'power' and
        n_power == 0
        """
        if not cls.__PARAMETERS_SETTED:
            
            cls.b_param = args[1]
            
            if args[0] != PotentialForms.Power or args[3] != 0:
                raise IntegralException("Invalid Potential given, {} accepts only"
                    " Constant radial potential (Gogny Force), that is "
                    "PotentialForms.Power and CentralMEParameters.n_power=0. Got: [{}][{}]"
                    .format(cls.__class__.__name__, args[0], args[3]))
            
            cls.PARAMS_FORCE = args
            
            cls._a_coeff_nkl_memo   = {}
            cls._baseIntegrals_memo = {}
            
            cls.__PARAMETERS_SETTED = True
            
        
    @classmethod
    def delteteInteractionParameters(cls):
        cls.__PARAMETERS_SETTED = False
        cls._a_coeff_nkl_memo   = {}
        cls._baseIntegrals_memo = {}
    
    
    def _testing_valid_qqnn_DpDp_positive(self):
        if hasattr(self, 'valid_qqnn'):
            return
        
        l1, m1, l2, m2 = self.wf_i.l, self.wf_i.m_l, self.wf_j.l, self.wf_j.m_l
        l3, m3, l4, m4 = self.wf_k.l, self.wf_k.m_l, self.wf_l.l, self.wf_l.m_l
        
        self.valid_qqnn = set()
        
        for m1 in range(-l1, l1+1):
            m1_q = m1 + 1
            l1_q = l1 + 1
            
            for m2 in range(-l2, l2+1):
                
                M = m1_q + m2
                
                for m3 in range(-l3, l3+1):
                    m3_q = m3 - 1
                    l3_q = l3 + 1
                    
                    for m4 in range(-l4, l4+1):
                        
                        M_q = m3_q + m4
                        
                        if ((l1_q + l2)%2 == (l3_q + l4)%2):
                            if M == M_q:
                                aux = (l1, m1, l3, m3, m2, m4)
                                self.valid_qqnn.add(aux)
        _=0
        
    
    #===========================================================================
    # run 
    #===========================================================================
    
    def value(self):
        if not self.__PARAMETERS_SETTED:
            raise Exception("Error: Interaction parameters have not been defined")
        
        
        l_i, m_i = self.wf_i.l, self.wf_i.m_l
        l_k, m_k = self.wf_k.l, self.wf_k.m_l
        
        #self._big_N2Ljk = self.wf_i.l
        
        self._G_i_pp = self._gradientCoeff(1, 1, l_i, m_i)
        self._G_i_pm = self._gradientCoeff(1, 0, l_i, m_i)
        self._G_i_mp = self._gradientCoeff(0, 1, l_i, m_i)
        self._G_i_mm = self._gradientCoeff(0, 0, l_i, m_i)
        
        self._G_k_pp = self._gradientCoeff(1, 1, l_k, m_k)
        self._G_k_pm = self._gradientCoeff(1, 0, l_k, m_k)
        self._G_k_mp = self._gradientCoeff(0, 1, l_k, m_k)
        self._G_k_mm = self._gradientCoeff(0, 0, l_k, m_k)
        
        #TODO: checker to be removed
#         if (((l_i, m_i, l_k, m_k)==(1, -1, 1, 1)) 
#             and ((self.wf_j.m_l, self.wf_l.m_l)==(1, 1)) ):
#             _=0
        if (l_i, m_i, l_k, m_k, self.wf_j.m_l, self.wf_l.m_l) in self.valid_qqnn:
            _=0
        
        sum_ = 0
        sum_ += self._differentialPartRecouplingAndIntegtals(1, 1)
        sum_ += self._differentialPartRecouplingAndIntegtals(0, 1)
        sum_ += self._differentialPartRecouplingAndIntegtals(1, 0)
        sum_ += self._differentialPartRecouplingAndIntegtals(0, 0)
        
        return sum_
        
        # TODO: use angular_condition()
        #angular_condition(l1, l2, L)

#     ## Secure Version of the integral sum and coefficients
#     def _differentialPartRecouplingsAndIntegtal(self, diff_opp_i, diff_opp_k):
#         """
#         recoupling coefficients for the four spherical harmonics, related with 
#         the differential operators (lowers or increases the l value of the sph.h)
#         
#         :diff_opp_i 1 for (+) and 0 for (-) on the bra function
#         :diff_opp_k 1 for (+) and 0 for (-) on the ket function
#         """
#         l_i, m_i = self.wf_i.l, self.wf_i.m_l
#         l_k, m_k = self.wf_k.l, self.wf_k.m_l
#         
#         l_i = l_i + 1 if diff_opp_i else l_i - 1
#         l_k = l_k + 1 if diff_opp_k else l_k - 1
#         
#         aux = 0
#         if diff_opp_i:
#             if diff_opp_k:
#                 # D^{+}f_i * D^{+}f_k
#                 aux += (self._G_i_pp * self._G_k_mp)\
#                         * self._recouplingSH(l_i+1, m_i+1, l_k+1, m_k-1)
#                 aux -= (self._G_i_mp * self._G_k_pp)\
#                         * self._recouplingSH(l_i+1, m_i-1, l_k+1, m_k+1)
#             else:
#                 # D^{+}f_i * D^{-}f_k
#                 aux += (self._G_i_pp * self._G_k_mm)\
#                         * self._recouplingSH(l_i+1, m_i+1, l_k-1, m_k-1)
#                 aux -= (self._G_i_mp * self._G_k_pm)\
#                         * self._recouplingSH(l_i+1, m_i-1, l_k-1, m_k+1)
#         else:
#             if diff_opp_k:   
#                 # D^{-}f_i * D^{+}f_k
#                 aux += (self._G_i_pm * self._G_k_mp)\
#                         * self._recouplingSH(l_i-1, m_i+1, l_k+1, m_k-1)
#                 aux -= (self._G_i_mm * self._G_k_pp)\
#                         * self._recouplingSH(l_i-1, m_i-1, l_k+1, m_k+1)
#             else:
#                 # D^{-}f_i * D^{-}f_k
#                 aux += (self._G_i_pm * self._G_k_mm)\
#                         * self._recouplingSH(l_i-1, m_i+1, l_k-1, m_k-1)
#                 aux -= (self._G_i_mm * self._G_k_pm)\
#                         * self._recouplingSH(l_i-1, m_i-1, l_k-1, m_k+1)
#        
#         aux *= self._radialIntegrals(diff_opp_i, diff_opp_k)
#                 
#         return aux


    def _differentialPartRecouplingAndIntegtals(self, diff_opp_i, diff_opp_k):
        """
        recoupling coefficients for the four spherical harmonics, related with 
        the differential operators (lowers or increases the l value of the sph.h)
         
        :diff_opp_i 1 for (+) and 0 for (-) on the bra function
        :diff_opp_k 1 for (+) and 0 for (-) on the ket function
        """
        l_i, m_i = self.wf_i.l, self.wf_i.m_l
        l_k, m_k = self.wf_k.l, self.wf_k.m_l
         
#         l_i = l_i + 1 if diff_opp_i else l_i - 1
#         l_k = l_k + 1 if diff_opp_k else l_k - 1
        
        # product of coefficients from the gradient formula 
        # [a1 = GiGk_positive](coupling of Ys) - [a2 = GiGk_negat](coupling o Ys)
        a1 = self._coeffs_G_positive(diff_opp_i, diff_opp_k)
        a2 = self._coeffs_G_negative(diff_opp_i, diff_opp_k)
        if diff_opp_i:
            if diff_opp_k:
                # D^{+}f_i * D^{+}f_k
                a1 *= self._recouplingSphHarm(l_i+1, m_i+1, l_k+1, m_k-1)
                a2 *= self._recouplingSphHarm(l_i+1, m_i-1, l_k+1, m_k+1)
            else:
                # D^{+}f_i * D^{-}f_k
                if a1 != 0:
                    a1 *= self._recouplingSphHarm(l_i+1, m_i+1, l_k-1, m_k-1)
                if a2 != 0:
                    a2 *= self._recouplingSphHarm(l_i+1, m_i-1, l_k-1, m_k+1)
        else:
            if diff_opp_k:   
                # D^{-}f_i * D^{+}f_k
                if a1 != 0:
                    a1 *= self._recouplingSphHarm(l_i-1, m_i+1, l_k+1, m_k-1)
                if a2 != 0:
                    a2 *= self._recouplingSphHarm(l_i-1, m_i-1, l_k+1, m_k+1)
            else:
                # D^{-}f_i * D^{-}f_k
                if a1 != 0:
                    a1 *= self._recouplingSphHarm(l_i-1, m_i+1, l_k-1, m_k-1)
                if a2 != 0:
                    a2 *= self._recouplingSphHarm(l_i-1, m_i-1, l_k-1, m_k+1)
        aux = 0
        if (a1 - a2):
            aux = (a1 - a2)*self._globalRadialIntegrals(diff_opp_i, diff_opp_k)
        return aux


    def _coeffs_G_positive(self, diff_opp_i, diff_opp_k):
        """
        :diff_opp_i, diff_opp_k (the differential terms) 0 for {-}, 1 for {+}
        """
        if diff_opp_i == 0:
            if self._G_i_pm == 0:
                return 0
            if diff_opp_k:
                # D^{-}f_i * D^{+}f_k
                return (self._G_i_pm * self._G_k_mp)
            else:
                # D^{-}f_i * D^{-}f_k
                if self._G_k_mm == 0:
                    return 0
                return (self._G_i_pm * self._G_k_mm)
        elif diff_opp_k == 0:
            # D^{+}f_i * D^{-}f_k
            if self._G_k_mm == 0: 
                return 0
            return (self._G_i_pp * self._G_k_mm)
        
        else: 
            # D^{+}f_i * D^{+}f_k
            return (self._G_i_pp * self._G_k_mp)
        
    def _coeffs_G_negative(self, diff_opp_i, diff_opp_k):
        """
        :diff_opp_i, diff_opp_k (the differential terms)
        """
        if diff_opp_i == 0:
            if self._G_i_mm == 0:
                return 0
            if diff_opp_k:
                # D^{-}f_i * D^{+}f_k
                return (self._G_i_mm * self._G_k_pp)
            else:
                # D^{-}f_i * D^{-}f_k
                if self._G_k_pm == 0:
                    return 0
                return (self._G_i_mm * self._G_k_pm)
            
        elif diff_opp_k == 0:
            # D^{+}f_i * D^{-}f_k
            if self._G_k_pm == 0: 
                return 0
            return (self._G_i_mp * self._G_k_pm)
        
        else: 
            # D^{+}f_i * D^{+}f_k
            return (self._G_i_mp * self._G_k_pp)
        
    def _recouplingSphHarm(self, l1, m1, l3, m3):
        """ 
        _recoupling Spherical Harmonics:
        [Y_{l_i', m_i'}Y_{l_j, l_j}]* [Y_{l_k', m_k'}Y_{l_l, m_l}]
        
        where l_i' come from the gradient formula, we rename this value to 1 and 3
        """
        l2, m2 = self.wf_j.l, self.wf_j.m_l
        l4, m4 = self.wf_l.l, self.wf_l.m_l
        
        M_L = m1 + m2
        
        if ((l1 + l2) + (l3 + l4))%2 == 1:
            # it cannot be valid L'=l3+l4 if the elements if they don't share parity
            return 0
        if (m3 + m4) != M_L:
            return 0
        
        aux = 0
        for L in range(abs(l1 - l2), l1 + l2 +1, 2):
            # parity clebsh-gordan imply same parity for L and l1+l2
            #L' = L
            if not angular_condition(l3, l4, L):
                continue
            elif abs(M_L) > L:
                continue
            aux += (
                safe_clebsch_gordan(l1, l2, L, 0, 0, 0)
                * safe_clebsch_gordan(l1, l2, L, m1, m2, M_L)
                * safe_clebsch_gordan(l3, l4, L, 0, 0, 0)
                * safe_clebsch_gordan(l3, l4, L, m3, m4, M_L)
            )
        
        return aux * np.sqrt((2*l1 + 1)*(2*l2 + 1)*(2*l3 + 1)*(2*l4 + 1)) /(4*np.pi)
        
        
    def _gradientCoeff(self, m_increase, l_increase, l, m):
        """
        G{m_increase, l_increase} coefficient from the gradient formula:
             G{++} -> Y{l+1}{m+1}, G{+-} -> Y{l-1}{m+1} etc.
             
        :m_increase     1 for (+1) and 0 for (-1)
        :l_increase     1 for (+1) and 0 for (-1)
        :l and 
        :m integers from angular momentum
        
        if l_increase = -1, then there are restricted m values        
        returns <int> 0 in that case
        """
                
        if l_increase == 0:
            # l_sign = -1
            if m_increase:
                # m_sign = +1:   G^{+-}
                aux = (l - m - 1)*(l - m)
                if aux == 0:
                    return 0
            else:
                # m_sign = -1:   G^{--}
                aux = (l + m - 1)*(l + m)
                if aux == 0:
                    return 0
            return np.sqrt(aux/ (2*(2*l - 1)*(2*l + 1)))

        else:
            # l_sign = +1
            if m_increase:
                # m_sign = +1:   G^{++}
                aux = (l + m + 1)*(l + m + 2)
            else:
                # m_sign = -1:   G^{-+}
                aux = (l - m + 1)*(l - m + 2)
            return np.sqrt(aux/ (2*(2*l + 1)*(2*l + 3)))
        
        
    def _globalRadialIntegrals(self, diff_opp_i, diff_opp_k):
        """
        Recursion relation calling for each differential operator 
        D^{+}=d/dr - l/r
        D^{-}=d/dr + (l+1)/r
        
        :diff_opp_i 1 for (+) and 0 for (-) on the bra function
        :diff_opp_k 1 for (+) and 0 for (-) on the ket function
        """
        
        if diff_opp_i:
            # D^{+}f_i
            ci1 = -np.sqrt(self.wf_i.n + self.wf_i.l + 1.5)
            ci2 = -np.sqrt(self.wf_i.n)
        else:
            # D^{-}f_i
            ci1 = np.sqrt(self.wf_i.n + self.wf_i.l + 0.5)
            ci2 = np.sqrt(self.wf_i.n + 1)
        if diff_opp_k:
            # D^{+}f_k
            ck1 = -np.sqrt(self.wf_k.n + self.wf_k.l + 1.5)
            ck2 = -np.sqrt(self.wf_k.n)
        else:
            # D^{-}f_k
            ck1 = np.sqrt(self.wf_k.n + self.wf_k.l + 0.5)
            ck2 = np.sqrt(self.wf_k.n + 1)
        
        l_i = self.wf_i.l - (-1)**diff_opp_i
        l_k = self.wf_k.l - (-1)**diff_opp_k
        
        n_i, n_k = self.wf_i.n, self.wf_k.n
        n_i_incr = self.wf_i.n + (-1)**diff_opp_i
        n_k_incr = self.wf_k.n + (-1)**diff_opp_k
        
        aux  = ci1*ck1*self._singleRadialIntegrals(n_i,      l_i, n_k,      l_k)
        aux += ci1*ck2*self._singleRadialIntegrals(n_i,      l_i, n_k_incr, l_k)
        aux += ci2*ck1*self._singleRadialIntegrals(n_i_incr, l_i, n_k,      l_k)
        aux += ci2*ck2*self._singleRadialIntegrals(n_i_incr, l_i, n_k_incr, l_k)
        
        return (1/self.b_param**2) * aux
        
    
    def _singleRadialIntegrals(self, n_i, l_i, n_k, l_k):
        """
        Individual integral for the four functions with the modified indexes for
        the radial functions (i, k) after the gradient formula
        """
        # TODO: might change l_i for +-1 depending on the operator, apply to a 
        # instance objetct self.L = li+lj+lk+ll  + (dli + dlk)
        big_L = l_i + self.wf_j.l + l_k + self.wf_l.l + l_k
        sum_ = 0
        
        for k1 in range(0, n_i +1):
            a1 = self._aCoeff_nlk(n_i, l_i, k1)
            
            for k2 in range(0, self.wf_j.n +1):
                a2 = self._aCoeff_nlk(self.wf_j.n, self.wf_j.l, k2)
                
                for k3 in range(0, n_k +1):
                    a3 = self._aCoeff_nlk(n_k, l_k, k3)
                    
                    for k4 in range(0, self.wf_l.n +1):
                        a4 = self._aCoeff_nlk(self.wf_l.n, self.wf_l.l, k4)
                        
                        big_K = 2*(k1+k2+k3+k4) + big_L + 2
                        
                        sum_ +=  a1*a2*a3*a4 * self._baseIntegral(big_K)
                        
        return sum_ / (self.b_param**5)
    
    def _aCoeff_nlk(self, n, l, k):
        """
        Access to the memorization object of the coefficient for the radial
        Coefficient for the radial function evaluation in series.
        
        Returns without the 1/sqrt(b^3) normalization factor,  
        """
        key_ = '({},{},{})'.format(n, l, k)
        
        if key_ not in self._a_coeff_nkl_memo:
            
            aux = 0.5*(fact(n) - gamma_half_int(2*(n+l+1))) # + 3*np.log(self.b_param)
            aux += gamma_half_int(2*(n+l+1)) - (fact(n) + fact(n-k) 
                                                + gamma_half_int(2*(k+l+1)))
            
            self._a_coeff_nkl_memo[key_] = (-1)**k * np.exp(aux)
        
        return self._a_coeff_nkl_memo.get(key_)
    
    def _baseIntegral(self, K):
        """
        Access to the memorization object of the base integral
        integral for Gogny:  (r/b_param)^K * V(r)* exp(-2*(r/b)^2)
        """
        # TODO: Extend for different self.PARAMS_FORCE 
        
        if K not in self._baseIntegrals_memo:
            b = self.b_param
            integral_ = (0.5/(b**3)) * np.exp(gamma_half_int(2*K+3))\
                        * talmiIntegral(K, PotentialForms.Gaussian, b, b)
            self._baseIntegral_memo[K] = integral_
        
        return self._baseIntegral_memo.get(K)
#        if diff_opp_i:
#             # D^{+}f_i
#             ci1 = -np.sqrt(self.wf_i.n + self.wf_i.l + 1.5)
#             ci2 = -np.sqrt(self.wf_i.n)
#             if diff_opp_k:
#                 # D^{+}f_i * D^{+}f_k
#                 ck1 = -np.sqrt(self.wf_k.n + self.wf_k.l + 1.5)
#                 ck2 = -np.sqrt(self.wf_k.n)
#             else:
#                 # D^{+}f_i * D^{-}f_k
#                 ck1 = np.sqrt(self.wf_k.n + self.wf_k.l + 0.5)
#                 ck2 = np.sqrt(self.wf_k.n + 1)
#         else:
#             # D^{-}f_i
#             ci1 = np.sqrt(self.wf_i.n + self.wf_i.l + 0.5)
#             ci2 = np.sqrt(self.wf_i.n + 1)
#             if diff_opp_k:
#                 # D^{-}f_i * D^{+}f_k
#                 ck1 = -np.sqrt(self.wf_k.n + self.wf_k.l + 1.5)
#                 ck2 = -np.sqrt(self.wf_k.n)
#             else:
#                 # D^{-}f_i * D^{-}f_k
#                 ck1 = np.sqrt(self.wf_k.n + self.wf_k.l + 0.5)
#                 ck2 = np.sqrt(self.wf_k.n + 1)

class _RadialTwoBodyDecoupled():
    """
    From T. Gonzalez Llarena thesis developments. Matrix elements use total 
    decoupling from the angular part. B/D coefficients are kept in static memory
    pattern for efficiency.
    
    Phi_{n1l1}*Phi_{n2l2}         = [norm_coeff{12}] * sum B * (r/b)**(2*p + l1+l2)
    r(d Phi_{n1l1}/dr)*Phi_{n2l2} = [norm_coeff{12}] * sum D * (r/b)**(2*p + l1+l2)
    
    sum range [0, n1+n2], B/D(n1l1 n2l2, p) coefficients are not b_length dependent.
    and got the norm embedded (thesis definition separate the norm coefficient)
    
    Real coefficients have a factor (4*pi^3) (-)^n1+n2 multiplying, avoided here
    due the factor from the norm coefficient sqrt_(1/2*pi^3). When integrating, 
    factors cancel and left a factor 4*b_length.
        integral(r**2 dr Phi_{1}*Phi_{2}  V(r) Phi_{3}*Phi_{4})
    
    Some proofs and details of B coefficient can be found in:
    Talman, J. D. (1970)
        Some properties of three-dimensional harmonic oscillator wave fucntions.
        Nuclear Physics A, 141(2), 273-288 doi:10.1016/0375-9474(70)90847-x
    
    """
    
    _BCoeff_Decoup_Memo = {}
    _DCoeff_Decoup_Memo = {}
    
    DEBUG_MODE = False 
    
    def _r_dependentIntegral(self, *args):
        """ 
        Define the integral I(n,l(1,2), n',l'(1,2), p, p'), at the end of the 
        transformation.
        """
        raise IntegralException("Abstract method, define the integral")
    
    
    def _B_coeff(self, n1, l1, n2, l2, p):
        """ Same accessor than Moshinsky transformation, from [Llarena]:
        Include the normalization and the factor 2 from (4pi^3/2pi^3)
        B(n1,l1,n2,l2,p) =  c(1, without 2pi3) * c(2) * B(without 4pi3) * 2
        """
        tpl = _B_coeff_memo_accessor(n1, l1, n2, l2, p)
        
        if not tpl in self._BCoeff_Decoup_Memo:
            self._BCoeff_Decoup_Memo[tpl] = self._B_coeff_eval(n1, l1, n2, l2, p)
        
        return self._BCoeff_Decoup_Memo[tpl]
    
    def _D_coeff(self, n1, l1, n2, l2, p):
        """ D coefficients cannot access by B memory accessor, wave functions 
        are not interchangeable, the first function is derivated """
        
        tpl = ','.join(map(lambda x: str(x), (n1, l1, n2, l2, p)))
        
        if not tpl in self._DCoeff_Decoup_Memo:
            self._DCoeff_Decoup_Memo[tpl] = self._D_coeff_eval(n1, l1, n2, l2, p)
        
        return self._DCoeff_Decoup_Memo[tpl]
    
    
    def __sum_aux_denominator(self, n1, l1, n2, l2, p, k):
        
        denominator = sum(
            [fact(k), fact(n1 - k), fact(p - k), fact(n2 + k - p),
            gamma_half_int(2*(k + l1) + 3),  
            gamma_half_int(2*(p - k + l2) + 3)]  ## double factorial (n - 1)
        )
        return np.exp(denominator)
    
    def _B_coeff_eval(self, n1, l1, n2, l2, p):
        """ normalization factor multiplied here, factor 2 = (4*pi^3)/(2*pi^3) """
        sum_ = 0
        
        k_max = min(p, n1)
        k_min = max(0, p - n2)
        for k in range(k_min, k_max +1):
            sum_ += 1 / self.__sum_aux_denominator(n1, l1, n2, l2, p, k)
        
        return ((-1)**(p)) * sum_ * 2 * self._norm_coeff(n1, l1, n2, l2)
    
    
    def _D_coeff_eval(self, n1, l1, n2, l2, p):
        """ normalization factor multiplied here, factor 2 = (4*pi^3)/(2*pi^3) """
        sum_ = 0
        
        k_max = min(p, n1)
        k_min = max(0, p - n2)
        for k in range(k_min, k_max +1):
            sum_ += (2*k + l1) / self.__sum_aux_denominator(n1, l1, n2, l2, p, k)
        
        return ((-1)**(p)) * sum_ * 2 * self._norm_coeff(n1, l1, n2, l2)
    
    def _norm_coeff(self, n1, l1, n2, l2):
        """ Also extracted from Apendix D.1 of the thesis, 
        factor (1 / 2*pi^3) extracted."""
                
        aux =  fact(n1) + gamma_half_int(2*(n1 + l1) + 3)
        aux += fact(n2) + gamma_half_int(2*(n2 + l2) + 3)
        
        return np.exp(0.5 * aux)   
    
    @staticmethod
    def integral(type_integral, wf1_bra, wf2_bra, wf1_ket, wf2_ket, *args): 
        """
        type_integral:
            [1] differential type:  d(bra_1)/d_r * bra_2 * ket_1 *  (ket_2/r) 
            [2] 1/r w.f. type    :   (bra_1/r)   * bra_2 * ket_1 *  (ket_2/r)
        """
        self = _RadialTwoBodyDecoupled()
        args = (wf1_bra, wf2_bra, wf1_ket, wf2_ket)
        raise IntegralException("Abstract method, implement me (use _B_coeff, "
                                "_D_coeff, _norm_coeff and define _r_dependentIntegral")
        
        return self._integral(type_integral, *args)
        
    
    def _integral(self, type_integral, wf1_bra, wf2_bra, wf1_ket, wf2_ket, b_length):
        raise IntegralException("Abstract method, implement me (use _B_coeff, "
                                "_D_coeff, _norm_coeff and define _r_dependentIntegral")
        

class _RadialIntegralsLS(_RadialTwoBodyDecoupled):
    
    _r_integrals_memo = {}
    
    @classmethod
    def _r_dependentIntegral(cls, No2):
        """ Integral r^2*No2 exp(-2* r^2), b lengths extracted (b**6)
        :No2 stands for N over 2, being N = 2*(p+p') + sum{l} (+2 opt.)"""
        
        # l1 + l2 + l1_q + l2_q is even
        if not No2 in cls._r_integrals_memo:
            aux = np.exp(gamma_half_int(2*No2 + 1) - ((No2 + 1.5)*np.log(2)))
            cls._r_integrals_memo[No2] = aux
        return cls._r_integrals_memo[No2]
    
    @staticmethod
    def integral(type_integral, wf1_bra, wf2_bra, wf1_ket, wf2_ket, b_length): 
        """
        type_integral:
            [1] differential type:  d(bra_1)/d_r * bra_2 * ket_1 *  (ket_2/r) 
            [2] 1/r w.f. type    :   (bra_1/r)   * bra_2 * ket_1 *  (ket_2/r)
            [3] abs w.f. type    :    bra_1      * bra_2 * ket_1 *   ket_2   
        """
        self = _RadialIntegralsLS()
        args = (wf1_bra, wf2_bra, wf1_ket, wf2_ket, b_length)
        return self._integral(type_integral, *args)
    
    def _integral(self, type_integral, wf1_bra, wf2_bra, wf1_ket, wf2_ket, b_length):
        
        assert type_integral in (1,2,3), "type_integral can only be 1,2 or 3"
        
        n1_q, l1_q = wf1_bra.n, wf1_bra.l
        n2_q, l2_q = wf2_bra.n, wf2_bra.l
        n1, l1 = wf1_ket.n, wf1_ket.l
        n2, l2 = wf2_ket.n, wf2_ket.l
        
        sum_ = 0
        if self.DEBUG_MODE:
            XLog.write("R_int", type=type_integral, wf1=wf1_bra, wf2=wf2_bra, wf3=wf1_ket, wf4=wf2_ket)
        
        for p in range(n1+n2 +1):
            ket_coeff = self._B_coeff(n1, l1, n2, l2, p)
            
            if self.DEBUG_MODE: XLog.write("Ip", p=p, ket_C=ket_coeff)
            for p_q in range(n1_q+n2_q +1):
                bra_b = self._B_coeff(n1_q, l1_q, n2_q, l2_q, p_q)
                
                No2 = (p + p_q) + ((l1 + l2 + l1_q + l2_q)//2)
                if type_integral != 3:
                    I_1 = self._r_dependentIntegral(No2)
                    if self.DEBUG_MODE:
                        XLog.write("Ip_q", pq=p_q, bra_B=bra_b, N=No2, I_1=I_1,
                                   type_int=type_integral)
                
                if   type_integral == 1:
                    bra_d = self._D_coeff(n1_q, l1_q, n2_q, l2_q, p_q)
                    
                    I_2 = self._r_dependentIntegral(No2 + 1)
                    aux = ket_coeff * ((bra_d*I_1) - (bra_b*I_2))
                    sum_ += aux
                    if self.DEBUG_MODE:
                        XLog.write("Ip_q", I_2=I_2, bra_D=bra_d, val= aux)
                elif type_integral == 2:
                    aux = ket_coeff * bra_b * I_1
                    sum_ += aux
                    if self.DEBUG_MODE: XLog.write("Ip_q", val= aux)
                else: ## independent w.f. integral.
                    I_3 = self._r_dependentIntegral(No2+1)
                    aux = ket_coeff * bra_b * I_3
                    sum_ += aux
                    if self.DEBUG_MODE: XLog.write("Ip_q", val= aux, I_3=I_3)
        
        # norm_fact = np.prod([self._norm_coeff(wf) for wf in 
        #                                 (wf1_bra, wf2_bra, wf1_ket, wf2_ket)])
        norm_fact = b_length
        if self.DEBUG_MODE:
            XLog.write("R_int", sum=sum_, norm=norm_fact, value=norm_fact*sum_)
            
        return  norm_fact * sum_


class _RadialDensityDependentFermi(_RadialTwoBodyDecoupled):
    
    """
    Args: call from <static method> integral(*args)
        :wave_functions: wave functions <QN_1body_radial> bra_ 1, 2 ket_ 1, 2
        :b_length: oscillator parameter 
        :A <int>:  mass number
        :alpha: coefficient over the density (1/3 only valid)
    """
    
    _ASSOCIATED_LAGUERRE = True # True for associated Laguerre radial integral
    _DENSITY_APROX = False # True for profile, False for Fermi occupation filling
    _instance = None
    
    _R_DIM = 20
    
    @staticmethod
    def _getInstance():
        if _RadialDensityDependentFermi._instance == None:
            _RadialDensityDependentFermi._instance = _RadialDensityDependentFermi()
            _RadialDensityDependentFermi._instance.A = -1
            _RadialDensityDependentFermi._instance.Z = -1
            _RadialDensityDependentFermi._instance._nuclear_density = None
        return _RadialDensityDependentFermi._instance
    
    def _checkNucleusInstance(self, A, Z):
        """ check if the density has been calculated """
        if hasattr(self, 'A') and hasattr(self, 'Z'):
            if A == self.A and Z == self.Z:
                return True
        return False
    
    def _FermiDensity(self, A, r, Z=None):
        """
        Radial Fermi density without the exponential (for real Fermi densty 
        requires a 4pi angular factor: sum(|Y_lm|^2)[-l, l]) = (2*l + 1) / 4*pi
        :r = r (the factor 1/b should be multiplied externally to r if you want
                the real density for the system of length b)
        :Z <int or None> default define an hypothetical symmetrical nucleus :
            Z=A//2, N=A//2 + A%2
        """
        if self._checkNucleusInstance(A, Z):
            return self._nuclear_density
        else:            
            self.A = A
            self.Z = Z if Z else A//2
            self.N = A - self.Z
            
            if self.Z > A: 
                raise IntegralException("Fermi density for A < Z !!")
        
        ## factor for the r = (r/b) of the density to be (r/b_core)=(r/b)(b/b_c)
        xBoBcore = 1 
        if abs(self.b_length_core - self.b_length) > 0.01:
            xBoBcore = self.b_length / self.b_length_core
        
        aux = 0.0 * r
        occupied_states = getStatesAndOccupationOfFullNucleus(self.Z, self.N)
        for i in range(len(occupied_states)):
            sps, occ_Z, occ_N = occupied_states[i]
            ni, li, _ = sps
            
            #last = 0 if i < len(occupied_states) - 1 else self.A%2 
            occ = occ_Z + occ_N# + last
            if occ > 0:
                aux_i = 0.0 * r
                for p in range(2*ni +1):
                    aux_r  = ((r * xBoBcore)**(2*(p + li)))   #
                    aux_i += self._B_coeff(ni,li,ni,li, p) * aux_r
                aux += occ * aux_i
        
        ## we consider the internal integral to be exp(- (r/ b')^2) if b /= b_c
        if abs(self.b_length_core - self.b_length) > 0.01:
            aux *= np.exp(r  * (1 - (xBoBcore**2)))
            
        # import matplotlib.pyplot as plt
        # plt.plot(r/((2+0.3333)**0.5), (1 / (4*np.pi*(self.b_length_core**3))) 
        #             * aux  / np.exp(np.power(r/((2+0.3333)**0.5),2)))
        # plt.show()
        
        self._nuclear_density = aux
        return aux
    
    def _DensityProfile(self, A, r, Z=None):
        """ 
        Approximates the density with the form 
                rho =    rho0 / (1 + exp((r - R)/a)) 
        rho0 chosen for the R^3 integral to be A.
        R = r0 * A^(1/3)
        a = stiffness parameter
        """
        if self._checkNucleusInstance(A, Z):
            return self._nuclear_density
        else:            
            self.A = A
            self.Z = Z if Z else A//2
        
        # r0, a = 1.25, 0.65 # Borh Mothelson
        # r0, a = 1.1,  0.5  # Greiner 
        r0, a = 1.25, 0.524  # Kranne
        # r0, a = 
        b = 0.05  ## parameter for the bump
        
        R  = r0 * (self.A**(1/3))
        profile = lambda rr: (rr**2)*(1 + b*(rr**2)) / (1 + np.exp((rr - R)/a))
        
        ## 4*np.pi * is not done here, radial profile require the factor for the
        ## global integral = A, but the def. has no angular part, so the 1/4pi
        ## is extracted to the constant of the global radial integration. 
        aux  =  GaussianQuadrature.legendre(profile, 0, 15, 50)
        rho0 = self.A / aux   
        # For most nuclei rho0 = 0.17 fm-3
        
        ## This shape is required because we are integrated with Laguerre
        ## so we extract the exponential.
        ## delafuen: the core length has no place here.
        aux = (1 + b*(r**2)) / (1 + np.exp((r - R) / a))
        aux = rho0 * np.exp((r / self.b_length)**2) * aux 
        
        # rOb = r / self.b_length
        # aux2 = 4*np.pi * np.exp(rOb/2)
        # import matplotlib.pyplot as plt
        # plt.plot(rOb, aux)
        # plt.show()
        
        self._nuclear_density = aux
        return self._nuclear_density
    
    def _auxFunction(self, x, No2, A, Z, alpha):
        """ Function to be inserted in the Laguerre_ integral. """
        u = np.sqrt(x / (alpha + 2))
        if self._DENSITY_APROX:
            # Fermi profile integral
            density = self._DensityProfile(A, self.b_length * u, Z)
        else:
            # Fermi SHO density (density matrix = delta(i <= A))
            density = self._FermiDensity(A, u , Z)
        
        if self._ASSOCIATED_LAGUERRE:
            return (x**(No2 - 1.0)) * (density ** alpha)
        else:
            return (x**(No2 - 0.5)) * (density ** alpha)
    @staticmethod
    def _auxFunction_static(x, No2, A, Z, alpha):
    
        self = _RadialDensityDependentFermi._getInstance()
        return self._auxFunction(x, No2, A, Z, alpha)
    
    def _r_dependentIntegral(self, No2, A, Z, alpha):
        """ Radial integral for  (factors omitted)
            exp(-(alpha + 2)r**2) * r^2*No2 * fermi_density
        :No2 stands for N over 2, being N = 2*(p+p') + sum{l} + 2 
        """
        cte = 2 * ((alpha + 2)**(No2 + 0.5))
        ## cte is the same for both methods
        N_integr = self._R_DIM   ## 100
        if abs(self.b_length_core - self.b_length) > 0.01:
            N_integr = self._R_DIM  ## 180
        aux = GaussianQuadrature.laguerre(self._auxFunction_static, 
                                          (N_integr, 0.5), No2, A, Z, alpha)
        if self.DEBUG_MODE: XLog.write("Ip_q", Lag_int=aux)
        return aux / cte
    
    @staticmethod
    def integral(wf1_bra, wf2_bra, wf1_ket, wf2_ket, b_length, A, Z, alpha, 
                 b_length_core): 
        """
        :wfs: wave fucntions bra 1, 2 ket 1, 2
        :b_length: oscillator parameter 
        :A: <int>  mass number
        :Z: <int or None>
        :alpha: coefficient over the density
        :b_length_core: (opt) oscillator length of the core might be different
        """
        self = _RadialDensityDependentFermi._getInstance()
        args = (wf1_bra, wf2_bra, wf1_ket, wf2_ket, b_length, A, Z, alpha)        
        return self._integral(*args, b_length_core=b_length_core)
    
    def _integral(self, wf1_bra, wf2_bra, wf1_ket, wf2_ket, b_length, A, Z, alpha,
                  b_length_core=None):
                
        n1_q, l1_q = wf1_bra.n, wf1_bra.l
        n2_q, l2_q = wf2_bra.n, wf2_bra.l
        n1, l1 = wf1_ket.n, wf1_ket.l
        n2, l2 = wf2_ket.n, wf2_ket.l
        
        self.b_length = b_length    ## keep for _DensityProfile
        self.b_length_core = b_length
        if b_length_core:           ## set another b lenght for the core
            self.b_length_core = b_length_core
        if (l1 + l2 + l1_q + l2_q) % 2 == 1:
            raise IntegralException("(l1 + l2 + l1_q + l2_q) not even")
        
        sum_ = 0
        if self.DEBUG_MODE:
            XLog.write("R_int", a=wf1_bra, b=wf2_bra, c=wf1_ket, d=wf2_ket)
        
        # for p1 in range(n1_q+n2_q +1):
        for p1 in range(n1_q+n1 +1):
            ket_coeff = self._B_coeff(n1_q, l1_q, n1, l1, p1)
            # ket_coeff = self._B_coeff(n1_q, l1_q, n2_q, l2_q, p1)
            
            if self.DEBUG_MODE: XLog.write("Ip", p1=p1, ket_C=ket_coeff)
            for p2 in range(n2_q+n2 +1):
                bra_coeff = self._B_coeff(n2_q, l2_q, n2, l2, p2)
            # for p2 in range(n1+n2 +1):
                # bra_coeff = self._B_coeff(n1, l1, n2, l2, p2)
                # l1 + l2 + l1_q + l2_q is even
                No2 = (p1 + p2) + ((l1 + l2 + l1_q + l2_q)//2) + 1
                I_dd = self._r_dependentIntegral(No2, A, Z, alpha)
                
                if self.DEBUG_MODE:
                    XLog.write("Ip_q", p2=p2, bra_C=bra_coeff, N=No2, Idd=I_dd)
                
                aux = ket_coeff * bra_coeff * I_dd
                sum_ += aux
                if self.DEBUG_MODE: XLog.write("Ip_q", val= aux)
        
        norm_fact = 1 / ((4*np.pi)**alpha * (b_length**3)) # b_length**(3*(1 - 2))
        # norm_fact *= 4*np.pi
        if not self._DENSITY_APROX:
            norm_fact /=  self.b_length_core**(3*alpha)
        
        if self.DEBUG_MODE:
            XLog.write("R_int", sum=sum_, norm=norm_fact, value=norm_fact*sum_)
            
        return  norm_fact * sum_


class _RadialMultipoleMoment(_RadialTwoBodyDecoupled):
    
    """ Integral of a one body matrix element  <a | r^lambda | b> for the 
    multipole interaction. """
    
    @staticmethod
    def integral(moment, wf_bra, wf_ket, b_length): 
        """
        :moment <integer>
        """
        
        self = _RadialMultipoleMoment()
        args = (wf_bra, wf_ket, b_length)
        return self._integral(moment, *args)
    
    
    def _integral(self, moment, wf_bra, wf_ket, b_length):
        
        assert isinstance(moment, int) and moment >= 0, \
            "The moment must be non-negative integer"
            
        n_q, l_q = wf_bra.n, wf_bra.l
        n, l = wf_ket.n, wf_ket.l
        
        sum_ = 0
        if self.DEBUG_MODE:
            XLog.write("R_int", lamb=moment, wf1=wf_bra, wf2=wf_ket)
        
        for p in range(n_q+n +1):
            coeff = self._B_coeff(n_q, l_q, n, l, p)
            
            if self.DEBUG_MODE: XLog.write("Ip", p=p, ket_C=coeff)
            
            No2 = p + (l + l_q + moment)//2 + 1   # 3/2 = 1 + 1/2(in the integral)
            I_rm = self._r_dependentIntegral(No2)
            
            sum_ += coeff * I_rm
        
        norm_fact = b_length ** moment
        if self.DEBUG_MODE:
            XLog.write("R_int", sum=sum_, norm=norm_fact, value=norm_fact*sum_)
            
        return norm_fact * sum_
    
    
    def _r_dependentIntegral(self, No2):
        """ Integral r^2*No2 exp(-r^2), b lengths extracted
        :No2 stands for N over 2, being N = 2*(p) + (l1+ l2+ _Lambda) + 1
        
        Note: this integral is not the same as the LS or fermi (two body), those
        include a factor 2 in the exp(-r^2) that has to be included in the r/b 
        series. """
        
        return 0.5 * np.exp(gamma_half_int(2*No2 + 1))
    




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    dr = 0.001
    b_length = 1.2
    
    N_min = 6
    N_max = 7
    NZ_states = getStatesAndOccupationUpToLastOccupied(N_max, N_min)
    NZ_states = getStatesAndOccupationOfFullNucleus(N_min + 2, N_max, N_min)
    
    A = 0
    A_prev = max(1, 2*N_min + 1)
    Z = 0
    r = np.arange(0, 6, dr)
    rOb = r / b_length
    for spss, z, n in NZ_states:
        spss = shellSHO_Notation(*spss)
        Z += z
        A = A_prev + z + n
        for a_ in range(A_prev, A + 1):
            #A += 2*a
             
            aux = _RadialDensityDependentFermi()
            
            den = aux._FermiDensity(a_, rOb, z) 
            # den = aux._auxFunction(rOb, 6, A, 1/3)
            den /= np.exp(rOb**2)
            # den /= np.exp((7/3) * rOb**2)
            print(a_,'(', spss, ') integral=', 
                  b_length**(-3) * sum(den* np.power(r, 2))*dr)
                ## 4pi factor cancels with the normalization of the density
                ## but _FermiDensity dont have factor 1/(4pi * b^3)
            
            if a_ == A:
                plt.plot(rOb, den, label=f"A[{A}]={spss}")
            else:
                plt.plot(rOb, den, label=f"A[{A}]({a_})={spss}")
        A_prev = A
    plt.legend()
    plt.show()