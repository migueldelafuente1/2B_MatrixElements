'''
Created on Mar 12, 2021

@author: Miguel
'''
import numpy as np

from helpers.Enums import PotentialForms
from helpers.Helpers import gamma_half_int, fact, angular_condition,\
    safe_clebsch_gordan, double_factorial

class IntegralException(BaseException):
    pass

def talmiIntegral(p, potential, b_param, mu_param, n_power=0):
    """
    :p          index order
    :potential  form of the potential, from Poten
    :b_param    SHO length parameter
    :mu_param   force proportional coefficient (by interaction definition)
    :n_power    auxiliary parameter for power dependent potentials
    """
    # TODO: Might implement potential args by Enum and select them in the switch
    #potential = potential.lower()
    
    if potential == PotentialForms.Gaussian:
        return (b_param**3) / (1 + 2*(b_param/mu_param)**2)**(p+1.5)
    
    elif potential == PotentialForms.Coulomb:
        return (b_param**2) * np.exp(fact(p) - gamma_half_int(2*p + 3)) / (2**.5)
    
    elif potential == PotentialForms.Yukawa:
        sum_ = 0.
        for i in range(2*p+1 +1):
            aux = fact(2*p + 1) - fact(i) - fact(2*p + 1 - i)
            if i % 2 == 0:
                aux += fact((i + 1)//2)
            else:
                aux += gamma_half_int(i + 1)
            
            aux += (2*p + 1 - i) * np.log(b_param/(2*mu_param))
            
            sum_ += (-1)**(2*p + 1 - i) * np.exp(aux)
            
        sum_ *= mu_param * (b_param**2) * np.exp((b_param/(2*mu_param))**2) 
        sum_ /= np.exp(gamma_half_int(2*p + 3))
        
        return sum_
    
    elif potential == PotentialForms.Power:
        if n_power == 0:
            return b_param**3
        aux = gamma_half_int(2*p + 3 + n_power) - gamma_half_int(2*p + 3)
        
        return np.exp(aux + n_power*(3*np.log(b_param) - np.log(mu_param)))
        
    elif potential == PotentialForms.Gaussian_power:
        raise IntegralException("Talmi integral 'gaussian_power' not implemented")
    else:
        raise IntegralException("Talmi integral [{}] is not defined, valid potentials: {}"
                        .format(potential, PotentialForms.members()))


def gaussianIntegralQuadrature(function, a, b, order, *args, **kwargs):
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
    
    if not 'roots_legendre' in globals():
        from scipy.special import roots_legendre
    A = (b - a) / 2
    B = (b + a) / 2
    
    x_i, w_i = roots_legendre(order, mu=False)
    x_i = A*x_i + B
    
    #integral = A * sum(w_i * function(x_i, *args, **kwargs))
    #return integral
    
    integral = 0.0
    for i in range(len(x_i)):
        integral += w_i[i] * function(x_i[i], *args, **kwargs)
    
    #print("order", order, "=",integral)
    return A * integral
    
    

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
    
    
class _RadialIntegralsLS():
    
    """
    From T. Gonzalez Llarena thesis developments
    
    """
    
    def _TalmiIntegral(self, p, p_q, l1, l2, l1_q, l2_q):
        
        # l1 + l2 + l1_q + l2_q is even
        N = (p + p_q) + ((l1 + l2 + l1_q + l2_q)//2)
        
        return np.exp(gamma_half_int(2*N + 1)) / (2**(N + 1.5))
    
    def __sum_aux_denominator(self, n1, l1, n2, l2, p, k):
        
        denominator = sum(
            [fact(k), fact(n1 - k), fact(p - k), fact(n2 + k - p),
            double_factorial(2*(k + l1) + 1), 
            double_factorial(2*(p - k + l2) + 1)]
        )
        
        return np.exp(denominator)
    
    def _B_coeff(self, n1, l1, n2, l2, p):
        
        sum_ = 0
        
        k_max = min(p, n1)
        k_min = max(0, p - n2)
        for k in range(k_min, k_max +1):
            sum_ += 1 / self.__sum_aux_denominator(n1, l1, n2, l2, p, k)
        
        return 4*(np.pi**2)*(2**(l1 + l2 + 2 + p)) * ((-1)**(n1 + n2 + p)) * sum_
    
    def _D_coeff(self, n1, l1, n2, l2, p):
        
        sum_ = 0
        
        k_max = min(p, n1)
        k_min = max(0, p - n2)
        for k in range(k_min, k_max +1):
            sum_ += (2*k + l1) / self.__sum_aux_denominator(n1, l1, n2, l2, p, k)
        
        return 4*(np.pi**2)*(2**(l1 + l2 + 2 + p)) * ((-1)**(n1 + n2 + p)) * sum_ 
    
    def _norm_coeff(self, wf):
        """ Also extracted from Apendix D.1 of the thesis """
                
        aux = fact(wf.n) + gamma_half_int(2*(wf.n + wf.l) + 3)
        
        return ((-1)**wf.n) * np.sqrt(np.exp(aux) / (2 * (np.pi**3)))       
    
    
    @staticmethod
    def integral(type_integral, wf1_bra, wf2_bra, wf1_ket, wf2_ket): 
        """
        type_integral:
            [1] differential type:  d(bra_1)/d_r * bra_2 * ket_1 *  (ket_2/r) 
            [2] normal type      :   (bra_1/r)   * bra_2 * ket_1 *  (ket_2/r)
        """
        self = _RadialIntegralsLS()
        return self._integral(type_integral, wf1_bra, wf2_bra, wf1_ket, wf2_ket)
        
    
    def _integral(self, type_integral, wf1_bra, wf2_bra, wf1_ket, wf2_ket):
        
        assert type_integral in (1, 2), "type_integral can only be 1 or 2"
        
        n1_q, l1_q = wf1_bra.n, wf1_bra.l
        n2_q, l2_q = wf2_bra.n, wf2_bra.l
        n1, l1 = wf1_ket.n, wf1_ket.l
        n2, l2 = wf2_ket.n, wf2_ket.l
        
        sum_ = 0
        
        for p_q in range(n1_q+n2_q +1):
            
            if type_integral == 1:
                bra_coeff = self._D_coeff(n1_q, l1_q, n2_q, l2_q, p_q)
            else:
                bra_coeff = self._B_coeff(n1_q, l1_q, n2_q, l2_q, p_q)
            
            for p in range(n1+n2 +1):
                ket_coeff = self._B_coeff(n1, l1, n2, l2, p)
                
                I_ = self._TalmiIntegral(p, p_q, l1, l2, l1_q, l2_q)
                sum_ += bra_coeff * ket_coeff * I_
        
        
        norm_fact = np.prod([self._norm_coeff(wf) for wf in 
                                        (wf1_bra, wf2_bra, wf1_ket, wf2_ket)])
        return  norm_fact * sum_
                
            
            
        
