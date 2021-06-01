'''
Created on May 24, 2021

@author: Miguel

This module evaluate the two body matrix element for a gaussian interaction
V_12 = V0 * exp(-|r1 - r2|^2 / mu^2)
on the simple harmonic oscillator functions using the multipole expansion.

    < a, b (L) || V_12 || c, d (L') >
'''

import numpy as np
from helpers.Helpers import fact, gamma_half_int, safe_3j_symbols, safe_racah
from helpers.integrals import gaussianIntegralQuadrature
from helpers.WaveFunctions import QN_2body_L_Coupling, QN_1body_radial
from scipy.constants.constants import lb


#===============================================================================
# RADIAL SHO COEFFICIENTS
#
# R_nl = b^-3/2 * exp_(-r^2/2b^2) * SUM{k} sho_coeff(nkl) (-)^k (r/b)^(2*k+l)
#
#===============================================================================
def _norm_sho(n, l):
    """ return logarithmic value (increase precission for very large n values) """
    #n, l = sp_state.n, sp_state.l
    # Suhonen_
    #aux = fact(n) - gamma_half_int(2*(n + l) + 3)
    #return np.sqrt(2 * np.exp(aux) / ((np.pi**0.5) * b_length**3))
    
    # Moshinsky_ or Llarena_
    aux = fact(n) + gamma_half_int(2*(n + l) + 3)
    #return 0.5 * (np.log(2 / (b_length**3)) + aux)
    return 0.5 * (np.log(2) + aux)

def _sho_series_coeff(sp_state, k):
    """ return logarithmic value. (insert the -1^k phase from outside) """
    n, l = sp_state.n, sp_state.l
    # Suhonen_
    #aux = ((fact(n) + gamma_half_int(2*(n + l) + 3)) 
    #       - (fact(k) + fact(n-k) + gamma_half_int(2*(k + l) + 3)))
    
    # Moshinsky_ or Llarena_
    aux = fact(k) + fact(n - k) + gamma_half_int(2*(k + l) + 3)
    
    return  _norm_sho(n, l) - aux #gamma_half_int(2*(n + l) + 3)

def _radialSHOfunction(r, b, sp_st):
    """ Function to plot R_nl(r, b) """
    r_b = r/b
    
    aux = 0.0
    if r is isinstance(r, np.ndarray):
        aux = np.zeros(len(r))
    
    for k in range(sp_st.n +1):
        aux += np.exp(_sho_series_coeff(sp_st, k)) * ((-1)**k) * (r_b**(2*k + sp_st.l))
    
    return (b**-1.5) * np.exp(-0.5*(r_b**2)) * aux

#===============================================================================
# TESTING NORMALIZATION
#===============================================================================
def _testNormalizationSingleRadialFunction(n, l, b_length=1):
    
    sp_state = QN_1body_radial(n, l)
    
    N_coeff2 = fact(n) + gamma_half_int(2*(n+l)+3)# / np.sqrt(np.pi)
    #N_coeff_mine = _norm_sho(sp_state, b_length)
    
    sum_ = 0.
    sum_mine = 0.
    for k1 in range(n +1):
        aux1 = -(fact(k1) + fact(n-k1) + gamma_half_int(2*(k1+l)+3))
        aux1_mine = _sho_series_coeff(sp_state, k1)
        for k2 in range(n +1):
            aux2 = -(fact(k2) + fact(n-k2) + gamma_half_int(2*(k2+l)+3))
            aux2_mine = _sho_series_coeff(sp_state, k2)
            
            aux = np.exp(N_coeff2 + aux1 + aux2 + gamma_half_int(2*(k1+k2+l) + 3))
            aux_mine = np.exp(aux1_mine + aux2_mine + gamma_half_int(2*(k1+k2+l) + 3))
            
            sum_ += aux * ((-1)**(k1 + k2))
            sum_mine += 0.5 * aux_mine * ((-1)**(k1 + k2))
            
    
    return sum_, sum_mine
            
def _testNormalizationRadialFunctionsRange(N_max=5, b_length=1):
        
    print("Test Normalization for SHO coefficients, b length = ", b_length, '\n')
    count = 0
    fails = [0, 0]
    for l in range(N_max +1):
        for n in range((N_max - l)//2 +1):
            
            sum_, sum_mine = _testNormalizationSingleRadialFunction(n, l, b_length)
            
            print("n,l: ({:2} {:2}) = {:11.9f} , mine = {:11.9f}".format(n, l, sum_,
                                                                       sum_mine))
            if abs(sum_ - 1) > 1.e-9:
                fails[0] += 1
            if abs(sum_mine - 1) > 1.e-9:
                fails[1] += 1
            count += 1
    print("\nFAIL [{} of {}] , FAILs with my definition [{} of {}]"
            .format(fails[0], count, fails[1], count))

def _plot_functions(a, b, c, d, b_length, mu_parameter, V_0, r_Max, show=False):
    
    import matplotlib.pyplot as plt
    import time
    
    r = np.arange(0, r_Max, 0.01)
    
#     a = bra.sp_state_1
#     b = bra.sp_state_2
#     c = ket.sp_state_1
#     d = ket.sp_state_2
    
    gauss = V_0 * np.exp(-((r)/mu_parameter)**2)
    
    f = plt.figure(time.strftime("%H%M%S"))
    for label, sp_st in (('a', a), ('b', b), ('c', c), ('d', d)):
        f = _radialSHOfunction(r, b_length, sp_st)
        label = 'g_' + label + ':' + str(sp_st)
        plt.plot(r, f, label=label)
    plt.plot(r, gauss, label="V(1,2) Gaussian")
    
    plt.legend()
    plt.title('Radial Functions (b={} fm) \nand Gaussian Interaction (mu={} fm) On the Range'
              .format(b_length, mu_parameter))
    plt.xlabel('r (fm)')
    plt.ylabel('R_nl(r; b), V(r=r1-r2) [MeV]')
    
    if show:
        plt.show()
    
    
    
    
#==============================================================================
# INTEGRAL FUNCTIONS 
#==============================================================================
 
#  Coefficient for the Legendre_ polynomial integral
def _Legendre_coeffs(multipole_ord, i):
    """ 
    Return the c_[lambda][i] and the phase for the expansion of the gaussian
    in Legendre polynomials:
    
    v_lambda = exp(-(r1^2 + r2^2)/mu^2) 
        * sum_[i=0: floor(lambda/2)] {
            C_[lambda, i] * (exp(+2*r1*r2/mu^2) * sum1_[j] 
                             + (-)^phase exp(-2*r1*r2/mu^2) * sum2_[j])
                             }
    return C_[lambda, i] (LOGARITHMIC value), phase
    """
    aux_c  = fact(2*(multipole_ord - i)) - (fact(i) + fact(multipole_ord - i))
    aux_c -= (multipole_ord + 1)*np.log(2)
    #aux_c = np.exp(aux_c)
    #aux_c *= ((2*multipole_ord + 1) / (2**(multipole_ord + 1)))
    
    phase = multipole_ord - 2*i + 1
    return aux_c, phase


class _Args:
    
    a_st = 'a_st'
    b_st = 'b_st'
    c_st = 'c_st'
    d_st = 'd_st'
    
    L_ac = 'L_ac'
    L_bd = 'L_bd'
    
    b_length  = 'b_length'
    mu_param = 'mu_param'
    mu_b_coef = 'mu_b_coef'
    
    multipole_ord = 'multipole_ord'
    
    k_a = 'k_a'
    k_b = 'k_b'
    k_c = 'k_c'
    k_d = 'k_d'
    
    i   = 'i'
    j   = 'j'
    
    r1 = 'r1'
    
    r_Max = 'r_Max'
    quad_ord = 'quad_ord'
    
# radial integral for the overlap v_k(r1, r2) on r2 w.f.
def _gaussianFunction(r2, r1, mu_b_coef, mu_param):
    power_ = (-(r2**2) * mu_b_coef) + (2 * r1 * r2 /(mu_param**2))
    return np.exp(power_)

def _r2_dependentFunction(r2, **kwargs):
    """ 
    The positive or negative part in the gaussian exponent function will 
    appear with the r1 sign 
    """
    j    = kwargs.get(_Args.j) 
    L_bd = kwargs.get(_Args.L_bd)
    r1   = kwargs.get(_Args.r1)
    mu_b_coef = kwargs.get(_Args.mu_b_coef)
    mu_param  = kwargs.get(_Args.mu_param)
    
    aux = r2**(L_bd - (j + 1))
    aux *= _gaussianFunction(r2, r1, mu_b_coef, mu_param)
    return aux

def _coeff_r2_integral(multipole_ord, i, j, mu2r1_const):
    """ Return (not logarithm value)"""
    aux  = mu2r1_const**(j + 1)   #(mu_param**2 / (2*r1))**(j+1)
    aux /= np.exp(fact(multipole_ord - (2*i) - j))
    return aux

def _radialIntegral_over_r2(**kwargs):
    
    b_st = kwargs.get(_Args.b_st)
    d_st = kwargs.get(_Args.d_st)
    r1   = kwargs.get(_Args.r1)
    b_length = kwargs.get(_Args.b_length)
    mu_param = kwargs.get(_Args.mu_param)
    multipole_ord = kwargs.get(_Args.multipole_ord)
    r_Max       = kwargs.get(_Args.r_Max) * b_length
    quad_order  = kwargs.get(_Args.quad_ord)
    
    mu2r1_const = (mu_param**2) / (2 * r1)
    sum_ = 0.0
    for i in range(0, multipole_ord//2 +1):
        kwargs[_Args.i] = i
        
        legendre_coeff, phase = _Legendre_coeffs(multipole_ord, i)
        
        for k_b in range(b_st.n +1):
            kwargs[_Args.k_b] = k_b
            alpha_nlk_b = _sho_series_coeff(b_st, k_b)
            
            for k_d in range(d_st.n +1):
                kwargs[_Args.k_d] = k_d
                alpha_nlk_d = _sho_series_coeff(b_st, k_d)
                
                L_bd = 2*(k_b + k_d) + (b_st.l + d_st.l) + 2
                kwargs[_Args.L_bd] = L_bd
                
                aux = (alpha_nlk_b + alpha_nlk_d + legendre_coeff
                        - (L_bd * np.log(b_length)))
                
                aux = ((-1)**(k_b + k_d + i)) * (2*multipole_ord + 1) * np.exp(aux)
                
                for j in range(multipole_ord - 2*i +1):
                    kwargs[_Args.j] = j
                    
                    _d_coeff = _coeff_r2_integral(multipole_ord,  i, j, mu2r1_const)
                    
                    # negative integral
                    kwargs[_Args.r1] *= -1
                    neg_int = gaussianIntegralQuadrature( _r2_dependentFunction, 
                                                         0, r_Max, quad_order,
                                                         **kwargs)
                    # positive integral (restore r1)
                    kwargs[_Args.r1] *= -1
                    pos_int = gaussianIntegralQuadrature( _r2_dependentFunction, 
                                                         0, r_Max, quad_order,
                                                         **kwargs)
                    
                    sum_ += aux * _d_coeff * ((pos_int * ((-1)**j)) 
                                              + (((-1)**phase) * neg_int))
    return sum_  


# radial integral (without normalization coefficient and constants)
def _r1_dependentFunction(r1, **kwargs):
    # getters
    mu_b_coef = kwargs.get(_Args.mu_b_coef)
    L_ac = kwargs.get(_Args.L_ac)
    
    # set
    kwargs[_Args.r1] = r1
    
    power_ = - (r1**2) * mu_b_coef
    
    # the quadrature roots never come over r=0 (x=-1)
    aux  = np.exp((L_ac * np.log(r1)) + power_) 
    aux *= _radialIntegral_over_r2(**kwargs)
    
    return aux


def _radialIntegral_over_r1(bra, ket, k, b_length, mu_param, r_Max, quad_ord):
    
    kwargs = {
        _Args.a_st : bra.sp_state_1,
        _Args.b_st : bra.sp_state_2,
        _Args.c_st : ket.sp_state_1,
        _Args.d_st : ket.sp_state_2,
        
        _Args.b_length : b_length,
        _Args.mu_param : mu_param,
        
        _Args.r_Max : r_Max,
        _Args.quad_ord : quad_ord,
        _Args.multipole_ord : k,
        _Args.mu_b_coef : (1/(b_length**2) + 1/(mu_param**2))
    }
    
    sum_ = 0.
    for k_a in range(bra.sp_state_1.n +1):
        kwargs[_Args.k_a] = k_a
        alpha_nlk_a = _sho_series_coeff(bra.sp_state_1, k_a)
        
        for k_c in range(ket.sp_state_1.n +1):
            kwargs[_Args.k_c] = k_c
            alpha_nlk_b = _sho_series_coeff(ket.sp_state_1, k_c)
            L_ac = (2*(k_a + k_c)) + (bra.sp_state_1.l + ket.sp_state_1.l) + 2
            kwargs[_Args.L_ac] = L_ac
            
            aux = np.exp(alpha_nlk_a + alpha_nlk_b - (L_ac * np.log(b_length)))
            aux *= (-1)**(k_a + k_c)
            
            sum_ += aux * gaussianIntegralQuadrature(_r1_dependentFunction, 
                                                     0, 
                                                     r_Max * b_length,
                                                     quad_ord,
                                                     **kwargs)
    return (b_length**-2) * sum_
    return sum_


# coupling coefficients (multipole_ expansion)
def _coupling_coeff(bra, ket, multipole_moment):
    
    a, b = bra.sp_state_1, bra.sp_state_2
    c, d = ket.sp_state_1, ket.sp_state_2
    L, L_q = bra.L, ket.L 
    
    if L != L_q:
        return 0.0
    if ((a.l + c.l)%2 != (b.l + b.l)%2) or (a.l - c.l + multipole_moment)%2 == 1:
        return 0.0
    
    coupling  = np.sqrt(np.prod([(2*wf.l + 1) for wf in (a, b, c, d)]))
    coupling *= (-1)**(L)
    coupling *= safe_3j_symbols(a.l, multipole_moment, c.l, 0, 0, 0)
    coupling *= safe_3j_symbols(b.l, multipole_moment, d.l, 0, 0, 0)
    #coupling *= (-1)**(a.l + b.l + c.l + d.l) 
    coupling *= safe_racah(c.l, d.l, a.l, b.l,  L, multipole_moment)    
    
    return np.sqrt(2*L + 1) * coupling
    

def _gaussian2BMatrixElement(bra, ket, b, mu_parameter, V_0=1, r_Max=10, quad_ord=10):
    assert isinstance(bra, QN_2body_L_Coupling), "Bra <QN_2body_L_Coupling> required"
    assert isinstance(ket, QN_2body_L_Coupling), "Ket <QN_2body_L_Coupling> required"
    
    print("EVALUATION: <", bra,"||exp(-(r_12/mu)^2)||", ket, ">")
    sum_  = 0.
    # multipole_ expansion
    for k in range(abs(bra.L - ket.L), bra.L + ket.L  +1):
        # coupling coefficients
        mom_coupl = _coupling_coeff(bra, ket, k)
        if abs(mom_coupl) < 1.0e-9:
            print("\t multipole_[{:2}]   COUPL=0.0".format(k))
            continue
        # radial integral ()
        radial_integral = _radialIntegral_over_r1(bra, ket, k, 
                                                  b, mu_parameter, 
                                                  r_Max, quad_ord)
        mom_matrix_element = mom_coupl * radial_integral
        sum_ += mom_matrix_element
        
        print("\t multipole_[{:2}]   COUPL={:7.6f}  RADIAL={:8.6f} = {:7.6f}"
              .format(k, mom_coupl, radial_integral, mom_matrix_element))
    
    return  V_0 * sum_


#===============================================================================
# l=0 GAUSSIAN MATRIX ELEMENTS QUADRATURE
#===============================================================================
def _ssss_r2_dependentFunction(r2, **kwargs):
    # getters
    mu_b_coef = np.sqrt(kwargs.get(_Args.mu_b_coef))
    mu2 = kwargs.get(_Args.mu_param)**2
    L_bd = kwargs.get(_Args.L_bd)
    r1 = kwargs.get(_Args.r1)
    
    
    power_ = (mu_b_coef * r2) + (r1 / (mu_b_coef * mu2))
    power_ = - (power_**2)
    
    # the quadrature roots never come over r=0 (x=-1)
    # L_bd' = L_bd - 1
    aux  = np.exp((L_bd * np.log(r2)) + power_)     
    return aux

def _ssss_radialIntegral_over_r2(**kwargs):
    
    nb = kwargs.get(_Args.b_st)
    nd = kwargs.get(_Args.d_st)
    b_length = kwargs.get(_Args.b_length)
    r_Max       = kwargs.get(_Args.r_Max) * b_length
    quad_order  = kwargs.get(_Args.quad_ord)
    
    sum_ = 0.0
    for k_b in range(nb +1):
        alpha_nlk_b = _sho_series_coeff(QN_1body_radial(nb, 0), k_b)
        
        for k_d in range(nd +1):
            alpha_nlk_d = _sho_series_coeff(QN_1body_radial(nd, 0), k_d)
            
            L_bd = 2*(k_b + k_d) + 1
            kwargs[_Args.L_bd] = L_bd
            
            aux = (alpha_nlk_b + alpha_nlk_d - (L_bd * np.log(b_length)))
            
            aux = ((-1)**(k_b + k_d)) * np.exp(aux)
            
            # negative integral
            kwargs[_Args.r1] *= -1
            neg_int = gaussianIntegralQuadrature(_ssss_r2_dependentFunction, 
                                                 0, r_Max, quad_order,
                                                 **kwargs)
            # positive integral (restore r1)
            kwargs[_Args.r1] *= -1
            pos_int = gaussianIntegralQuadrature( _ssss_r2_dependentFunction, 
                                                 0, r_Max, quad_order,
                                                 **kwargs)
            
            sum_ += aux * (neg_int - pos_int)
    return sum_  


# radial integral (without normalization coefficient and constants)
def _ssss_r1_dependentFunction(r1, **kwargs):
    # getters
    mu_b_coef = kwargs.get(_Args.mu_b_coef)
    b_param2  = kwargs.get(_Args.b_length)**2
    mu2     = kwargs.get(_Args.mu_param)**2
    L_ac    = kwargs.get(_Args.L_ac)
    
    # set
    kwargs[_Args.r1] = r1
    
#     power_ = - (r1**2) * (mu_b_coef - ((1/mu2) * (1 + (1/(mu_b_coef * mu2)))))
    power_  = mu_b_coef - (1/(mu_b_coef * (mu2**2)))
    power_ *= - (r1**2)
    
    # the quadrature roots never come over r=0 (x=-1)
    # L_ac = L_ac - 1
    aux  = np.exp((L_ac * np.log(r1)) + power_) 
    aux *= _ssss_radialIntegral_over_r2(**kwargs)
    
    return aux



def _ssss_gaussian2BMatrixElement(na, nb, nc, nd, b, mu_param, V_0=1, r_Max=10, quad_ord=10):
    
    kwargs = {
        _Args.b_st: nb,
        _Args.d_st: nd, 
        
        _Args.b_length : b_length,
        _Args.mu_param : mu_param,
        
        _Args.r_Max : r_Max,
        _Args.quad_ord : quad_ord,
        _Args.mu_b_coef : (1/(b_length**2)) + (1/(mu_param**2))
    }
    
    sum_ = 0.
    for k_a in range(na +1):
        alpha_nlk_a = _sho_series_coeff(QN_1body_radial(na, 0), k_a)
        
        for k_c in range(nc +1):
            alpha_nlk_b = _sho_series_coeff(QN_1body_radial(nc, 0), k_c)
            L_ac = (2*(k_a + k_c)) + 1
            kwargs[_Args.L_ac] = L_ac
            
            aux = np.exp(alpha_nlk_a + alpha_nlk_b - (L_ac * np.log(b_length)))
            aux *= (-1)**(k_a + k_c)
            
            sum_ += aux * gaussianIntegralQuadrature(_ssss_r1_dependentFunction, 
                                                     0, 
                                                     r_Max * b_length,
                                                     quad_ord,
                                                     **kwargs)
    return 0.25 * V_0 * ((mu_param/b_length)**2) * sum_

#===============================================================================
# MAIN
#===============================================================================

if __name__ == '__main__':
    
#     _testNormalizationRadialFunctionsRange(40, b_length=1.5)
#     print(_testNormalizationSingleRadialFunction(20, 0, b_length=1.5))
    
    b_length = 1.0
    mu_param = 1.0
    V_0      = 1
    
    a, b = QN_1body_radial(1, 3), QN_1body_radial(1, 1)
    c, d = QN_1body_radial(1, 3), QN_1body_radial(1, 1)
    
    bra = QN_2body_L_Coupling(a, b, 2)  
    ket = QN_2body_L_Coupling(c, d, 2)
    
    quad_ord = 120
    r_max = 6
#     for quad_ord in range(5, 41, 5):
# #     for r_max in range(1, 12 , 5):
#         print("quad_ord, rmax: ", quad_ord, r_max, ": ",
#               _gaussian2BMatrixElement(bra, ket, b_length, mu_param, V_0, 
#                                        quad_ord=quad_ord))#, r_Max=r_max))
    n  = 0
    na = n
    nb = n
    nc = n
    nd = n
    
    la = 2
    lb = 2
    lc = 2
    ld = 2
    L  = 4
    
#     for quad_ord in (i for i in range(50, 101, 10)):
#         a, b = QN_1body_radial(n, la), QN_1body_radial(n, lb)
#         c, d = QN_1body_radial(n, lc), QN_1body_radial(n, ld)
#         #_plot_functions(a, b, c, d, b_length, mu_param, V_0, r_max)
#         print("quad_ord, rmax: ", quad_ord, r_max)
#         print("me = ", _gaussian2BMatrixElement(QN_2body_L_Coupling(a, b, L), 
#                                                 QN_2body_L_Coupling(c, d, L), 
#                                                 b_length, mu_param, V_0, 
#                                                 quad_ord=quad_ord, r_Max=r_max))
    for mu_param in (1.0, 0.7):
        a, b = QN_1body_radial(na, la), QN_1body_radial(nb, lb)
        c, d = QN_1body_radial(nc, lc), QN_1body_radial(nd, ld)
        _plot_functions(a, b, c, d, b_length, mu_param, V_0, r_max)
        print("quad_ord, rmax:", quad_ord, r_max, '  (b, m)=', b_length, mu_param)
        print("me =", _gaussian2BMatrixElement(QN_2body_L_Coupling(a, b, L), 
                                                QN_2body_L_Coupling(c, d, L), 
                                                b_length, mu_param, V_0, 
                                                quad_ord=quad_ord, r_Max=r_max))
#         print("me (ssss) = ", _ssss_gaussian2BMatrixElement(na, nb, nc, nd, 
#                                                             b, mu_param, V_0, 
#                                                             r_max, int(quad_ord//.7)))
        
        
    
    
#     a, b = QN_1body_radial(0, la), QN_1body_radial(0, lb)
#     c, d = QN_1body_radial(0, lc), QN_1body_radial(0, ld)
#     _plot_functions(a, b, c, d, b_length, mu_param, V_0, r_max)
#     print("quad_ord, rmax: ", quad_ord, r_max)
#     print("me = ", _gaussian2BMatrixElement(QN_2body_L_Coupling(a, b, L), 
#                                             QN_2body_L_Coupling(c, d, L), 
#                                             b_length, mu_param, V_0, 
#                                             quad_ord=quad_ord, r_Max=r_max))
#     
#     a, b = QN_1body_radial(1, la), QN_1body_radial(1, lb)
#     c, d = QN_1body_radial(1, lc), QN_1body_radial(1, ld)
#     _plot_functions(a, b, c, d, b_length, mu_param, V_0, r_max)
#     print("quad_ord, rmax: ", quad_ord, r_max)
#     print("me = ", _gaussian2BMatrixElement(QN_2body_L_Coupling(a, b, L), 
#                                             QN_2body_L_Coupling(c, d, L), 
#                                             b_length, mu_param, V_0, 
#                                             quad_ord=quad_ord, r_Max=r_max))
#     
#     a, b = QN_1body_radial(2, la), QN_1body_radial(2, lb)
#     c, d = QN_1body_radial(2, lc), QN_1body_radial(2, ld)
#     print("quad_ord, rmax: ", quad_ord, r_max)
#     print("me = ", _gaussian2BMatrixElement(QN_2body_L_Coupling(a, b, L), 
#                                             QN_2body_L_Coupling(c, d, L),
#                                             b_length, mu_param, V_0, 
#                                             quad_ord=quad_ord, r_Max=r_max))
#     
    _plot_functions(a, b, c, d, b_length, mu_param, V_0, r_max, show=True)
    