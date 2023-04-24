'''
Created on Apr 4, 2023

@author: Miguel
'''
from helpers import SCIPY_INSTALLED
if SCIPY_INSTALLED:
    from scipy.special import genlaguerre
    from scipy.special import lpmn

from sympy import S
import numpy as np

from helpers.Helpers import readAntoine, safe_clebsch_gordan, fact,\
    double_factorial

_angular_Y_KM_me_memo = {}
_radial_2Body_functions = {}

def angular_Y_KM_index(K, M, is_half_integer):
    """
    METHOD FROM FORTRAN DENS_TAURUS: index runs from 1
    
    Index accessor for angular momentum states, tuple (K, M) can be stored in
    order from 1 to the length of the K maximum of the array.
     
    * For Integer : K = 0,1,2, ...  ! M=(0), (-1,0,+1), ...
          index =       (K)**2        +      K + M          + 1
                  (previous elements)   (range 0 to 2K + 1)   (starting index)
     
    * For Half-Integers : K = 1,3,5, ...;   M=(-1,+1), (-3,-1,+1,+3), ...
          I(K)  = (K+1)/2
          index = [I(K-1)**2 + I(K-1) = I*(I-1)] +   (K + M)/2           + 1
                       (previous elements)        (range 0 to 2K + 1) (start)
     
    * This function can be used to define the length of the K,M array, by giving
      the last valid element: angular_momentum_index(K_max, K_max, half_integer)
    """
    if (is_half_integer):
        return ((K*K - 1)//4) + ((K + M)//2) + 1
    else:
        return K*(K + 1) + M + 1



def _angular_Y_KM_memo_accessor(indx_a, indx_b, indx_K):
    """ 
        indexing_ = (ja_ma[HafInt], jb_mb[HalfInt], K_M[Int])
    Remember the index do not contain information from the orbital part!
    """
    # list_ = (indx_a, indx_b, indx_K)
        
    return '{},{},{}'.format(indx_a, indx_b, indx_K)

def _angularYCoeff_evaluation(la, ja, mja, lb, jb, mjb, K):
    if (la + lb + K) % 2 == 1:
        return 0.0, 1.0
    
    args  = (K, S(ja)/2, S(jb)/2, (mjb-mja)//2, S(mja)/2, S(mjb)/2)
    val   = safe_clebsch_gordan(*args)
    val  *= safe_clebsch_gordan(S(ja)/2, S(jb)/2, K,  S(1)/2, -S(1)/2,  0)
    
    phase = (-1)**(((ja-1)//2) - jb + 1)
    val *= phase * 0.28209479177387814 * ((ja+1)**.5)
    #                                           1 / sqrt(4*pi)
    return val, phase


def assolegendre(l, m, x):
    """
    function assolegendre
                                                                            
    Calculates the value of the associated Legendre polynomial P^m_l(x) using the
    algorithm given in the Numerical Recipes in Fortran 90, which is based on the
    recurrence relations:
    P^m_m(x) = (-1)**m (2m-1)!! sqrt(1-x**2)  => build P^m_m for any m
    P^m_m+1(x) = x (2m+1) P^m_m(x)            => build P^m_m+1 from previous one
    (l-m+1) P^m_l+1 = x (2l+1) P^m_l(x) - (l+m) P^m_l-1(x)
                                              => build all other possible l
                                                                            
    In addition, I have added the symmetry for negative m:
    P^-m_l = (-1)**m (l-m)!/(l+m)!
    """
    
    # Checks the validity of the arguments
    if ( (l < 0) or (abs(m) > l) or (abs(x) > 1) ) :
        print ("(a)", "Wrong argument(s) in function assolegendre")
    
    # Transforms m in -m when m is negative
    phase = 1.0
    
    if ( m < 0 ) :
        m = abs(m)
        phase = ((-1)**m) * np.exp((fact(l-m)) - (fact(l+m)))
    
    # Compute P^m_m
    pmm = 1.0
    if ( m > 0 ) :
        somx2 = ( (1.0-x) * (1.0+x) )**0.5
        pmm = np.exp(double_factorial(2*m-1)) * (somx2**m)
        if ( m % 2 == 1 ): pmm = -pmm
    
    if ( l == m ) :
        leg = pmm
    else:
        pmmp1 = x * (2*m +1) * pmm
        
        # Compute P^m_m+1
        if ( l == m+1 ) :
            leg = pmmp1
        # Compute P^m_l for l > m+1
        else:
            for ll in range(m+2, l +1):
            # do l1 = m+2, l
                pll = (x*(2*ll-1)*pmmp1 - (ll+m-1)*pmm) / (ll-m)
                pmm = pmmp1
                pmmp1 = pll
            
            leg = pll
    
    leg = phase * leg
    return leg


def sphericalHarmonic(K,M, angle=(0,0), theta_as_costh=True):
    """
    Version for of the scipy:
        ArgS: Y_KM(theta, phi) = Y_KM(angle)
    :angle = <floats> (theta or cos(theta), phi)
    :theta_as_costh <bool>=True : form of the theta (altitude) angle
    
    """
    C_lm  = np.sqrt((2*K + 1) / 12.566370614359172) 
    C_lm *= np.exp( (0.5*(fact(K-M) - fact(K+M)))  + (1j* M * angle[1]) )

    if not theta_as_costh:
        angle[0] = np.cos(angle[0])
    
    leg_poly = assolegendre(K, M, angle[0])
    return C_lm * leg_poly 

def _buildAngularYCoeffsArray(sh_states_list):
    """
    Introduces coupling angular coefficients for the introduced list of shell 
    states (with orbital L > 10 format).
        Returns the maximum multipole order K that can be coupled for the states
        
    * Note: if it is called with two different valence spaces, it will be append
        the parts from the new list, but cannot couple with previous elements in
        the _angular_Y_KM_me_memo dict.
    * Note: 
    """
    global _angular_Y_KM_me_memo
    
    f = open('test_.txt', 'w+')
    
    already_done = set( list(_angular_Y_KM_me_memo.keys()) )
    K_max = 0
    
    for i in range(len(sh_states_list)):
        sh_st_a = sh_states_list[i]
        _,la,ja = readAntoine(sh_st_a, l_ge_10=True)
        
        for j in range(i, len(sh_states_list)):
            sh_st_b = sh_states_list[j]
            _,lb,jb = readAntoine(sh_st_b, l_ge_10=True)
            
            if (la, ja, lb, jb) in already_done: 
                continue # j and l tuples from different n index are the same, skip
            else:
                already_done.add( (la, ja, lb, jb) )
            
            for K in range(max(0, abs(ja-jb)//2), (ja+jb)//2 +1):
                if (K + la + lb) % 2 == 1: continue
                K_max = max(K_max, K)
                for mja in range(-ja, ja+1, 2):
                    for mjb in range(-jb, jb+1, 2):
                        M = (mjb - mja) // 2
                        if abs(M) > K: continue
                        
                        indx_a = angular_Y_KM_index(ja, mja, True)
                        indx_b = angular_Y_KM_index(jb, mjb, True)
                        indx_K = angular_Y_KM_index( K,   M, False)
                        
                        key_1 = _angular_Y_KM_memo_accessor(indx_a, indx_b, indx_K)
                        if key_1 in _angular_Y_KM_me_memo: continue
                        
                        if key_1 == '2,12,9' or key_1 == '12,2,9':
                            _=0
                        a_km_val, phs = _angularYCoeff_evaluation(la, ja, mja, 
                                                                  lb, jb, mjb, K) 
                        _angular_Y_KM_me_memo[key_1] = a_km_val
                        
                        if abs(a_km_val) > 1.0e-10:
                            str_ = f"{la:3}{ja:3}{mja:3}{indx_a:3} | {lb:3}{jb:3}{mjb:3}{indx_b:3} | {K:3}{M:3}{indx_K:3} = {a_km_val:20.15f}{phs:9.3f}\n"
                            f.write(str_)
                        
                        if indx_a == indx_b: continue
                        
                        indx_K = angular_Y_KM_index( K,  -M, False)
                        key_2 = _angular_Y_KM_memo_accessor(indx_b, indx_a, indx_K)
                        _angular_Y_KM_me_memo[key_2] = ((-1)**M) * a_km_val
                        if abs(a_km_val) > 1.0e-10:
                            str_ = f"{lb:3}{jb:3}{mjb:3}{indx_b:3} | {la:3}{ja:3}{mja:3}{indx_a:3} | {K:3}{M:3}{indx_K:3} = {((-1)**M)*a_km_val:20.15f}{phs:9.3f}\n"
                            f.write(str_)
                        

            _ = 0
    f.close()
    _= 0
    return K_max
    
    ## REMOVE ONCE IMPLEMENTED MODULE
    # #===========================================================================
    # # Test for the Y_KM matrix elements
    # #===========================================================================
    # not_found_in_bench = {}
    # not_found_in_test  = {}
    # failed_   = {}
    # passed_   = {}
    # with open('Y_ab_K_matrix_elements.gut', 'r') as f:
    #     dat0 = f.readlines()[2:]
    #
    #     for il, line in enumerate(dat0):
    #         a, b, k_val = line.split(' | ')
    #         i_a = int(a.split()[-1])
    #         i_b = int(b.split()[-1])
    #         k, val = k_val.split(' = ')
    #         i_k = int(k.split()[-1])
    #         M = int(k.split()[-2])
    #         val_bench = float(val.split()[0])
    #         key_bench  = _angular_Y_KM_memo_accessor(i_a,  i_b, i_k)
    #         key_bench2 = _angular_Y_KM_memo_accessor(i_b, i_a, i_k)
    #
    #         if key_bench == '3,8,7' or key_bench == '8,3,7':
    #             _=0
    #
    #         val_test  = _angular_Y_KM_me_memo.get(key_bench,  None)
    #         val_test2 = _angular_Y_KM_me_memo.get(key_bench2, None)
    #
    #         if val_test == None and val_test2 == None:
    #             not_found_in_test[key_bench] = val_bench
    #         else:
    #             if not val_test:
    #                 val_test = val_test2 * ((-1)**M)
    #             if abs(val_bench - val_test) < 1.0e-8:
    #                 passed_[key_bench] = val_bench
    #             else:
    #                 failed_[key_bench] = (val_bench, val_test)
    #
    #     for key_test, val_test in _angular_Y_KM_me_memo.items():
    #         key2 = key_test.split(',')
    #         key2 = ','.join((key2[1], key2[0], key2[2]))
    #         if abs(val_test) < 1.0e-8:
    #             continue
    #         if key_test in failed_ or key2 in failed_:
    #             continue
    #         if key_test in passed_ or key2 in passed_:
    #             continue
    #         if key_test in not_found_in_test or key2 in not_found_in_test:
    #             continue
    #         not_found_in_bench[key_test] = val_test
    #
    # print(" TEST Y_KM coefficients. -----------------------------------")
    # print(f" PASSED : {len(passed_):}")
    # print(f" FAILED : {len(failed_):}")
    # print(f" BENCH NOT in TEST : {len(not_found_in_test):}")
    # print(f" TEST NOT in BENCH : {len(not_found_in_bench):}")
    # print(" ----------------------------------- TEST Y_KM coefficients. ")   
    #
    #
    # _= 0


def _radial2Body_bench_Generator(na,la, nb,lb, HO_b_length=1.0):
    """
    returns the lambda function to pass for the radial grid.
        radial grid must not include the r/b. This is done here
    
    Constant:
        Anla = np.sqrt( (2**(na+la+2) * fact(na)) / 
                      (np.sqrt(np.pi) * double_factorial(2*na+2*la+1)) )
        Anlb = np.sqrt( (2**(nb+lb+2) * fact(nb)) / 
                      (np.sqrt(np.pi) * double_factorial(2*nb+2*lb+1)) )
    """
    
    C_ab  = ((na+la+nb+lb+4)*np.log(2)) + fact(na) + fact(nb)
    C_ab -= np.log(np.pi)
    C_ab -= double_factorial(2*na+2*la+1) + double_factorial(2*nb+2*lb+1)
    C_ab  = np.exp(0.5 * C_ab)
    
    alpha_a = la + 0.5
    alpha_b = lb + 0.5
    
    wf_prod_on_r = lambda r: C_ab * ((r/HO_b_length)**(la+lb)) \
                  * genlaguerre(na,alpha_a)((r/HO_b_length)**2) \
                  * genlaguerre(nb,alpha_b)((r/HO_b_length)**2) \
                  * (HO_b_length**-3.0) / np.exp((r/HO_b_length)**2.0)
    return wf_prod_on_r


def _buildRadialFunctionsArray(sh_states_list, HO_b_length=1.0):
    """
    Construct and storage the radial functions (as functionals) in terms of their
    n,l index tuple. 
        IMPORTANT: Calling this function will erase the functions stored before.
    
        * Same notes as in the angular coefficients:
    It can be called several times, but it won't be coupling between the states
    of the different calls. In this case, also the b length could be different 
    between the states.
    
    I.e: This function can be called for the spatial_density and the matrix 
    elements over that density with other b_lenght (or other w.functions).
      >> This can be solved by saving (deep-copying the array)
    """
    global _radial_2Body_functions
    _radial_2Body_functions = {} ## Reset of the Radial WF
    
    already_done = set()
    for i in range(len(sh_states_list)):
        sh_st_a = sh_states_list[i]
        na,la,_ = readAntoine(sh_st_a, l_ge_10=True)
        
        for j in range(i, len(sh_states_list)):
            sh_st_b = sh_states_list[j]
            nb,lb,_ = readAntoine(sh_st_b, l_ge_10=True)
            
            if (na,la, nb,lb) in already_done: 
                continue # j and l tuples from different n index are the same, skip
            else:
                already_done.add( (na,la, nb,lb) )
            
            lambda_funct = _radial2Body_bench_Generator(na,la, nb,lb, HO_b_length)
            _radial_2Body_functions[(na,la, nb,lb)] = lambda_funct
            
            if (na,la) == (nb,lb): continue # symmetrical a/b
            _radial_2Body_functions[(nb,lb, na,la)] = lambda_funct
            
    return _radial_2Body_functions
    
    