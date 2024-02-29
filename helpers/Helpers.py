'''
Created on Feb 25, 2021

@author: Miguel
'''

import os
from copy import deepcopy
#===============================================================================
#%% Constants
#===============================================================================

class Constants:
    HBAR      = 6.582119e-22    # MeV s
    HBAR_C    = 197.327053# << value in Taurus ## 197.326963      # MeV fm 
    
    M_PROTON  = 938.27208816      # MeV/c2
    M_NEUTRON = 939.56542052      # MeV/c2
    M_NUCLEON = 931.494028      # MeV/c2
    M_ELECTRON= 0.51099891      # MeV/c2
    M_MEAN    = 938.91875434      # (m_Proton + m_Neutron) / 2
    
    ALPHA     = 7.297353e-3
    e_CHARGE  = 1.602176e-19    # C


_LINE_1 = "\n====================================================================\n"
_LINE_2 = "\n--------------------------------------------------------------------\n"

#===============================================================================
# %% Factorials, Double Factorials and GammaFunction Value Storing
#===============================================================================

import numpy as np

_fact = []
_double_fact = []
_gamma_half_int = []
_FACTORIAL_DIM = -1

def fact_build(max_order_fact):
    # construct a global factorial base
    global _fact
    global _FACTORIAL_DIM
    
    if _FACTORIAL_DIM > -1:
        print("::::   Calling Factorial build: from [{}] to [{}]   ::::"
              .format(_FACTORIAL_DIM, _FACTORIAL_DIM + max_order_fact + 1))
    
    if not _fact:
        _fact = [np.log(1)]
        min_order = 1
    else:
        min_order = len(_fact)
    
    for i in range(min_order, max_order_fact + 1):
        _fact.append(_fact[i-1] + np.log(i))
    
    _FACTORIAL_DIM = min_order + max_order_fact
    # !! logarithm values for the factorial !!

def double_fact_build(max_order_fact):
    # construct a global double factorial base
    global _double_fact
    
    if not _double_fact:
        _double_fact = [np.log(1), np.log(1)]
        min_order = 2
    else:
        min_order = len(_double_fact)
    
    for i in range(min_order, max_order_fact + 1):
        _double_fact.append(_double_fact[i-2] + np.log(i))
        
    # !! logarithm values for the factorial !!

fact_build(200)
double_fact_build(200)

    
def gamma_half_int_build(max_order_fact):
    """ Gamma n over 2: [G(1/2), G(2/2), G(3/2), ...]"""
    global _gamma_half_int
    
    if not _gamma_half_int:
        _gamma_half_int = [np.log(np.sqrt(np.pi)), 0, 
                           np.log(0.5*np.sqrt(np.pi)), 0]
        min_order = 4
    else:
        min_order = len(_gamma_half_int)
    
    for i in range(min_order, max_order_fact + 1):
        if i % 2 == 0:
            _gamma_half_int.append(_gamma_half_int[i-2] + np.log((i - 1)/2))
        else:
            _gamma_half_int.append(_gamma_half_int[i-2] + np.log((i - 1)//2))
    
    #_FACTORIAL_DIM = max_order_fact
    # !! logarithm values for the factorial !!

gamma_half_int_build(100)


def fact(i):
    """
    Integer Factorial function, access to stored factorials
    :int i
    :return the factorial value (logarithmic value)
    """
#     if i > _FACTORIAL_DIM:
#         fact_build(i + 100)
    return _fact[i]

def double_factorial(i):
    """ 
    (n)!! = 1*3*5* ... *n (if n is odd),  = 2*4*6* ... *n if n even
    """
    return _double_fact[i]
    
def gamma_half_int(i):
    """
    Gamma_Function ( n + 1/2 ) = (2n - 1)!! sqrt(pi) / 2^n
    : i     <integer>  the value over 2: i=n -> (n+1)/2
    """
    if (i <= 0):
        raise ValueError("Gamma Function G(n/2) need n >= 1, got: {}".format(i))
    return _gamma_half_int[i - 1]

#===============================================================================
#%%     sympy angular momentum fucntions
#===============================================================================
from sympy.physics.wigner import wigner_9j, racah, wigner_6j, clebsch_gordan, wigner_3j

def safe_wigner_9j(a,b,c, d,e,f, g,h,i):
    """ Wigner 9j symbol, same arguments as in Avoid the ValueError whenever the
     arguments don't fulfill the triangle relation. 
     { (a,b,c)
       (d,e,f)
       (g,h,i) } 
    """
    try: 
        args = (a,b,c, d,e,f, g,h,i)
        return float(wigner_9j(*args, prec=None))
    except ValueError or AttributeError:
        return 0

def safe_racah(a, b, c, d, ee, ff):
    """ Avoid the ValueError whenever the arguments don't fulfill the triangle
    relation. """
    try: 
        return float(racah(a, b, c, d, ee, ff, prec=None))
    except ValueError or AttributeError:
        return 0

def safe_wigner_6j(a,b,c, d,e,f):
    """ Wigner 6j symbol, same arguments as in Avoid the ValueError whenever the
     arguments don't fulfill the triangle relation. 
     { (a,b,c)
       (d,e,f) } 
    """
    try: 
        args = (a,b,c, d,e,f)
        return float(wigner_6j(*args, prec=None))
    except ValueError or AttributeError:
        return 0

def safe_clebsch_gordan(j1, j2, j3, m1, m2, m3):
    """
    :args   j1, j2, j3,   m1, m2, m3
    
    Calculates the Clebsch-Gordan coefficient
    < j1 m1, j2 m2 | j3 m3 >.
    
    Return float value for Zero Clebsh-Gordan coefficient, avoid Zero object
    """
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))

def safe_3j_symbols(j1, j2, j3, m1, m2, m3):
    """
    :args     j_1, j_2, j_3, m_1, m_2, m_3
    
    Calculates the Clebsch-Gordan coefficient for the base 
    < j1 m1, j2 m2 | j3 m3 >.
    """
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))

def _triangularRelation(a, b, c):
    """
    return True if a,b c are in triangular relation
    Args: multiplied by 2
    """
    if (a+b+c) % 2 != 0:
        return False
    return abs(a-b) < c or a+b > c

def expliclit_9j1o2S01(d,e,f, a,b,c, S):
    """
    all arguments must be given multiplied by 2 (also S)
    {(a+lam, b+mu, c+nu)
     (a    , b   , c   )
     (1/2  , 1/2 , S   )}
     2S = 0, 2
     lam,mu,nu = +- 1/2
     
     formulas from Varsalovich
    TODO: TEST with function from wigner package
    """
    raise Exception("This method is broken, fix it!")
    
    lam, mu, nu = d - a, e - b, f - c
    ## TODO: Triangular test
    if S not in (0, 2):
        raise Exception(f"This function is only valid for S=0,1 (S must be given as 2S), got [{S}]")
    if abs(lam) != 1 or abs(mu) != 1 or abs(nu) > 2:
        return 0.0
    if not (_triangularRelation(a, b, c) or _triangularRelation(d, a, 1) or
            _triangularRelation(d, e, f) or _triangularRelation(e, b, 1)):
        return 0.0
    # if abs(lam) == 1 or abs(mu) == 1:
    #     return 0.0
    key_ = (lam, mu)
    s    = (a + b + c) / 2
    den_ = (a+1)*(b+1)*(c+1)
    val  = 0.0
    if key_ != (1,1) and ((a==0) or (b==0)):
        return 0.0
    
    if S==0:
        if nu == 0:
            den_ *= 2
            if   key_ == ( 1, 1):
                val =  (((s+2)*(s+c+1)) / (den_*(a+2)*(b+2)) )**.5
            elif key_ == ( 1,-1):
                val =  (((s-b+1)*(s-a)) / (den_*(a+2)*(b)) )**.5
            elif key_ == (-1, 1):
                val = -(((s-b)*(s-a+1)) / (den_*(a)*(b+2)) )**.5
            elif key_ == (-1,-1):
                val =  (((s+1)*(s-c))   / (den_*(a)*(b)) )**.5
    elif S==2:
        if   nu == 2:
            den_ *= 3*(c+2)*(c+3) 
            if   key_ == ( 1, 1):
                val =  (((s+2)*(s+3)*(s-b+1)*(s-a+1)) / (den_*(a+2)*(b+2)) )**.5
            elif key_ == ( 1,-1):
                val = -(((s+2)*(s-c)*(s-b+1)*(s-b+2)) / (den_*(a+2)*(b)) )**.5
            elif key_ == (-1, 1):
                val =  (((s+2)*(s-c)*(s-a+1)*(s-a+2)) / (den_*(a)*(b+2)) )**.5
            elif key_ == (-1,-1):
                val = -(((s-c-1)*(s-c)*(s-b+1)*(s-a+1)) / (den_*(a)*(b)) )**.5
        elif nu == 0:
            den_ *= 3*(c+1)*(c)
            if   key_ == ( 1, 1):
                val = (((s+2)*(s-c+1)) / (den_*(a+2)*(b+2)) )**.5
                val *= (a-b)/2
            elif key_ == ( 1,-1):
                val = (((s-b+1)*(s-a)) / (den_*(a+2)*(b)) )**.5
                val *= 0.5 + ((a+b)/2)
            elif key_ == (-1, 1):
                val = (((s-b)*(s-a+1)) / (den_*(a)*(b+2)) )**.5
                val *= 0.5 + ((a+b)/2)
            elif key_ == (-1,-1):
                val = (((s+1)*(s-c))   / (den_*(a)*(b)) )**.5
                val *= (b-a)/2
        elif nu == -2:
            den_ *= 3*(c-1)*c
            if   key_ == ( 1, 1):
                val = -(((s-c+1)*(s-c+2)*(s-b)*(s-a)) / (den_*(a+2)*(b+2)) )**.5
            elif key_ == ( 1,-1):
                val = -(((s+1)*(s-c+1)*(s-a-1)*(s-a)) / (den_*(a+2)*(b)) )**.5
            elif key_ == (-1, 1):
                val = (((s+1)*(s-c+1)*(s-b-1)*(s-b)) / (den_*(a)*(b+2)) )**.5
            elif key_ == (-1,-1):
                val = (((s)*(s+1)*(s-b)*(s-a))   / (den_*(a)*(b)) )**.5
    return val



# args = (1,3,2, 2,2,2, 0)
# args = (4,1,3, 5,2,3, 0)
# test_ = expliclit_9j1o2S01(*args)
#
# S = 1
# fail_, tot_ = 0, 0
# for a in range(6):
#     for b in range(6):
#         for c in range(6):
#
#             for d in range(a-1, a+2, 1):
#                 for e in range(b-1, b+2, 1):
#                     for f in range(c-2, c+3, 2):
#
#                         args = (d/2,e/2,f/2, a/2,b/2,c/2, 1/2,1/2, S)
#                         bench_ = safe_wigner_9j(*args)
#                         test_ = expliclit_9j1o2S01(d,e,f, a,b,c, 2*S)
#
#                         if abs(test_) > 1.0e-5:
#                             if d==0 or e==0:
#                                 print(f"D/E= 0 fail:: ({d},{e},{f}) ({a},{b},{c}) S={S} Error: BNH_{bench_:6f} != {test_:6f}") 
#                             # print(f"({a},{b},{c}) ({d},{e},{f}) S={S} = {test_:6f}")
#                             pass
#                         if abs(bench_-test_) > 1.0e-5:
#                             print(f"({d},{e},{f}) ({a},{b},{c}) S={S} Error: BNH_{bench_:6f} != {test_:6f}")
#                             fail_ += 1
#                         tot_ += 1
# print(f"TEST done! Fail[{fail_}/{tot_}]")
#===============================================================================
# 
#===============================================================================

def angular_condition(l1, l2, L):
    """ Triangular condition on l1+l2 = L """
    if L <= l1 + l2:
        if abs(l1 - l2) <= L:
            return True
    return False


#===============================================================================
# Global dictionary for B coefficient memorization pattern
# Index : comma separated indexes string: 'n,l,n_q,l_q,p'
#
## need to be here to avoid circular import from integrals and transformations.

def _B_coeff_memo_accessor(n, l, n_q, l_q, p):
    """ 
    Key constructor for the memory storage of the coefficients.
        B(nl,n'l', p) coefficients are symmetric by nl <-> n'l' return the 
    lowest tuple (read from the left):
    
    _B_coeff_memo_accessor(1, 2, 0, 3, 1) >>> (0, 3, 1, 2, 1)
    _B_coeff_memo_accessor(0, 3, 1, 2, 1) >>> (0, 3, 1, 2, 1)
    
    Note: works for both Moshinsky_ and Talman_ coefficients
    
    Technical Note: The matrix element B(n,l,n',l', p) = B(n',l',n,l, p)
    """
    
    list_ = [*min((n, l), (n_q, l_q)), *max((n, l), (n_q, l_q)), p]
    return ','.join(map(lambda x: str(x), list_))

#===============================================================================

#===============================================================================
#     
#===============================================================================
# Print iterations progress
def printProgressBar (iteration, total, prefix='Progress:'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
    
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    length      - Optional  : character length of bar (Int)
    """
    # decimals = 1
    length = 50
    fill = '*'
    
    # percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    percent = "{:5.2f}".format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '_' * (length - filledLength)
    if os.getcwd().startswith('C:'):
        print("\r{0} |{1}| {2}% complete\r".format(prefix, bar, percent), end='', flush=True)
    else:
        print("\r{0} |{1}| {2}% complete\r".format(prefix, bar, percent), end='')#, end = '\r')
    
    # Print New Line on Complete
    if iteration == total: 
        print()

def prettyPrintDictionary(dictionary, level=0, delimiter=' . '):
    
    header = ''.join([delimiter]*level)
    for k, val in dictionary.items():
        if isinstance(val, dict):
            print(header+str(k)+': {')
            prettyPrintDictionary(val, level + 1, delimiter)
            print(header+'}')
        else:
            print(header+str(k)+':'+str(val))


#------------------------------------------------------------------------------ 

def __copyHamiltonian_4keyTo2keys(results_0, reverse=False):
    """ Convert hamiltonian from 4-key <*bra, *ket> to 2-key (bra, ket) form"""
    results = {}
    if not reverse:
        for key_, vals in results_0.items():
            bra_, ket_ = key_[:2], key_[2:]
            if not bra_ in results:
                results[bra_] = {ket_: vals, }
            else:
                results[bra_][ket_] = vals
    else:
        for bra_, kvals in results_0.items():
            for ket_, vals in kvals.items():
                results[(*bra_, *ket_)] = vals
    return results

def sortingHamiltonian(results, sorted_2b_comb, is_jt=False, l_ge_10=True):
    """
    This function returns a hamiltonian in the order given, applying the phase
    changes from the J or JT scheme.
    
    Note: use it before [recursiveSumOnDictionaries]
    Input:
        :results <dict>        {(1, 1): {(101, 1): [{J,T: [values]}] ...}}
        :sorted_2b_comb<list>  [(1,1), (1,101), (1,10001), ... ]
    Returns:
        the same result dictionary with the keys in the order of sorted_2b_comb
    """
    dict_0 = {}
    ## 0. Structure as the imported hamiltonian (keys involve (a,b,c,d):{J})
    if list(results.keys()) == []: return dict_0
    
    _transform_to_key2 = False
    if list(results.keys())[0].__len__() == 2:
        dict_0 = __copyHamiltonian_4keyTo2keys(results, reverse=True)
        _transform_to_key2 = True
    else:
        dict_0 = results
    
    # 1 read all the 4-keys and sort in the order bra-ket (assign permutat and particle order)
    dict_1 = {}
    for bk, vals in dict_0.items():
        permuts = getAllPermutationsOf2Bstates(bk, l_ge_10,  not is_jt)
        
        for i, item_ in enumerate(permuts):
            bk_perm, t_perm, phs = item_
            if phs == 0: continue ## redundant permutation
            
            not_in_ = not bk_perm in dict_1
            
            if is_jt:
                dict_1[bk_perm] = {0: {}, 1:{}}
                for T in (0,1):
                    for J in vals:
                        phs2 = phs * (-1)**(J + T)
                        if not_in_:
                            dict_1[bk_perm][T][J] = phs2 * vals[T][J]
                        else:
                            # Test if matches
                            assert abs(dict_1[bk_perm][T][J] 
                                       - phs2 * vals[T][J]) < 1.0e-6,\
                                       "[ERROR]: values of:{} does not match "\
                                       "\with previous:{}".format(bk_perm, bk)
            else:
                dict_1[bk_perm] = dict([(J, dict()) for J in vals])
                for J in vals:
                    phs2 = phs
                    if i not in (0,4, 3,7): ## double/non exchange has no J dep.
                        phs2 = phs * (-1)**(J + 1)
                    for T in range(6):
                        if not_in_:
                            dict_1[bk_perm][J][T] = phs2 * vals[J][t_perm[T]]
                        else:
                            # Test if matches
                            # Test if matches
                            assert abs(dict_1[bk_perm][J][T] 
                                   - phs2 * vals[J][t_perm[T]]) < 1.0e-6,\
                                       "[ERROR]: values of:{} does not match "\
                                       "\with previous:{}".format(bk_perm, bk)
    
    # 2 sort in the order of bra (sorting_order) and apply the phs-changes
    dict_2 = {}
    for i in range(len(sorted_2b_comb)):
        bra = sorted_2b_comb[i]
        for j in range(i, len(sorted_2b_comb)):
            ket = sorted_2b_comb[j]
            
            srt_key = (*bra, *ket)
            if srt_key in dict_1:
                dict_2[srt_key] = dict_1[srt_key]
    
    ## 0.2 In case of modifying to 4-key index: ----------
    if _transform_to_key2:
        dict_2 = __copyHamiltonian_4keyTo2keys(dict_2) 
    return dict_2


def recursiveSumOnDictionaries(dict2read, dict2write):
    """
    Given dictionaries with numerical end point values, sum the value to read if
    keys exists or create them if it doesn't.
    
    Subroutine (doesn't return) that changes dict2write.
    """
    if isinstance(dict2read, dict):
        for index, values in dict2read.items():
            
            if index in dict2write:
                dict2write[index] = recursiveSumOnDictionaries(values, 
                                                               dict2write[index])
            else:
                dict2write[index] = recursiveSumOnDictionaries(values, {})
        return dict2write
    else:
        # dict2read, dict2write are final values, copy or
        if dict2write == {}:
            return dict2read
        else:
            return dict2read + dict2write

def almostEqual(a, b, tolerance=0):
    """ Input """
    if tolerance == 0:
        return (a == b) and (abs(a - b) < 1e-40)
    
    return abs(a - b) < tolerance

#===============================================================================
#
#===============================================================================

elementNameByZ = {
    1 : "H",      2 : "He",     3 : "Li",     4 : "Be",     5 : "B",
    6 : "C",      7 : "N",      8 : "O",      9 : "F",      10 : "Ne",
    11 : "Na",    12 : "Mg",    13 : "Al",    14 : "Si",    15 : "P",
    16 : "S",     17 : "Cl",    18 : "Ar",    19 : "K",     20 : "Ca",
    21 : "Sc",    22 : "Ti",    23 : "V",     24 : "Cr",    25 : "Mn",
    26 : "Fe",    27 : "Co",    28 : "Ni",    29 : "Cu",    30 : "Zn",
    31 : "Ga",    32 : "Ge",    33 : "As",    34 : "Se",    35 : "Br",
    36 : "Kr",    37 : "Rb",    38 : "Sr",    39 : "Y",     40 : "Zr",
    41 : "Nb",    42 : "Mo",    43 : "Tc",    44 : "Ru",    45 : "Rh",
    46 : "Pd",    47 : "Ag",    48 : "Cd",    49 : "In",    50 : "Sn",
    51 : "Sb",    52 : "Te",    53 : "I",     54 : "Xe",    55 : "Cs",
    56 : "Ba",    57 : "La",    58 : "Ce",    59 : "Pr",    60 : "Nd",
    61 : "Pm",    62 : "Sm",    63 : "Eu",    64 : "Gd",    65 : "Tb",
    66 : "Dy",    67 : "Ho",    68 : "Er",    69 : "Tm",    70 : "Yb",
    71 : "Lu",    72 : "Hf",    73 : "Ta",    74 : "W",     75 : "Re",
    76 : "Os",    77 : "Ir",    78 : "Pt",    79 : "Au",    80 : "Hg",
    81 : "Tl",    82 : "Pb",    83 : "Bi",    84 : "Po",    85 : "At",
    86 : "Rn",    87 : "Fr",    88 : "Ra",    89 : "Ac",    90 : "Th",
    91 : "Pa",    92 : "U",     93 : "Np",    94 : "Pu",    95 : "Am",
    96 : "Cm",    97 : "Bk",    98 : "Cf",    99 : "Es",    100 : "Fm",
    101 : "Md",   102 : "No",   103 : "Lr",   104 : "Rf",   105 : "Db",
    106 : "Sg",   107 : "Bh",   108 : "Hs",   109 : "Mt",   110 : "Ds ",
    111 : "Rg ",  112 : "Cn ",  113 : "Nh",   114 : "Fl",   115 : "Mc",
    116 : "Lv",   117 : "Ts",   118 : "Og"
}

def getCoreNucleus(Z, N, MzHO=None):
    
    if MzHO != None and MzHO > 3:
        raise Exception("getCoreNucleus using shells that no longer lead a HO core. MzHO <= 3!")
    Z = Z if isinstance(Z, int) else int(Z)
    N = N if isinstance(N, int) else int(N)
    
    if (Z, N) == (0, 0)   or (MzHO == 0):
        return 'NO'
    elif (Z, N) == (2, 2) or (MzHO == 1):
        return '4He'
    elif (Z, N) == (8, 8) or (MzHO == 2):
        return '16O'
    elif (Z, N) == (20, 20) or (MzHO == 3):
        return '40Ca'
    elif (Z, N) == (28, 28):
        return '56Ni'
    elif (Z, N) == (20, 28):
        return '48Ca'
    elif (Z, N) == (28, 50):
        return '78Ni'
    elif (Z, N) == (50, 50):
        return '100Sn'
    elif (Z, N) == (50, 82):
        return '132Sn'
    elif (Z, N) == (82, 126):
        return '208Pb'
    else:
        return 'UNIDENTIFIED !'

## Default Ordering of the shells.
SHO_shell_order = ['S', 'P', 'SD', 'F', 'PF', 'G', 
                   'SDG', 'H', 'PFH', 'I', 'SDGI', 'J']

valenceSpacesDict = {
    'S'   : ('001',),
    'P'   : ('103','101'),
    'SD'  : ('205', '1001', '203'),
    'F'   : ('307',),
    'PF'  : ('1103', '305', '1101'),
    'G'   : ('409',),
    'SDG' : ('1205', '407', '2001', '1203'),
    'H'   : ('511',),
    'PFH' : ('509', '1307', '1305', '2103', '2101'),
    'I'   : ('613',),
    'SDGI': ('1409', '2205', '611', '1407', '3001', '2203'),
    'J'   : ('715',)
    }
valenceSpacesDict_l_ge10 = {
    'S'   : ('001',),
    'P'   : ('103','101'),
    'SD'  : ('205', '10001', '203'),
    'PF'  : ('307','10103', '305', '10101'),
    'SDG' : ('409','10205', '407', '20001', '10203'),
    'PFH' : ('511', '509', '10307', '10305', '20103', '20101'),
    'SDGI': ('613', '10409', '20205', '611', '10407', '30001', '20203'),
    'J'   : ('715',)
    }

valenceSpacesDict_l_ge10_byM = {
0 : ('001',) ,
1 : ('103', '101') ,
2 : ('205', '203', '10001') ,
3 : ('307', '305', '10103', '10101') ,
4 : ('409', '407', '10205', '10203', '20001') ,
5 : ('511', '509', '10307', '10305', '20103', '20101') ,
6 : ('613', '611', '10409', '10407', '20205', '20203', '30001') ,
7 : ('715', '713', '10511', '10509', '20307', '20305', '30103', '30101') ,
8 : ('817', '815', '10613', '10611', '20409', '20407', '30205', '30203', '40001') ,
9 : ('919', '917', '10715', '10713', '20511', '20509', '30307', '30305', '40103', '40101') ,
10 : ('1021', '1019', '10817', '10815', '20613', '20611', '30409', '30407', '40205', '40203', '50001') ,
11 : ('1123', '1121', '10919', '10917', '20715', '20713', '30511', '30509', '40307', '40305', '50103', '50101') ,
12 : ('1225', '1223', '11021', '11019', '20817', '20815', '30613', '30611', '40409', '40407', '50205', '50203', '60001') ,
13 : ('1327', '1325', '11123', '11121', '20919', '20917', '30715', '30713', '40511', '40509', '50307', '50305', '60103', '60101') ,
14 : ('1429', '1427', '11225', '11223', '21021', '21019', '30817', '30815', '40613', '40611', '50409', '50407', '60205', '60203', '70001') ,
15 : ('1531', '1529', '11327', '11325', '21123', '21121', '30919', '30917', '40715', '40713', '50511', '50509', '60307', '60305', '70103', '70101') ,
    }


angularMomentumLetterDict = {
    0:'s', 1:'p', 2:'d', 3:'f', 4:'g', 5:'h', 6:'i', 7:'j', 8:'k', 9:'l', 10:'m'
}

def readAntoine(index, l_ge_10=False):
    """     
    returns the Quantum numbers from string Antoine's format:
        :return: [n, l, j], None if invalid
        
    :l_ge_10 <bool> [default=False] format for l>10.
    """
    if isinstance(index, str):
        index = int(index)
    
    if(index == 1):
        return[0, 0, 1]
    else:
        if index % 2 == 1:
            _n_division = 10000 if l_ge_10 else 1000
            n = int((index)/_n_division)
            l = int((index - (n*_n_division))/100)
            j = int(index - (n*_n_division) - (l*100))# is 2j 
            
            if (n >= 0) and (l >= 0) and (j > 0):
                return [n, l, j]
    
    raise Exception("Invalid state index for Antoine Format [{}]".format(index))

def getAllPermutationsOf2Bstates(tb_states, l_ge_10, is_j_scheme=True):
    """ 
    Given all permutations of the quantum numbers < a,b| c,d> for pnpn states
    return in this order:
        [(0,1,2,3), (1,0,2,3), (0,1,3,2), (1,0,3,2),    :: bra-ket
         (2,3,0,1), (3,2,0,1), (2,3,1,0), (3,2,1,0) ]   :: ket-bra
    with their phases, if the quantum numbers are equal
    * If it cannot be permuted, then phs = 0
    
    OUTPUT: (all states permutations, particle-label perm associated, phs)
    
        WARNING: give the tb_states in the correct order (the one for export)
    In case of JT-scheme, it will not be considered the t_perm
    """
    _return_format2 = False
    if len(tb_states) == 2:
        _return_format2 = True
        tb_states = (tb_states[0][0], tb_states[0][1],
                     tb_states[1][0], tb_states[1][1])
    
    QQNN_PERMUTATIONS = [(0,1,2,3), (1,0,2,3), (0,1,3,2), (1,0,3,2),
                         (2,3,0,1), (3,2,0,1), (2,3,1,0), (3,2,1,0)]
    PART_LABEL_PERMUT = [(1,2,3,4), (3,4,1,2), (2,1,4,3), (4,3,2,1),
                         (1,3,2,4), (3,1,4,2), (2,4,1,3), (4,2,3,1)]
    
    permuts = []
    perv_permuts = set()
    for i, key_perm in enumerate(QQNN_PERMUTATIONS):
        phs = [1, 1]
        if   i in (1, 3, 6, 7):
            phs[0] = (-1)**((readAntoine(tb_states[0], l_ge_10)[2] + 
                             readAntoine(tb_states[1], l_ge_10)[2]) // 2)
        if i in (2, 3, 5, 7):
            phs[1] = (-1)**((readAntoine(tb_states[2], l_ge_10)[2] + 
                             readAntoine(tb_states[3], l_ge_10)[2]) // 2)
        
        phs = phs[0] * phs[1]
        
        st_perm = tuple([tb_states[k] for k in key_perm])
        t_perm  = tuple([0, *PART_LABEL_PERMUT[i], 5]) if is_j_scheme else None
        if st_perm in perv_permuts:
            phs = 0
        perv_permuts.add(st_perm)
        permuts.append( (st_perm, t_perm, phs))
    
    if _return_format2:
        permuts = []
        for bk, t_perm, phs in permuts:
            permuts.append( ((bk[0], bk[1]), (bk[2], bk[3]), t_perm, phs) )
    return permuts


def shellSHO_Notation(n, l, j=0):
    """
    Give the Shell state: (n,l)=(0,3) -> '0f', (n,l,j)=(1,2,3) -> '1d3/2' 
    """
    j = str(j)+'/2' if j != 0 else ''
    
    return '{}{}{}'.format(str(n), angularMomentumLetterDict[l], j)


shell_filling_order = []
shell_filling_order_ge_10 = []

def _shellFilling():
    """
    return the states ordered by default (the order in valenceSpacesDict).
    """
    
    for shell in SHO_shell_order:
        sp_states = valenceSpacesDict[shell]
        
        for spss in sp_states:
            n, l, j = readAntoine(spss)
            st_ge_10 = str((10000*n) + (1000*l) + j)
            
            deg = j + 1
            shell_filling_order.append((spss, (n,l,j), deg))
            shell_filling_order_ge_10.append((st_ge_10, (n,l,j), deg))

_shellFilling()

def getStatesAndOccupationUpToLastOccupied(N, N_min=0):
    """ Function that return the quantum numbers occupied from an N_min (which 
    must fill a subshell, otherwise it will return the subshell as empty)
    until an N max.
    
    Use: States occupied to evaluate the density of N particles.
    """
    end_ = False
    vals = []
    i, n = 0, 0
    while(not end_):
        
        #if l_ge_10:
        _, spss, deg = shell_filling_order_ge_10[i]
        # else:
        #     _, spss, deg = shell_filling_order[i]
        
        n += deg
        i += 1
        if n > N_min:
            if n <= N: 
                vals.append((spss, deg))
            else:
                vals.append((spss, deg - (n - N)))
                end_ = True
        #print(f"st[{x}] n:{n:2} deg:{deg:2}  "+str(vals))
    return vals

def getStatesUpToLastOccupied(N, N_min=0):
    
    vals = getStatesAndOccupationUpToLastOccupied(N, N_min)
    vals = [val[0] for val in vals]
    return vals

def getStatesAndOccupationOfFullNucleus(Z, N, NZ_min=0):
    """ Do the same that getStatesAndOccupationUpToLastOccupied() but for proton
    and Neutrons separatedly, the return is a tuple with all the states and the 
    ocuppation for protons, neutrons:
    
    <list>[<tuple>(<tuple>spss, <int> occupation protons, <int> occ. neutrons)]
    
    Output: N=8, Z=5 N_min=2: 
    [((0, 1, 3), 3, 4), 
     ((0, 1, 1), 0, 2), 
     ((1, 0, 1), 0, 0)] 
    """
    occupation_Z = getStatesAndOccupationUpToLastOccupied(Z, NZ_min)
    occupation_N = getStatesAndOccupationUpToLastOccupied(N, NZ_min)
    
    qqnn = map(lambda x: x[0], max(occupation_Z, occupation_N))
    occupation = []
    i = 0
    for spss in qqnn:
        deg_Z = occupation_Z[i][1] if i < len(occupation_Z) else 0
        deg_N = occupation_N[i][1] if i < len(occupation_N) else 0
        
        occupation.append((spss, deg_Z, deg_N))
        i += 1
    return occupation
        

# if __name__ == "__main__":
#
#     print('13=\n'+'\n'.join(map(lambda x: str(x), getStatesUpToLastOcupied(13))))
#     print('43=\n'+'\n'.join(map(lambda x: str(x), getStatesUpToLastOcupied(43))))
#     print('10=\n'+'\n'.join(map(lambda x: str(x), getStatesUpToLastOcupied(10, 2))))
#     print('23=\n'+'\n'.join(map(lambda x: str(x), getStatesUpToLastOcupied(23))))
