'''
Created on Feb 25, 2021

@author: Miguel
'''
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

def safe_wigner_9j(*args):
    """ Wigner 9j symbol, same arguments as in Avoid the ValueError whenever the
     arguments don't fulfill the triangle relation. """
    try: 
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

def safe_wigner_6j(*args):
    """ Wigner 6j symbol, same arguments as in Avoid the ValueError whenever the
     arguments don't fulfill the triangle relation. """
    try: 
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
    """
    
    list_ = [*min((n, l), (n_q, l_q)), *max((n, l), (n_q, l_q)), p]
    return ','.join(map(lambda x: str(x), list_))

#===============================================================================

#===============================================================================
#     
#===============================================================================
def prettyPrintDictionary(dictionary, level=0, delimiter=' . '):
    
    header = ''.join([delimiter]*level)
    for k, val in dictionary.items():
        if isinstance(val, dict):
            print(header+str(k)+': {')
            prettyPrintDictionary(val, level + 1, delimiter)
            print(header+'}')
        else:
            print(header+str(k)+':'+str(val))
        


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
    111 : "Rg ",  112 : "Cn ",  113 : "Nh",   114 : "Fl",   115 : "Mc"
}

def getCoreNucleus(Z, N):
    
    Z = Z if isinstance(Z, int) else int(Z)
    N = N if isinstance(N, int) else int(N)
    
    if (Z, N) == (0, 0):
        return 'NO'
    elif (Z, N) == (2, 2):
        return '4He'
    elif (Z, N) == (8, 8):
        return '16O'
    elif (Z, N) == (20, 20):
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
    'F'   : ('307',),
    'PF'  : ('10103', '305', '10101'),
    'G'   : ('409',),
    'SDG' : ('10205', '407', '20001', '10203'),
    'H'   : ('511',),
    'PFH' : ('509', '10307', '10305', '20103', '20101'),
    'I'   : ('613',),
    'SDGI': ('10409', '20205', '611', '10407', '30001', '20203'),
    'J'   : ('715',)
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
