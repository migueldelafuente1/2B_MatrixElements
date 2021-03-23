# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:55:30 2018

@author: miguel
"""
import numpy as np

from helpers.Helpers import fact, safe_racah
#===============================================================================
# BRODY - MOSHINSKY TRANSFORMATION FUNCTIONS
#===============================================================================

def matrix_r2(n,l,N,L, n_q,l_q,N_q,L_q, lambda_):
    """
    Matrix elements for -r1^2 necessary for the BM Transformation.
    """
    if((n_q<0) or (l_q<0) or (N_q<0) or(L_q<0)):
        return 0
    
    # Tabulated Relations for the 6 non-zero (n_q,l_q,N_q,L_q) elements
    if(n_q == n - 1):
        # relations 1, 3, 4
        if(l_q == l):
            if(N_q == N):
                if(L_q == L):
                    # relation 1 (n_q=n - 1, l_q=l    , N_q=N    , L_q=L    )
                    return 0.5 * np.sqrt(n*(n + l + 0.5))
                else:
                    return 0
            else:
                return 0
        
        elif(l_q == l + 1):
            if(N_q == N - 1):
                if(L_q == L + 1):
                    # relation 3 (n_q=n - 1, l_q=l + 1, N_q=N - 1, L_q=L + 1)
                    racah_aux = safe_racah(l,(l + 1),L,(L + 1), 1, lambda_)
                        
                    return ((-1)**(lambda_ + L + l)
                            * np.sqrt(n*N*(l + 1)*(L + 1)) * racah_aux)
                else:
                    return 0
                
            elif(N_q == N):
                if(L_q == L - 1):
                    # relation 4 (n_q=n - 1, l_q=l + 1, N_q=N   , L_q=L - 1)
                    racah_aux = safe_racah(l,(l+1),L,(L-1), 1, lambda_)
                    
                    return ((-1)**(lambda_ + L + l)
                            * np.sqrt(n*L*(l + 1)*(N + L + 0.5)) * racah_aux)
                else:
                    return 0
            else: 
                return 0
            
        else:
            return 0
            
        
    elif(n_q == n):
        # relations 2,5,6
        if(l_q == l - 1):
            if(N_q == N - 1):
                if(L_q == L + 1):
                    # relation 5 (n_q=n    , l_q=l - 1, N_q=N - 1, L_q=L + 1)
                    racah_aux = safe_racah(l,(l - 1),L,(L + 1), 1, lambda_)
                    
                    return ((-1)**(lambda_ + L + l)
                            * np.sqrt(N*l*(n + l + 0.5)*(L + 1)) * racah_aux)
                else:
                    return 0
                
            elif(N_q == N):
                if(L_q == (L-1)):
                    # relation 6 (n_q=n    , l_q=l - 1, N_q=N - 1, L_q=L - 1)
                    racah_aux = safe_racah(l,(l - 1),L,(L - 1), 1, lambda_)
                    
                    return ((-1)**(lambda_ + L + l)
                            * np.sqrt(L*l*(n + l + 0.5)*(N + L + 0.5)) 
                            * racah_aux)
                else:
                    return 0
            else:
                return 0
                
        elif(l_q == l):
            if(N_q == (N-1)):
                if(L_q == L):
                    # relation 2 (n_q=n    , l_q=l    , N_q=N - 1, L_q=L    )
                    return 0.5*np.sqrt(N*(N + L + 0.5))
                else:
                    return 0
            else:
                return 0
        else:
            return 0
    
    else:
        return 0
    
    return ValueError


def _A_coeff(l1, l, l2, L, x):
    """ Coefficient for the BMB_00 """
    
    const = 0.5 * (fact(l1 + l + x + 1) + fact(l1 + l - x)
                   + fact(l1 + x - l) - fact(l + x - l1))
    
    const += 0.5 * (fact(l2 + L + x + 1) + fact(l2 + L - x)
                    + fact(l2 + x - L) - fact(L + x - l2))
    
    const = np.exp(const)
    aux_sum = 0.0
    
    # limits for non negative factorials
    # q is non negative
    c1 = l1 - l
    c2 = l2 - L
    c3 = -x - 1

    c4 = l + l1
    c5 = L + l2

    max_ = min(c4, c5);
    min_ = max(max(max(max(c1, c2), c3), x), 0)
    
    for q in range(min_, max_ +1):
        
        if( ((l + q - l1)%2) == 0 ):
            numerator = (fact(l + q - l1) + fact(L + q - l2))
            
            denominator = ((fact((l + q - l1)//2) 
                + fact((l + l1 - q)//2) + fact(q - x) + fact(q + x + 1)
                + fact((L + q - l2)//2) + fact((L + l2 - q)//2)))
            
            aux_sum += (-1)**((l + q - l1)//2) * np.exp(numerator - denominator)
    
    return const * aux_sum

#===============================================================================
# BMB coefficients by Memorization Pattern
#===============================================================================
#
# Global dictionary for BMB coefficients memorization pattern
# Index : comma separated indexes string: 'n,l,N,L,n1,l1,n2,l2,lambda'
# 
# Size in dynamic memory (acceptable), includes negative indexes.
# 25   K bmb_s -> 320   KB
# 120  K bmb_s -> 1.25  MB
# 1000 K bmb_s -> 10.5  MB
#
#===============================================================================

def _BMB_memo_accessor(n, l, N, L, n1, l1, n2, l2, lambda_):
    return ','.join(map(lambda x: str(x), [n, l, N, L, n1, l1, n2, l2, lambda_]))

_BMB_Memo = {}

def BM_Bracket(n, l, N, L, n1, l1, n2, l2, lambda_):
    """ _Memorization Pattern for Brody-Moshinsky coefficients. """
    
    args = (n, l, N, L, n1, l1, n2, l2, lambda_)
    tpl = _BMB_memo_accessor(*args)
    
    global _BMB_Memo
    
    if not tpl in _BMB_Memo:
        if (n1 == 0) and (n2 == 0):
            args = (n, l, N, L, l1, l2, lambda_)
            _BMB_Memo[tpl] = _BM_Bracket00_evaluation(*args)
        else:
            _BMB_Memo[tpl] = _BM_bracket_evaluation(*args)
        
    return _BMB_Memo[tpl]

def BM_Bracket00(n,l,N,L, l1,l2, lambda_):
    """ _Memorization Pattern for Brody-Moshinsky coefficients. """
    return BM_Bracket(n, l, N, L, 0, l1, 0, l2, lambda_)

#===============================================================================
def _BMB_initial_comprobations(n, l, N, L, n1, l1, n2, l2, lambda_):
    
    # Non-negative conditions over constants
    if((n<0) or (l<0) or (N<0) or (L<0)):
        return 0
    if((n1<0) or (l1<0) or (n2<0) or (l1<0) or (lambda_<0)):
        return 0
    
    # Energy condition
    if (2*(n1 + n2) + l1 + l2) != (2*(n + N) + l + L):
        return 0 
    # Angular momentum conservation
    if (abs(l1 - l2) > lambda_) or ((l1 + l2) < lambda_):
        return 0
    if (abs(l - L) > lambda_) or ((l + L) < lambda_):
        return 0
    
    return 1

# def BM_Bracket00(n,l,N,L, l1,l2, lambda_):
def _BM_Bracket00_evaluation(n, l, N, L, l1, l2, lambda_):
    """ Limit of the recurrence relation n1=n2=0 """
     
    if _BMB_initial_comprobations(n, l, N, L, 0, l1, 0, l2, lambda_) == 0:
        return 0
    
    const = ((fact(l1) + fact(l2) + fact(n + l) + fact(N + L)) -
             (fact(2*l1) + fact(2*l2) + fact(n) + fact(N) +
              fact(2*(n + l) + 1) + fact(2*(N + L) + 1)))
    const += ((np.log(2*l + 1) + np.log(2*L + 1)) - ((l + L)*np.log(2)))
    
    aux_sum = 0.0
    
    max_ = min((l + l1), (L + l2))
    min_ = max(abs(l - l1), abs(L - l2))
    for x in range(min_, max_ +1):
        
        racah_aux = safe_racah(l, L, l1, l2, lambda_, x)
        
        aux_sum += (2*x + 1) * _A_coeff(l1, l, l2, L, x) * racah_aux
            
    return np.exp(0.5 * const) * aux_sum * ((-1)**(n + l + L - lambda_))


# def BM_Bracket(n, l, N, L, n1, l1, n2, l2, lambda_):
def _BM_bracket_evaluation(n, l, N, L, n1, l1, n2, l2, lambda_):
    """ 
    Brody Moshinsky Transformation brackets:
        <n,l,N,L | n1,l1,n2,l2 (lambda, mu=0)>
    """
    
    if _BMB_initial_comprobations(n, l, N, L, n1, l1, n2, l2, lambda_) == 0:
        return 0
    
    # RECURRENCE RELATIONS
    # there are only 6 non-zero combinations of n'l'N'L' 
    if(n1 == 0):
        if(n2 == 0):
            # BMB00
            return BM_Bracket00(n,l,N,L, l1,l2, lambda_)
        else:
            # permute the n1 l1 with n2 l2
            phase = (-1)**(L - lambda_)
            
            aux_sum = 0.
            aux_sum += (matrix_r2(n,l,N,L, n-1,l,N,L, lambda_)*
                        BM_Bracket(n-1,l,N,L, n2-1,l2,n1,l1, lambda_))
            aux_sum += (matrix_r2(n,l,N,L, n,l,N-1,L, lambda_)*
                        BM_Bracket(n,l,N-1,L, n2-1,l2,n1,l1, lambda_))
            aux_sum += (matrix_r2(n,l,N,L, n-1,l+1,N-1,L+1, lambda_)*
                        BM_Bracket(n-1,l+1,N-1,L+1, n2-1,l2,n1,l1, lambda_))
            aux_sum += (matrix_r2(n,l,N,L, n-1,l+1,N,L-1, lambda_)*
                        BM_Bracket(n-1,l+1,N,L-1, n2-1,l2,n1,l1, lambda_))
            aux_sum += (matrix_r2(n,l,N,L, n,l-1,N-1,L+1, lambda_)*
                        BM_Bracket(n,l-1,N-1,L+1, n2-1,l2,n1,l1, lambda_))
            aux_sum += (matrix_r2(n,l,N,L, n,l-1,N,L-1, lambda_)*
                        BM_Bracket(n,l-1,N,L-1, n2-1,l2,n1,l1, lambda_))
            
            aux_sum *= phase
            
            return  np.sqrt(1./(n2*(n2 + l2 + 0.5))) * aux_sum
            
    else:
        # normal case
        aux_sum = 0.
        aux_sum += (matrix_r2(n,l,N,L, n-1,l,N,L, lambda_)*
                    BM_Bracket(n-1,l,N,L, n1-1,l1,n2,l2, lambda_))
        aux_sum += (matrix_r2(n,l,N,L, n,l,N-1,L, lambda_)*
                    BM_Bracket(n,l,N-1,L, n1-1,l1,n2,l2, lambda_))
        aux_sum += (matrix_r2(n,l,N,L, n-1,l+1,N-1,L+1, lambda_)*
                    BM_Bracket(n-1,l+1,N-1,L+1, n1-1,l1,n2,l2, lambda_))
        aux_sum += (matrix_r2(n,l,N,L, n-1,l+1,N,L-1, lambda_)*
                    BM_Bracket(n-1,l+1,N,L-1, n1-1,l1,n2,l2, lambda_))
        aux_sum += (matrix_r2(n,l,N,L, n,l-1,N-1,L+1, lambda_)*
                    BM_Bracket(n,l-1,N-1,L+1, n1-1,l1,n2,l2, lambda_))
        aux_sum += (matrix_r2(n,l,N,L, n,l-1,N,L-1, lambda_)*
                    BM_Bracket(n,l-1,N,L-1, n1-1,l1,n2,l2, lambda_))
        
        return np.sqrt(1./(n1*(n1 + l1 + 0.5))) * aux_sum
        
    return ValueError("Error in BM Braket <(nlNL){}, (n,l)12{} (lambda={})>"
                      .format((n,l,N,L), (n1,l1, n2,l2), lambda_))



