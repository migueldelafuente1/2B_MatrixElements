'''
Created on 20 dic 2024

@author: delafuente
'''

from helpers.Enums import CentralGeneralizedMEParameters, SHO_Parameters, \
    BrinkBoekerParameters, CentralWithExchangeParameters
from matrix_elements.CentralForces import CentralGeneralizedForce_JTScheme,\
    CentralForce_JTScheme
from helpers.WaveFunctions import QN_2body_jj_JT_Coupling, QN_1body_jj

def _permutate_ket_wavefunction(ket):
    raise  Exception("TODO:")
    kets = []
    return kets

def _get_nlj_qn_by_Nshell(rho):
    qqnn = []
    for N1 in range(0, rho+1):
        N2 = rho - N1
        for n1 in range(N1//2 +1):
            l1 = N1 - 2*n1
            for n2 in range(N2//2 + 1):
                l2 = N2 - 2*n2
                for s in (1, -1):
                    if l1!=0 or s!=-1:
                        j1 = 2*l1 + s
                    if l2!=0 or s!=-1:
                        j2 = 2*l2 + s
                
                    qqnn.append( ((n1,l1,j1), (n2,l2,j2)) )
    return qqnn

def _get_J_coupling(a, b, c, d):
    ja, jb = a[2], b[2]
    jc, jd = c[2], d[2]
    jab = [j for j in range(abs(ja-jb), ja+jb + 1, 2)]
    jcd = [j for j in range(abs(jc-jd), jc+jd + 1, 2)]
    
    valid_j = []
    for j in jab:
        if j in jcd: valid_j.append(j // 2)
    return valid_j

def test_antisymmetry_quantum_numbers(MZmin, MZmax, TBME_class):
    """
    Iterate all elements for bra and ket for all p-n combinations
    requires method to permutate and test
    """
    
    pass_ = [0, 0, 0, 0]
    fail_ = [0, 0, 0, 0]
    all_  = [0, 0, 0, 0]
    
    TBME_class.setInteractionParameters(**kwargs)
    
    for rho1 in range(MZmin, 2*MZmax+1):
        qqnn1 = _get_nlj_qn_by_Nshell(rho1)
        print()
        print(" * rho 1 =", rho1, "  # qqnn=", len(qqnn1))
        for rho2 in range(rho1, 2*MZmax+1):
            qqnn2 = _get_nlj_qn_by_Nshell(rho2)
            print(" * * rho 2 =", rho2, "  # qqnn=", len(qqnn2))
            
            for a, b in qqnn1:
                aa, bb = QN_1body_jj(*a, mt=-1), QN_1body_jj(*b, mt=-1)
                for c, d in qqnn2:
                    cc, dd = QN_1body_jj(*c, mt=-1), QN_1body_jj(*d, mt=-1)
                    for J in _get_J_coupling(a, b, c, d):
                        for T in (0, 1):
                            bra = QN_2body_jj_JT_Coupling(aa, bb, J, T)
                            ket = QN_2body_jj_JT_Coupling(cc, dd, J, T)
                            
                            k_e = QN_2body_jj_JT_Coupling(dd, cc, J, T)
                            
                            phs = T + J + (c[2] + d[2])//2
                            phs = (-1)**phs
                            me     = TBME_class(bra, ket)
                            me_exc = TBME_class(bra, k_e) 
                            me_inv  = TBME_class(ket, bra)
                            meiexc = TBME_class(k_e, bra)
                            
                            zero = [abs(m.value)<1.0e-6 for m in (me, me_exc,
                                                                  me_inv, meiexc)]
                            for i in range(4): all_[i] += 1
                            if all(zero): 
                                for i in range(4): pass_[i] += 1
                                continue
                            
                            ## me - me_exch 
                            if abs(me.value - phs*me_exc.value) < 1.0e-6: 
                                pass_[0] += 1
                            else:
                                print(" ERR: <ab,cd> =", aa, bb, cc, dd, 
                                      f"={me.value:9.3f} J,T={J},{T} p={phs} ={me_exc.value:9.3f}") 
                                fail_[0] += 1
                            ## me = me invrsed
                            if abs(me.value - me_inv.value) < 1.0e-6: 
                                pass_[1] += 1
                            else: fail_[1] += 1
                                
                            ## me inversed - me exchanged
                            if abs(me_inv.value - phs*me_exc.value) < 1.0e-6: 
                                pass_[2] += 1
                            else: fail_[2] += 1
                            ## me inversed - me_inversed exchanged
                            if abs(me_inv.value - phs*meiexc.value) < 1.0e-6: 
                                pass_[3] += 1
                            else: fail_[3] += 1
        print(f"PASS : {pass_}")
        print(f"FAIL : {fail_}")
        print(f"TOTAL: {all_}")

if __name__ == '__main__':
    
    kwargs = {
        CentralGeneralizedMEParameters.potential:   {'name': 'gaussian'},
        CentralGeneralizedMEParameters.constant :   {'value': 1,},
        CentralGeneralizedMEParameters.mu_length:   {'value': 1,},
        CentralGeneralizedMEParameters.constant_R:  {'value': 1,},
        CentralGeneralizedMEParameters.mu_length_R: {'value': 1,},
        CentralGeneralizedMEParameters.potential_R: {'name': 'gaussian'},
        BrinkBoekerParameters.Wigner:     {'value': 1000,} ,
        BrinkBoekerParameters.Bartlett:   {'value': 0,},
        BrinkBoekerParameters.Heisenberg: {'value': 0,},
        BrinkBoekerParameters.Majorana:   {'value': 0,},
        SHO_Parameters.b_length: 1.856,
    }
    # kwargs = {
    #     CentralWithExchangeParameters.potential  : {'name': 'gaussian'},
    #     CentralWithExchangeParameters.Wigner     : {'value': 1.0},
    #     CentralWithExchangeParameters.Heisenberg : {'value': 0.0},
    #     CentralWithExchangeParameters.Bartlett   : {'value': 0.0},
    #     CentralWithExchangeParameters.Majorana   : {'value': 0.0},
    #     CentralWithExchangeParameters.mu_length  : {'value': 1.2},
    #     SHO_Parameters.b_length: 1.856,
    # }
    
    test_antisymmetry_quantum_numbers(0, 3, CentralGeneralizedForce_JTScheme)
    # test_antisymmetry_quantum_numbers(0, 3, CentralForce_JTScheme)
    
    