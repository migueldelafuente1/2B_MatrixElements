'''
Created on 30 may 2025

@author: delafuente
'''
from helpers.Helpers import almostEqual

def read_J_hamiltonian(path_hamil, ignore_lines=1):
    """
    read a J-scheme hamiltonian:
    returns the sh_states
    """
    sh_states = []
    hamil_J   = {}
    ## Import the complete hamiltonian 2b
    with open(path_hamil) as f:
        data = f.readlines()[ignore_lines:]
        
        for line in data:
            line = line.strip()
            if line.startswith('0 5 '):
                aux = line.split()
                sh_abcd = tuple(aux[2:6])
                Jmin, Jmax = int(aux[6]), int(aux[7])
                J = Jmin
                
                hamil_J[sh_abcd] = dict([(j, [0,]*6) for j in range(J, Jmax+1)])
                continue
            else:
                assert J <= Jmax, "invalid J"
                j_vals = [float(x) for x in line.split()]
                for t in range(6):
                    hamil_J[sh_abcd][J][t] = j_vals[t]
                J += 1
    for sh_abcd in hamil_J.keys():
        sh_states = sh_states + list(sh_abcd)
    sh_states = list(set(sh_states))
    
    return sh_states, hamil_J

def test_aaaa_abab_J_symmetry_properties():
    """
    Statement: any Hamiltonian (tensor, exchange, gradient ... dependent) that 
    is isospin-conserving must have the following relations for pppp, pnpn & pnpn:
    
        * <aa|v|aa>_J
            * even J: <pp|v|pp> = <pn|v|pn> = <pn|v|np>
            *  odd J: <pp|v|pp> = 0  and <pn|v|pn> = - <pn|v|np>
            
        * <ab|v|ab>_J   (a /= b)
            * <pn|v|np> = <pp|v|pp> - <pn|v|pn>
        
    1. Evaluate the desired Hamiltonian and set the path [_2b_path]
    2. run.
    
    """
    _2b_path = '../results/KK_test_abSymm.2b'
    # _2b_path = '../results/KK_test_abSymm.com'
    
    # _2b_path = '../results/KK_MZ.2b'
    # _2b_path = '../results/KK_MZ.com'
    
    _, hamil_J = read_J_hamiltonian(_2b_path)
    
    TOL = 1.0e-8
    ok, fail, total = 0, 0, 0
    for sh_abcd, j_vals in hamil_J.items():
        
        if sh_abcd[:2] != sh_abcd[2:]: continue
        
        for J, t_vals in j_vals.items():
            if sh_abcd[0] == sh_abcd[1]:
                if J % 2 == 0:
                    X = [almostEqual(t_vals[0], t_vals[1], TOL), 
                         almostEqual(t_vals[0], t_vals[2], TOL),
                         almostEqual(t_vals[3], t_vals[0], TOL),
                         almostEqual(t_vals[5], t_vals[0], TOL)]
                    if all(X):
                        ok += 4
                    else:
                        fail += X.count(True)
                        _st1 = ' '.join([f"{x: >6}" for x in sh_abcd])
                        _st2 = ' '.join([f"{x: >+6.4f}".replace('+',' ') for x in t_vals])
                        print(f"[FAIL] (aa|aa) J even: [{_st1},J:{J}] = {_st2}  ({X})")
                else:
                    X = [almostEqual(t_vals[0],  0, TOL), 
                         almostEqual(t_vals[1], -t_vals[2], TOL),
                         almostEqual(t_vals[3], -t_vals[4], TOL),
                         almostEqual(t_vals[5],  t_vals[0], TOL)]
                    if all(X):
                        ok += 4
                    else:
                        fail += X.count(True)
                        _st1 = ' '.join([f"{x: >6}" for x in sh_abcd])
                        _st2 = ' '.join([f"{x: >+6.4f}".replace('+',' ') for x in t_vals])
                        print(f"[FAIL] (aa|aa) J odd: [{_st1},J:{J}] = {_st2}  ({X})")
                total += 4
            else:
                X = [almostEqual(t_vals[0] - t_vals[1], t_vals[2], TOL), 
                     almostEqual(t_vals[1], t_vals[4], TOL),
                     almostEqual(t_vals[2], t_vals[3], TOL),
                     almostEqual(t_vals[5], t_vals[0], TOL)]
                if all(X):
                    ok += 4
                else:
                    fail += X.count(True)
                    _st1 = ' '.join([f"{x: >6}" for x in sh_abcd])
                    _st2 = ' '.join([f"{x: >+6.4f}".replace('+',' ') for x in t_vals])
                    print(f"[FAIL] (ab|ab) J _N/A: [{_st1},J:{J}] = {_st2} ({X})")
                total += 4
    
    print("\nResult Of the test (every element counts up to 4):")
    print(f"\tOK[{ok}] FAIL[{fail}] / [{total}]") 



if __name__ == '__main__':
    
    test_aaaa_abab_J_symmetry_properties()