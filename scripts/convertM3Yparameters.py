'''
Created on 30 dic 2024

@author: delafuente
'''
import numpy as np
from helpers.Helpers import prettyPrintDictionary

## Matrix to pass from P[TS] to Bartlett, Majorana ... (order of the outcomming operator)
CONV_MATRIX = [
    #W, -H,  B, -M 
    [1,  1, -1, -1],  # P 00 = SO
    [1, -1, -1,  1],  # P 10 = SE 
    [1,  1,  1,  1],  # P 01 = TE
    [1, -1,  1, -1],  # P 11 = TO
]
CONV_MATRIX = np.array(CONV_MATRIX) 

## Matrix to pass from P[TS] (article ) to Bartlett, Majorana ... (inverse)
CONV_INV_MATRIX = [
    #SO  SE  TE  TO
    [ 1,  1,  1,  1],  # 1
    [ 1, -1,  1, -1],  # -Heisenberg
    [-1, -1,  1,  1],  # Bartlett
    [-1,  1,  1, -1],  # -Majorana
]
CONV_INV_MATRIX = np.array(CONV_INV_MATRIX) * 0.25

## TODO: Test if it is inverse
print(np.dot(CONV_INV_MATRIX, CONV_MATRIX), "== unitary(4)\n")
print(np.dot(CONV_MATRIX, CONV_INV_MATRIX), "== unitary(4)\n")
print()


def print_xml_elements(params_c, params_ls, params_t):
    
    part_c, part_ls, part_t = {}, {}, {}
    for i in range(3):
        part  = f'part_{i+1}'
        vals  = [e[i] for e in params_c[:4]]
        vals_c = np.dot(CONV_INV_MATRIX, np.array(vals))
        vals_c = np.append(vals_c,  params_c[4][i])
        
        if i < 2:
            vals    = [e[i] for e in params_ls[:4]]
            vals_ls = np.dot(CONV_INV_MATRIX, np.array(vals))
            vals_ls = np.append(vals_ls, params_ls[4][i])
            
            vals    = [e[i] for e in params_t[:4]]
            vals_t  = np.dot(CONV_INV_MATRIX, np.array(vals))
            vals_t  = np.append(vals_t,  params_t[4][i])
        
        part_ls[part], part_t[part] = {}, {}
        for j, elem in enumerate(('Wigner', 'Heisenberg', 'Bartlett', 'Majorana', 'mu_length')):
            sgn=1 #sgn = -1 if elem=='Heisenberg' else 1
            if i == 0:
                part_c[elem] = {part: sgn*vals_c[j], }
            else:
                part_c[elem][part]  = sgn*vals_c[j]
            if i < 2:
                part_ls[part][elem] = {'value': sgn*vals_ls[j], }
                part_t [part][elem] = {'value': sgn*vals_t [j], }
        
        if i < 2:
            part_ls[part]['potential'] = {'name': 'yukawa', }
            part_t [part]['potential'] = {'name': 'yukawa', }
    
    ## Central:
    print("\t\t<M3Y_yukawians active='True'>")
    print(f"\t\t\t<!-- Yukawians from force M3Y - Central -->")
    for elem in ( 'mu_length', 'Wigner', 'Bartlett', 'Heisenberg', 'Majorana'):
        line = f"\t\t\t<{elem: <11}"
        for i in range(3):
            part  = f'part_{i+1}'
            if elem == 'mu_length':
                line += f"part_{i+1}='{part_c[elem][part]:6.4f}' "
            else:
                line += f"part_{i+1}='{part_c[elem][part]:6.6f}' "
        line += '/>'
        print(line)
    print('\t\t</M3Y_yukawians>')
    
    #===========================================================================
    # # Extension for the new combined Tensor LS yukawian matrix elements
    print("\t\t<M3Y_tensor active='True'>")
    print(f"\t\t\t<!-- Yukawians from force M3Y - Tensor S12 -->")
    for elem in ( 'mu_length', 'Wigner', 'Bartlett', 'Heisenberg', 'Majorana'):
        line = f"\t\t\t<{elem: <11}"
        for i in range(2):
            part  = f'part_{i+1}'
            if elem == 'mu_length':
                line += f"part_{i+1}='{part_t[part][elem]['value']:6.4f}' "
            else:
                line += f"part_{i+1}='{part_t[part][elem]['value']:6.6f}' "
        line += '/>'
        print(line)
    print('\t\t</M3Y_tensor>')
    print("\t\t<M3Y_SpinOrbit active='True'>")
    print(f"\t\t\t<!-- Yukawians from force M3Y - Spin-Orbit -->")
    for elem in ( 'mu_length', 'Wigner', 'Bartlett', 'Heisenberg', 'Majorana'):
        line = f"\t\t\t<{elem: <11}"
        for i in range(2):
            part  = f'part_{i+1}'
            if elem == 'mu_length':
                line += f"part_{i+1}='{part_ls[part][elem]['value']:6.4f}' "
            else:
                line += f"part_{i+1}='{part_ls[part][elem]['value']:6.6f}' "
        line += '/>'
        print(line)
    print('\t\t</M3Y_SpinOrbit>')
    
    #===========================================================================
    
    for i in range(2):
        print("\t\t<TensorS12 active='True'>")
        part  = f'part_{i+1}'
        print(f"\t\t\t<!-- Yukawians from force M3Y - tensor : {part} -->")
        for elem in ('potential', 'mu_length', 'Wigner', 'Bartlett', 'Heisenberg', 'Majorana'):
            line = f"\t\t\t<{elem: <11} "
            for k, val in part_t[part][elem].items():
                if elem != 'potential': 
                    val = f"{val:6.4f}" if elem == 'mu_length' else f"{val:6.6f}"
                line += f"{k}='{val}'"
            line += '/>'
            print(line)
        print('\t\t</TensorS12>')
    for i in range(2):
        print("\t\t<SpinOrbitFiniteRange active='True'>")
        part  = f'part_{i+1}'
        print(f"\t\t\t<!-- Yukawians from force M3Y - LS : {part} -->")
        for elem in ('potential', 'mu_length', 'Wigner', 'Bartlett', 'Heisenberg', 'Majorana'):
            line = f"\t\t\t<{elem: <11} "
            for k, val in part_ls[part][elem].items():
                if elem != 'potential': 
                    val = f"{val:6.4f}" if elem == 'mu_length' else f"{val:6.6f}"
                line += f"{k}='{val}'"
            line += '/>'
            print(line)
        print('\t\t</SpinOrbitFiniteRange>')
        
    #--------------------------------------------------------------------------
    printAsDictionary(part_c,  0, name='central')
    printAsDictionary(part_ls, 1, name='spinOrb')
    printAsDictionary(part_t,  1, name='tensor')

def printAsDictionary(part, type_, name=''):
    delim = '    '
    # print(f'{name}=','{')
    print(f'"{name}":','{')
    if type_ == 1:
        new_dict = {}
        for k, dict_ in part.items():
            for k_part, val in dict_.items():
                if k_part == 'potential': continue
                val = val['value']
                if k_part in new_dict:
                    new_dict[k_part][k] = val
                else:
                    new_dict[k_part] = {k: val, }
        part = new_dict
            
    for k_part, dicts_ in part.items():
        vals_ = ''
        for k, val in dicts_.items():
            # vals_ += f'"{k}": "{val:7.4f}", '
            k = int(k.replace('part_', ''))
            vals_ += f'{k}: {val:7.4f}, '
        # print(f"{delim}'{k_part}': {{{vals_}}},")
        print(f"{delim}'{k_part}': {{{vals_}}},")
    print('},')
    
if __name__ == '__main__':
    
    # M3Y - P2
    #===========================================================================
    par_p2 = [
        [-11900,  2730, 31.389,], # SO
        [  8027, -2880,-10.463,], # SE
        [  6080, -4266,-10.463,], # TE
        [  3800,  -780,  3.488,], # TO
        [  0.25,   0.4,  1.414,], # 1/mu
    ]
    par_p2_LS = [
        [  0.0,  0.0], # SO
        [  0.0,  0.0], # SE
        [-9181.8,  -606.6], # TE
        [-3414.6, -1137.6], # TO
        [ 0.25, 0.40] # 1/mu
    ]
    par_p2_T = [
        [    0.0, 0.0], # SO
        [    0.0, 0.0], # SE
        [-131.52,-3.708], # TE
        [  29.28, 1.872], # TO
        [   0.40, 0.70] # 1/mu 
    ]
    print("P2 - M3Y parametrization")
    print_xml_elements(par_p2, par_p2_LS, par_p2_T)
    print()
    
    # M3Y - P6
    #===========================================================================
    par_p2 = [
        [  -728,  1386, 31.389,], # SO
        [ 10766, -3520,-10.463,], # SE
        [  8474, -4594,-10.463,], # TE
        [ 12453, -1588,  3.488,], # TO
        [  0.25,   0.4,  1.414,], # 1/mu
    ]
    par_p2_LS = [
        [  0.0,  0.0], # SO
        [  0.0,  0.0], # SE
        [ -11222.2, -741.4], # TE
        [  -4173.4,-1390.4], # TO
        [ 0.25, 0.40] # 1/mu
    ]
    par_p2_T = [
        [  0.0,  0.0], # SO
        [  0.0,  0.0], # SE
        [-1096, -30.9], # TE
        [  244,  15.6], # TO
        [   0.40, 0.70] # 1/mu 
    ]
    
    par_p6_dens = [
        [  0.0], # SO
        [  384], # SE
        [ 1930], # TE
        [  0.0], # TO
        ]
    
    print("P6 - M3Y parametrization")
    print_xml_elements(par_p2, par_p2_LS, par_p2_T)
    print()
    
    # M3Y - P0
    #===========================================================================
    par_p0 = [
        [-1418,  950, 31.389,], # SO
        [11466,-3556,-10.463,], # SE
        [13967,-4594,-10.463,], # TE
        [11345,-1900,  3.488,], # TO
        [ 0.25,  0.4,  1.414,], # 1/mu
    ]
    par_p0_LS = [
        [  0.0,  0.0], # SO
        [  0.0,  0.0], # SE
        [-5101, -337], # TE
        [-1897, -632], # TO
        [ 0.25, 0.40] # 1/mu
    ]
    par_p0_T = [
        [  0.0,  0.0], # SO
        [  0.0,  0.0], # SE
        [-1096,-30.9], # TE
        [  244, 15.6], # TO
        [ 0.40,  0.70] # 1/mu 
    ]
    print("P0 - M3Y parametrization")
    print_xml_elements(par_p0, par_p0_LS, par_p0_T)
    print()
    