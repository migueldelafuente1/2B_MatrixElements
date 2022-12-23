'''
Created on Oct 26, 2022

@author: Miguel
'''
import numpy as np
import os
from copy import deepcopy, copy

import helpers
from helpers.Helpers import valenceSpacesDict_l_ge10_byM
if helpers.SCIPY_INSTALLED and helpers.MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d




HAMIL_FOLDER = 'hamils_MZ5'




def importMatrixElementsJscheme(hamil_filename, MZmin, MZmax):
    
    global HAMIL_FOLDER
    
    valid_states_ = []
    for M in range(MZmin, MZmax+1):
        valid_states_ = valid_states_ + list(valenceSpacesDict_l_ge10_byM[M])
    
    with open(os.path.join(HAMIL_FOLDER, hamil_filename), 'r') as f:
        data = f.readlines()
        
    skip_block = False
    final_2b  = [f'Truncated MZ=[{MZmin}, {MZmax}] From_ '+data[0], ]
    data_2b = {}
    
    for line in data[1:]:
        l_aux = line.strip()
        header = l_aux.startswith('0 5 ')
        
        if header:
            _, _,a,b,c,d, j0,j1 = l_aux.split()
            elem_abcd = (int(a), int(b), int(c), int(d))
            j0, j1 = int(j0), int(j1)
            J = j0
            
            skip_block = False
            for qn in (a, b, c, d): 
                qn = '001' if qn == '1' else qn 
                if qn not in valid_states_:
                    skip_block = True
                    break
            
            if not skip_block:
                data_2b[elem_abcd] = dict([(j, None) for j in range(j0, j1+1)])
                final_2b.append(line)
            continue
        
        if skip_block: continue
        
        t_block = [float(v) for v in line.strip().split()]
        data_2b[elem_abcd][J] = t_block
        final_2b.append(line)
        J += 1
    
    h2_filename = 'bch_{}_MZ{}.2b'.format(hamil_filename, MZmax)
    h2_text = ''.join(final_2b)[:-2]  # omit the last jump /n
    with open(h2_filename, 'w+') as f:
        f.write(h2_text)
    
    return data_2b
    
def interpolationLinear(b, b0, b1, v0, v1):
    """ Basic interpolation, linear """
    v_int = v0 + ((v1 - v0)*(b - b0) / (b1 - b0))
    
    return v_int

def interpolationCuadratic(b, x1,x2,x3, y1,y2,y3):
    """ Interpolation for a parabola y(b) = A(b - x1)^2 + B(b - x1) + C
    points (x, y(x)) might be shuffled. """
    C = y1
    A = (y3-C - ((x3-x1)*(y2 - C)/(x2-x1))) / (((x3-x1)**2) - ((x3-x1)*(x2-x1)))
    B = (y2 - (A*((x2 - x1)**2) + C)) / (x2 - x1)
    
    return A*((b - x1)**2) + B*(b - x1) + C
    

def interpolationSplines(b, x=[], y=[]):
    
    assert len(x) == len(y), 'Lens for x and y are not equal'
    assert len(x) > 1, 'number of values to interpolate must be greater than 2'
    
    x = np.array(x)
    y = np.array(y)
    f = interp1d(x, y, kind='cubic')
    
    return f(b)



if __name__ == '__main__':
    
    MZmin, MZmax = 0, 5
    
    hamil_files = {}
    b_lengths_val = []
    b_lengths = []
    valid_qqnn = set()
    js_by_qqnn = {}
    
    for hf in filter(lambda x: x.endswith('.2b'), os.listdir(HAMIL_FOLDER)):
        ## expected format: "****_B1_52**_***.2b" for b=1.52... 
        _, bun, bdec, _ = hf.split('_')
        b = bun[1:]+'.'+bdec
        b_lengths.append(b)
        b_lengths_val.append(float(b))
    
        hamil = importMatrixElementsJscheme(hf, MZmin, MZmax)
        hamil_files[b] = (hf, hamil)
        
        valid_qqnn.update( set(hamil.keys()) )
    
    for st in hamil:
        js_by_qqnn[st] = list(hamil[st].keys()) 
    ## ********************************************************************* ##
    ## % Test the interpolation for known matrix elements
    
    b2interpolate = 1.700
    key_exact = None
    b_sorted = []
    for b, b_len in enumerate(b_lengths_val):
        if abs(b_len - b2interpolate) < 0.01:
            key_exact = b_lengths[b]
            continue # skip the test b = b2interpolate
        diff = b_len - b2interpolate
        b_sorted.append((diff, b))
    b_sorted = sorted(b_sorted, key=lambda x: x[1])
    print(b_sorted[:3], "close to b=", b2interpolate)
    
    N_SPLINES = 9
    b_nn      = [b_lengths_val[x[1]] for x in b_sorted[:N_SPLINES]]
    b_nn_keys = [b_lengths[x[1]] for x in b_sorted[:N_SPLINES]]
    dict_val  = {}
    diffs_exact   = {}
    diffs_abs     = {}
    
    diff_max = 0.0
    diff_list = []
    
    for st in valid_qqnn:
        dict_val   [st] = dict([(j, [None,]*6) for j in js_by_qqnn[st]])
        diffs_exact[st]   = deepcopy(dict_val[st])
        diffs_abs  [st]   = deepcopy(dict_val[st])
        
        for j in js_by_qqnn[st]:
            v_nn = [[None,]*6, [None,]*6, [None,]*6]
            v_nn = [hamil_files[k][1][st][j] for k in b_nn_keys]
            
            top_fail = False
            fail_msg = []
            for mt in range(6):
                vvs = [v_nn[i][mt] for i in range(len(b_nn))]
                
                if   N_SPLINES < 2:
                    raise Exception("Cannot interpolate with less than 2 values")
                elif N_SPLINES == 2:
                    inter_val = interpolationLinear(b2interpolate, *b_nn, *vvs)
                elif N_SPLINES == 3:
                    inter_val = interpolationCuadratic(b2interpolate,*b_nn,*vvs)
                else:
                    inter_val = interpolationSplines(b2interpolate, x=b_nn,y=vvs)
                
                dict_val[st][j][mt] = inter_val
                
                if not key_exact: 
                    continue
                exc_v = hamil_files[key_exact][1][st][j][mt]
                
                diff_ = (exc_v - inter_val)                
                diffs_exact [st][j][mt] = diff_
                
                diff_ = diff_ / exc_v  if abs(diff_) > 1.0e-12 else 0.0
                diff_ = abs(diff_)
                
                if diff_ > 1.0:
                    if mt in (2,3,4): 
                        continue
                    fail_msg.append(
                        f'({inter_val:7.4f} neq {exc_v:7.4f}: {100*diff_:4.1f}%){mt}')
                    top_fail = True
                
                diffs_abs   [st][j][mt] = diff_
                
                diff_max = max(diff_max, diff_)
                if diff_ > 0.50:
                    diff_list.append(diff_)
        
            if top_fail:
                print(f"[WRNG] ", *fail_msg, "on", st)
    plt.figure()
    counts, bins = np.histogram(diff_list, bins=30)
    # plt.stairs(bins, counts)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title("Distr. of differences in absolute value.\n Excluded < 1&")
    plt.show()
    
    ## ********************************************************************* ##
    
    #%% Plots
    #% Plot of the evolution for different type of matrix elements
    
    st_2print = ((203,)*4, 
                 (205,)*4,
                 (10001, )*4)
    st_2print = ((203,205,203,205), 
                 (203,10001,203,10001), 
                 (205,10001,205,10001), 
                 
                 (203,205,203,10001),
                 
                 (203,205,205, 10001),)
    st_2print = valid_qqnn
    # st_2print = ((203,205,203,10001), )
    # st_2print = ((203,205,205, 10001), 
    #              )
    J, mt = 4, 0
    
    fig, ax = plt.subplots()
    for st in st_2print:
        if st not in valid_qqnn:
            continue
        y = []
        for b in b_lengths:
            v_dict = hamil_files[b][1][st]
            if J not in v_dict.keys():
                continue                
            y.append(v_dict[J][mt])
        
        if len(y) > 0:
            ax.plot(b_lengths_val, y, label=str(st))
        y = []
        
    plt.legend()
    plt.show()
    
    
        
    
    
    
    
    