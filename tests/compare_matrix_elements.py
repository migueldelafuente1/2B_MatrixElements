'''
Created on 31 may 2025

@author: delafuente

Comparison of the matrix elements from a reference interaction to another one 
(for the reference valence space).

'''
import numpy as np
import statistics as st
from helpers.io_manager import TBME_Reader, readMatrixElementsJScheme
from helpers import MATPLOTLIB_INSTALLED
from helpers.Helpers import readAntoine, valenceSpacesDict_l_ge10_byM,\
    almostEqual, shellSHO_Notation
from copy import deepcopy

if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15

__2BME_abcd_ORDER = []

def __get_js_minmax(abcd_):
    js = [readAntoine(x, l_ge_10=1)[2] for x in abcd_]
    J_max = min(sum(js[:2])//2, sum(js[2:])//2)
    J_min = max(abs(js[0]-js[1])//2, abs(js[2]-js[3])//2)
    return J_min, J_max
    
def printTableMatrixElements2b(sh_states, interactions, *hamils, 
                               hamil_is_JT=[]):
    """
    print a table of the matrix elements 
    """
    global __2BME_abcd_ORDER
    N_inter = len(interactions)
    if not hamil_is_JT: hamil_is_JT = [0,]*len(interactions)
    
    key_by_st = dict([(sh, i+1) for i, sh in enumerate(sh_states)])
    print("\n",
          "  ".join([f"{i+1}: {shellSHO_Notation(*readAntoine(sh, 1))}" 
                     for i, sh in enumerate(sh_states)]),
          "\n")
    
    templ_abcd_J = " {} {} {} {}    {} "
    templ_v_J    = " {: >9.5f} {: >9.5f} {: >9.5f} {: >9.5f}    "
    templ_v_JT   = " {: >9.5f} {: >9.5f} {: >9.5f} "
    templ_s_J    = " {: >9} {: >9} {: >9} {: >9}    "
    templ_s_JT   = " {: >9} {: >9} {: >9} "
    L1, L2, L3 = 22, 44, 32
    
    title_ = templ_abcd_J.format(*"     ")
    for i, inter in enumerate(interactions):
        L = L2 
        if hamil_is_JT[i]: L = L3
        T, T_str = len(inter), inter
        if T % 2 == 1: 
            T, T_str = T+1, T_str + " "
        SPC = " "*(((L - T) // 2)) 
        title_ += SPC + T_str + SPC
    print(title_)
    
    title_ = templ_abcd_J.format(*"abcdJ")
    for i, inter in enumerate(interactions):
        if hamil_is_JT[i]:
            title_ += templ_s_JT.format("T=1", "pn-pn", "pn-np")
        else:
            title_ += templ_s_J.format("pp-pp", "pn-pn", "pn-np", "nn-nn")
    print(title_ + "\n" +"-"*len(title_)) 
            
    for abcd_ in __2BME_abcd_ORDER:
        all_null = all([not (abcd_ in hamil) for hamil in hamils])
        if all_null: continue
        
        keys_ = [key_by_st[x] for x in abcd_]
        
        J_min, J_max = __get_js_minmax(abcd_)
         
        for J in range(J_min, J_max+1):
            if J != J_min: head_ = templ_abcd_J.format(*"    ", J)
            else:          head_ = templ_abcd_J.format(*keys_, J)
            
            value_line = ''
            for i, hamil in enumerate(hamils):
                vals = [0.0,]*3 if hamil_is_JT[i] else [0.0,]*4
                
                if abcd_ in hamil:
                    if hamil_is_JT[i]:
                        vals = [hamil[abcd_][J][t] for t in (5, 1, 2)] 
                        # 5 to avoid using pppp due Coulomb
                    else:
                        vals = [hamil[abcd_][J][t] for t in (0, 1, 2, 5)]
                
                if hamil_is_JT[i]:
                    value_line += templ_v_JT.format(*vals)
                else:
                    value_line += templ_v_J. format(*vals)
            print(head_ + value_line)
        print("")
        
def printTableMatrixElements2b_2(sh_states, interactions, *hamils, 
                               hamil_is_JT=[]):
    """
    print a table of the matrix elements.
    Substitute the J elements by JT if required.
    """
    global __2BME_abcd_ORDER
    N_inter = len(interactions)
    if not hamil_is_JT: hamil_is_JT = [0,]*len(interactions)
    
    key_by_st = dict([(sh, i+1) for i, sh in enumerate(sh_states)])
    print("\n",
          "  ".join([f"{i+1}: {shellSHO_Notation(*readAntoine(sh, 1))}" 
                     for i, sh in enumerate(sh_states)]),
          "\n")
    
    templ_abcd_J = " {} {} {} {}    {} "
    templ_v_J    = " {: >9.5f} {: >9.5f} {: >9.5f} {: >9.5f}    "
    templ_v_JT   = " {: >9.5f} {: >9.5f}    "
    templ_s_J    = " {: >9} {: >9} {: >9} {: >9}    "
    templ_s_JT   = " {: >9} {: >9}    "
    L1, L2, L3 = 22, 44, 24
    
    title_ = templ_abcd_J.format(*"     ")
    for i, inter in enumerate(interactions):
        L = L2 
        if hamil_is_JT[i]: L = L3
        T, T_str = len(inter), inter
        if T % 2 == 1: 
            T, T_str = T+1, T_str + " "
        SPC = " "*(((L - T) // 2)) 
        title_ += SPC + T_str + SPC
    print(title_)
    
    title_ = templ_abcd_J.format(*"abcdJ")
    for i, inter in enumerate(interactions):
        if hamil_is_JT[i]:
            title_ += templ_s_JT.format("T=1", "T=0")
        else:
            title_ += templ_s_J.format("pp-pp", "pn-pn", "pn-np", "nn-nn")
    print(title_ + "\n" +"-"*len(title_)) 
            
    for abcd_ in __2BME_abcd_ORDER:
        all_null = all([not (abcd_ in hamil) for hamil in hamils])
        if all_null: continue
        
        keys_ = [key_by_st[x] for x in abcd_]
        
        J_min, J_max = __get_js_minmax(abcd_)
         
        for J in range(J_min, J_max+1):
            if J != J_min: head_ = templ_abcd_J.format(*"    ", J)
            else:          head_ = templ_abcd_J.format(*keys_, J)
            
            value_line = ''
            for i, hamil in enumerate(hamils):
                vals = [0.0,]*3 if hamil_is_JT[i] else [0.0,]*4
                
                if abcd_ in hamil:
                    if hamil_is_JT[i]:
                        Aabcd = (-1)**J
                        Aabcd = ((1 - Aabcd*(abcd_[0]==abcd_[1])) *
                                 (1 - Aabcd*(abcd_[2]==abcd_[3])) )
                        Aabcd = 1/Aabcd if not almostEqual(Aabcd, 0) else 0
                        vals = [hamil[abcd_][J][5],
                                Aabcd*(hamil[abcd_][J][1] - hamil[abcd_][J][2])]
                        
                        # 5 to avoid using pppp due Coulomb
                    else:
                        vals = [hamil[abcd_][J][t] for t in (0, 1, 2, 5)]
                
                if hamil_is_JT[i]:
                    value_line += templ_v_JT.format(*vals)
                else:
                    value_line += templ_v_J. format(*vals)
            print(head_ + value_line)
        print("")

def printMatrixElementsJ2b_1vs2(hamil_1, hamil_2, sh_states_sorted, interactions,
                                separate_by_particles=False, separate_by_J=False,
                                ):
    """
    The hamiltonian keys must follow the same order (the reading method already
    implements the sorting from valence_space_array)
    
    hamil_1 is the reference matrix elements.
    hamil_2 will be the ones to compare, 
    
    display plot for each matrix element abcd/J/t as (X(h-1), Y(h-2))
    """
    common_me_by_part = {}
    onlyH1_me_by_part = {}
    onlyH2_me_by_part = {}
    # TBME_Reader._JSchemeIndexing
    abcd_all = []
    E_max, E_min = -999999, 9999999
    INTER_1, INTER_2 = interactions
    J_max = max([readAntoine(x, l_ge_10=1)[2] for x in sh_states_sorted])
    
    x_values, y_values = {}, {}
    for J in range(J_max+1):
        x_values[J] = dict( [(t, []) for t in range(6)] )
        y_values[J] = dict( [(t, []) for t in range(6)] )
    
    for abcd_, j_vals_1 in hamil_1.items():
        abcd_all.append(abcd_)
        if abcd_ in hamil_2:
            j_vals_2 = hamil_2[abcd_]            
            common_me_by_part[abcd_] = dict([(j, [None,]*6) for j in j_vals_1.keys()])
            for J in j_vals_1.keys():
                for t in range(6):
                    x, y = j_vals_1[J][t], j_vals_2[J][t]
                    
                    rand_xy = 1e-6*np.random.random(), 1e-6*np.random.random()
                    x_values[J][t].append(x+rand_xy[0])
                    y_values[J][t].append(y+rand_xy[1])
                    common_me_by_part[abcd_][J][t] = (x, y)
                    if t in (0, 5, 1, 2):
                        E_max = max([E_max, x, y])
                        E_min = min([E_min, x, y])
        else:
            print(f"ME_1 [{INTER_1}] =",abcd_, f"not in Hamil 2 [{INTER_2}]")
            onlyH1_me_by_part[abcd_] = {}
            for J, j_vals in j_vals_1.items():
                for t in (0, 5, 1, 2):
                    x = j_vals[t]
                    E_max, E_min = max([E_max, x]), min([E_min, x])
                onlyH1_me_by_part[abcd_][J] = [(x, x) for x in j_vals]
    
    for abcd_, j_vals_2 in hamil_2.items():
        if not abcd_ in hamil_1:
            abcd_all.append(abcd_)
            print(f"ME_2 [{INTER_2}] =",abcd_, f"not in Hamil 1 [{INTER_2}]")
            onlyH2_me_by_part[abcd_] = {}
            for J, j_vals in j_vals_2.items():
                onlyH2_me_by_part[abcd_][J] = [(x, x) for x in j_vals]
    
    figures_, axes_ = [], []
    single_image    = False
    subplot_idx     = {0: (0,0), 5: (0,1), 1: (1,0), 2: (1,1)}
    subplot_label   = {0: 'pp-pp', 1: 'pn-pn', 2: 'pn-np', 5: 'nn-nn'}
    subplts_n = 2 if separate_by_particles else 1
    if separate_by_J:
        for J in range(J_max+1):
            fig, axs = plt.subplots(subplts_n, subplts_n, figsize=(7, 7))
            figures_.append(fig)
            axes_.append(axs)
    else:
        single_image = not separate_by_particles
        
        fig, axs = plt.subplots(subplts_n, subplts_n, figsize=(7, 7))
        figures_.append(fig)
        axes_.append(axs)
    
    _COLOR = {0: 'r', 1: 'g', 2:'b', 5:'k'}
    x_line = np.linspace(E_min*1.1, E_max*1.1, 100)
    y_line = - .571 * x_line
    ISOS_VALUES = (0, 1, 2, 5)
    print(f"\nMATRIX ELEMENTS R-Pearson VALUES [{INTER_1} vs {INTER_2}] -------------")
    ## Print every matrix element
    def __chi_2(x, y):
        m = [0.5*(x[i]+y[i]) for i in range(len(x))]
        for i in range(len(x)): m[i] = m[i] if abs(m[i])>1e-6 else 1
        c = [((x[i]-y[i])/1)**2 for i in range(len(x))]
        return np.sqrt(sum(c) / (len(x)-1))
    R_total = [[], []]
    R_total_by_t = dict([(t, deepcopy([[], []])) for t in ISOS_VALUES])
    R_total_by_J = {}
    R_total_by_Jt = {}
    slope_total = dict([(t, 0) for t in ISOS_VALUES] + [('total', 0), ])
    slope_total_by_J = {}
    x_t0, y_t0 = [], []
    for J in range(J_max+1):
        x, y = [], []
        R_total_by_Jt[J] = {}
        slope_total_by_J[J] = dict([(t, 0) for t in ISOS_VALUES] + [('total', 0), ])
        print(f" J = {J}")
        for t in ISOS_VALUES: #range(6):
            x = x + x_values[J][t]
            y = y + y_values[J][t]
            # if t in (1, 2): 
            x_t0 = x_t0 + x_values[J][t]
            y_t0 = y_t0 + y_values[J][t]
            
            R_total_by_t[t][0] = R_total_by_t[t][0] + x_values[J][t]
            R_total_by_t[t][1] = R_total_by_t[t][1] + y_values[J][t]
            R_total_by_Jt[J][t] = 0
            if len(x_values[J][t]) > 1:
                c2 = __chi_2(x_values[J][t], y_values[J][t])
                R_total_by_Jt[J][t] = st.correlation(x_values[J][t], y_values[J][t])
                slope_total_by_J[J][t] = st.linear_regression(x_values[J][t],y_values[J][t])[0]
            print(f"   t[{t}={subplot_label[t]}][len:{len(x):2}] R={R_total_by_Jt[J][t]:12.4f} S={slope_total_by_J[J][t]:12.3f}  X2={c2:10.3f}")
        R_total_by_J[J] = 0 if len(x) < 1 else st.correlation(x, y)
        # if t in (1, 2): 
        slope_total_by_J[J]['total'] = 1 if len(x) < 1 else st.linear_regression(x, y)[0]
        c2 = __chi_2(x, y)
        print(f"      TOT(J)  R={R_total_by_J[J]:12.4f} S={slope_total_by_J[J]['total']:12.3f}  X2={c2:10.3f}")
        R_total[0] = R_total[0] + x
        R_total[1] = R_total[1] + y
        
    # slope_total['total'] = st.linear_regression(R_total[0], R_total[1])[0]
    slope_total['total'] = st.linear_regression(x_t0, y_t0)[0]
    c2_tot = __chi_2(*R_total)
    R_total = st.correlation(*R_total)
    
    print()
    for t in ISOS_VALUES:
        # if t in (1, 2): 
        slope_total[t] = st.linear_regression(R_total_by_t[t][0], R_total_by_t[t][1])[0]
        c2 = __chi_2(R_total_by_t[t][0], R_total_by_t[t][1])
        R_total_by_t[t] = st.correlation(R_total_by_t[t][0], R_total_by_t[t][1])
        print(f" All J t[{t}={subplot_label[t]}] R={R_total_by_t[t]:12.4f} S={slope_total[t]:12.3f}  X2={c2:10.3f}")
    print(f" WHOLE R: {R_total:12.6f}  S: {slope_total['total']:12.6f}   X2={c2_tot:10.3f}")
    print("--------------------------------------------------------------------------")
    
            
    for i, abcd_ in enumerate(abcd_all):
        J_min, J_max = __get_js_minmax(abcd_)
        #print(f"Print: [{i}] J[{J_min}:{J_max}] =", abcd_)
        for J in range(J_min, J_max+1):
            
            fig_idx = J if separate_by_J else 0 
            kwargs_com = {'marker': '.'}
            kwargs_    = {'fontsize': 15}
             
            for t in ISOS_VALUES:
                
                sub_idx = subplot_idx[t] if separate_by_particles else None
                if (i == 0): 
                    # if (not separate_by_J) and (J==J_min): 
                    #     kwargs_com['label'] = subplot_label[t]
                    # elif (separate_by_J and not separate_by_particles):
                    #     kwargs_com['label'] = subplot_label[t]
                    if (not separate_by_particles):
                        if (separate_by_J or (J==J_min)):
                            kwargs_com['label'] = subplot_label[t] 
                            if separate_by_J: 
                                kwargs_com['label'] += r"   $R_{JT}=$"+f"{R_total_by_Jt[J][t]:6.4f}"
                            else:
                                kwargs_com['label'] += r"   $R_{T}=$"+f"{R_total_by_t[t]:6.4f}"
                
                if abcd_ in common_me_by_part:
                    x, y = common_me_by_part[abcd_][J][t]
                    if  J==3 and t==5:
                        _ = 0
                    if np.sign(x) != np.sign(y):
                        print("   ME S.SIGNS: ", 
                              "<{} {}|v|{} {}>[J={}, {}]".format(*abcd_, J, subplot_label[t]),
                              f"[{INTER_1}]={x: >9.5f} [{INTER_2}]={y: >9.5f}")
                    ## elements in 
                    
                    if i == 0:
                        _X_INTER_TITLE = f'{INTER_1} me. (MeV)'
                        _Y_INTER_TITLE = f'{INTER_2} me. (MeV)'
                        if sub_idx:
                            #subplot_idx     = {0: (0,0), 5: (0,1), 1: (1,0), 2: (1,1)}
                            if sub_idx[0] == 1:
                                axes_[fig_idx][sub_idx].set_xlabel(_X_INTER_TITLE, **kwargs_)
                            if sub_idx[1] == 0:
                                axes_[fig_idx][sub_idx].set_ylabel(_Y_INTER_TITLE, **kwargs_)
                            axes_[fig_idx][sub_idx].axvline(x=0, linestyle='--', color='grey')
                            axes_[fig_idx][sub_idx].axhline(y=0, linestyle='--', color='grey')
                            axes_[fig_idx][sub_idx].plot(x_line, x_line, '--', color='grey')
                            if separate_by_J:
                                axes_[fig_idx][sub_idx].set_title(subplot_label[t]+
                                                           r"   $R_{JT}=$"+f"{R_total_by_Jt[J][t]:6.4f}",
                                                           **kwargs_)
                                figures_[fig_idx].suptitle(f"{INTER_2} vs {INTER_1} (J={J})\n"
                                                           r"$R_{J}=$"+f"{R_total_by_J[J]:6.4f}", 
                                                           **kwargs_)
                            else:
                                axes_[fig_idx][sub_idx].set_title(subplot_label[t]+
                                                           r"   $R_{T}=$"+f"{R_total_by_t[t]:6.4f}", 
                                                           **kwargs_)
                                figures_[fig_idx].suptitle(f"{INTER_2} vs {INTER_1} (All J)\n"
                                                           r"$R=$"+f"{R_total:6.4f}",
                                                           **kwargs_)
                        else:
                            axes_[fig_idx].axvline(x=0, linestyle='--', color='grey')
                            axes_[fig_idx].axhline(y=0, linestyle='--', color='grey')
                            # axes_[fig_idx].plot(x_line, y_line, 'k--')
                            axes_[fig_idx].plot(x_line, x_line, '--', color='grey')
                            # axes_[fig_idx].set_title()
                            axes_[fig_idx].set_xlabel(_X_INTER_TITLE, **kwargs_)
                            axes_[fig_idx].set_ylabel(_Y_INTER_TITLE, **kwargs_)
                            if separate_by_J:
                                figures_[fig_idx].suptitle(f"{INTER_2} vs {INTER_1}  (J={J})\n"
                                                           r"$R_{J}=$"+f"{R_total_by_J[J]:6.4f}", 
                                                           **kwargs_)
                            else:
                                figures_[fig_idx].suptitle(f"{INTER_2} vs {INTER_1}  (All isospin and J)\n"
                                                           r"$R=$"+f"{R_total:6.4f}"
                                                           # +r"   $\quad m_{T=0}=$"+f"{slope_total['total']:12.4f}"
                                                           +r"   $\quad m=$"+f"{slope_total['total']:12.4f}"
                                                           ,**kwargs_)
                                                           
                    if sub_idx:
                        axes_[fig_idx][sub_idx].scatter(x, y, color=_COLOR[t], **kwargs_com)
                    else:
                        axes_[fig_idx].scatter(x, y, color=_COLOR[t], **kwargs_com)
    
    
    for i in range(len(figures_)):
        if not separate_by_particles:
            #for t in ISOS_VALUES: axes_[fig_idx][subplot_idx[t]].legend() 
            axes_[i].legend( **kwargs_)
        plt.tight_layout()
        file_out = f"me{INTER_1}vs{INTER_2}"
        if separate_by_particles:
            file_out += '_byT'
        if separate_by_J:
            file_out += f'_J{i}'  ## J_min=0, J = i
        figures_[i].savefig(file_out+'.pdf')
    plt.show()
    
    
if __name__ == '__main__':
    
    filename_1 = '../savedHamilsBeq1/usdb_J_A16.2b'
    filename_2 = '../scripts/results/D1S_16O_Nsho2.2b'
    INTERACTIONS_ = ('USDB', 'D1S')
    filename_2 = '../scripts/results/B1_16O_Nsho2.2b'
    INTERACTIONS_ = ('USDB', 'B1')
    # filename_1 = '../scripts/results/D1S_t0_16O_Nsho2.2b'
    # filename_2 = '../scripts/results/D1S_dd_16O_Nsho2.2b'
    # INTERACTIONS_ = ('HAMIL', 'DD')
    
    print(" *** Importing Hamiltonian 1")
    fo = filename_1.split('/')[-1].replace('.2b', '_new.2b')
    hamil_1, sh_states_1, _aux = readMatrixElementsJScheme(filename_1, fo)
    __2BME_abcd_ORDER = deepcopy(_aux)
    print(" *** Importing Hamiltonian 2")
    fo = filename_2.split('/')[-1].replace('.2b', '_new.2b')
    hamil_2, sh_states_2, _aux = readMatrixElementsJScheme(filename_2, fo)
    assert sh_states_1 == sh_states_2, "Wrong states"
    
    filename_3 = '../scripts/results/D1S_16O_Nsho2.2b'
    # INTERACTIONS_ = ('D1S', 'B1', 'USDB')
    fo = filename_3.split('/')[-1].replace('.2b', '_new.2b')
    hamil_3, sh_states_3, _aux = readMatrixElementsJScheme(filename_3, fo)
    assert sh_states_1 == sh_states_3, "Wrong states"
    
    
    
    printMatrixElementsJ2b_1vs2(hamil_1, hamil_2, sh_states_1, INTERACTIONS_,
                                separate_by_particles=False, separate_by_J=False)
    # printMatrixElementsJ2b_1vs2(hamil_1, hamil_2, sh_states_1, INTERACTIONS_,
    #                             separate_by_particles=True, separate_by_J=False)
    # printMatrixElementsJ2b_1vs2(hamil_1, hamil_2, sh_states_1, INTERACTIONS_,
    #                             separate_by_particles=False, separate_by_J=True)
    # printMatrixElementsJ2b_1vs2(hamil_1, hamil_2, sh_states_1, INTERACTIONS_,
    #                             separate_by_particles=True, separate_by_J=True)
    
    # INTERACTIONS_ = ('D1S', 'USDB')
    # printTableMatrixElements2b(sh_states_1, INTERACTIONS_, 
    #                            hamil_2, hamil_1, hamil_is_JT=(1,1,1))
    # INTERACTIONS_ = ('HAMIL', 'DD', 'D1S')
    # printTableMatrixElements2b_2(sh_states_1, INTERACTIONS_, 
    #                            hamil_1, hamil_2, hamil_3, hamil_is_JT=(1,1,1))
    
