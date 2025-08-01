'''
Created on 27 may 2025

@author: delafuente

Tests to reduce interactions to a valence space, using the rules of the full 
HO occupation, obtaining the Core-energy and the Single Particle Energies for the
core.

27-5-25 : using the B1 interaction
28-5-25 : D1S, using a SHO density if the 18O (test for the Llarena Thesis)

'''
import numpy as np
import os, shutil
import sys
from helpers.TBME_SpeedRunner import TBME_SpeedRunner
from helpers.io_manager import castAntoineFormat2Str
sys.path.insert(1, os.path.realpath(os.path.pardir))

import xml.etree.ElementTree as et
from helpers.Enums import InputParts, SHO_Parameters, Output_Parameters,\
    ForceEnum, AttributeArgs, ForceFromFileParameters, ValenceSpaceParameters,\
    CoreParameters, DensityDependentParameters
from scripts.tests_Argone import set_valenceSpace_Subelement
from helpers.Helpers import elementNameByZ, valenceSpacesDict_l_ge10_byM,\
    readAntoine, Constants
from helpers.TBME_Runner import TBME_Runner


__CORES_BY_MZMIN = {0: (0,0), 1: (2,2), 2:(8,8), 3:(20,20), }

def __set_up_input_xml(Z, A, MZMin, MZMax, interaction,
                       read_inter_from=None, exclude_DD=False, only_DD=False):
    """
    Prepare the input shells, b, term de Argone interactions.
    """
    global USE_COULOMB, USE_COM_2B, ORIGINAL_INTER
    use_coulomb = USE_COULOMB
    assert interaction in ('B1', 'D1S', 'READ'), "Use these options"
    global FILENAME_XML_INTER
    
    __fn = FILENAME_XML_INTER.split('/')[-1]
    shutil.copy(FILENAME_XML_INTER, '.')
    root = et.parse(__fn).getroot()
    
    original_inter = ORIGINAL_INTER #'B1' #'D1S' #
    INTER_NC = f'{interaction}.xml'
    fn_xml = INTER_NC
    global A0, Z0
    
    hbaromega = 45*A**(-1/3) - 25*A**(-2/3)
    b_len     = Constants.HBAR_C / np.sqrt(Constants.M_MEAN * hbaromega)
    print("b length   =", b_len)
    print("hbar omega =",hbaromega)
    
    title_ = root.find(InputParts.Interaction_Title)
    out_   = root.find(InputParts.Output_Parameters)
    fn_    = out_.find(Output_Parameters.Output_Filename)
    ht_    = out_.find(Output_Parameters.Hamil_Type)
    com_   = out_.find(Output_Parameters.COM_correction)
    
    sho_ = root.find(InputParts.SHO_Parameters)
    hbo_ = sho_.find(SHO_Parameters.hbar_omega)
    b_   = sho_.find(SHO_Parameters.b_length)
    a_   = sho_.find(SHO_Parameters.A_Mass)
    z_   = sho_.find(SHO_Parameters.Z)
    
    cor_ = root.find(InputParts.Core).find(CoreParameters.innert_core)
    
    
    valenSp  = root.find(InputParts.Valence_Space)
    for elem_ in valenSp.findall(ValenceSpaceParameters.Q_Number):
        valenSp.remove(elem_)
    valenSp  = set_valenceSpace_Subelement(valenSp, MZMin, MZMax)
    forces   = root.find(InputParts.Force_Parameters)
    
    tree = et.ElementTree(root)
    
    ## SHO params and Valence space
    b_.text  = str(b_len)
    hbo_.text = str(hbaromega)
    a_.text  = str(A)
    z_.text  = str(Z)
    
    com_.text = '1' if USE_COM_2B else '0'
    _val_str = "" if MZMax!=MZMin else f"_Nsho{MZMin}"
    
    if interaction != 'READ':
        fn_.text = f"{interaction}_A{A}".replace('.', '')
        title_.set(AttributeArgs.name,  
                   f" [{interaction}] B_LEN={b_len:3.2f}")
        ht_.text = '3'
        
        out_path_interaction = 'results/' + fn_.text
        f  = forces.find(ForceEnum.Coulomb)
        f.set(AttributeArgs.ForceArgs.active, str(use_coulomb))
        
        if (interaction == 'D1S') :
            if (not exclude_DD) or only_DD:
                f = forces.find(ForceEnum.Density_Dependent_From_File)
                f.set(AttributeArgs.ForceArgs.active, 'True')
                f_file = f.find(DensityDependentParameters.file)
                # f_file.set(AttributeArgs.name, 'initial_wf_1.txt')
                f_file.set(AttributeArgs.name, 'final_wf.txt')
                f_x0   = f.find(DensityDependentParameters.x0)
                f_x0.set(AttributeArgs.value, '1')
                # f_x0   = f.find(DensityDependentParameters.constant)
                # f_x0.set(AttributeArgs.value, '0')
                f_alph = f.find(DensityDependentParameters.alpha)
                f_alph.set(AttributeArgs.value, '0.333333333333')
                f_core = f.find(DensityDependentParameters.core)
                if f_core != None:
                    f.remove(f_core)
                
                if only_DD:
                    fn_.text = f"{interaction}_dd_A{A}{_val_str}".replace('.', '')
                    title_.set(AttributeArgs.name, f" [{interaction}] DD interaction only B_LEN={b_len:3.2f}")
                    ht_.text = '4'
                    out_path_interaction = 'results/' + fn_.text
                    
                    com_.text = '0'
                    f  = forces.find(ForceEnum.Brink_Boeker)
                    f.set(AttributeArgs.ForceArgs.active, 'False')
                    f  = forces.find(ForceEnum.Coulomb)
                    f.set(AttributeArgs.ForceArgs.active, 'False')
                    f  = forces.find(ForceEnum.SpinOrbitShortRange)
                    f.set(AttributeArgs.ForceArgs.active, 'False')
            else:
                fn_.text = f"{interaction}_t0_A{A}{_val_str}".replace('.', '')
                title_.set(AttributeArgs.name,  
                           f" [{interaction}] B_LEN={b_len:3.2f} Without DD term (t3=0)")
                out_path_interaction = 'results/' + fn_.text
    else:
        nstr = MZMin if MZMax==MZMin else f"{MZMin}to{MZMax}"
        fn_.text = f"{original_inter}_A{A}_Nsho{nstr}".replace('.', '')
        out_path_interaction = 'results/' + fn_.text
        
        ht_.text = '3'
        z_c, n_c = __CORES_BY_MZMIN[MZMin - 1]
        cor_.set(CoreParameters.protons,  str(z_c))
        cor_.set(CoreParameters.neutrons, str(n_c)) 
        
        title_.set(AttributeArgs.name,  
                   f"Reduced Hamil for_{original_inter} MZ={MZMax} B_LEN={b_len:3.2f}")
        
        f  = forces.find(ForceEnum.Brink_Boeker)
        f.set(AttributeArgs.ForceArgs.active, 'False')
        f  = forces.find(ForceEnum.Coulomb)
        f.set(AttributeArgs.ForceArgs.active, 'False')
        f  = forces.find(ForceEnum.SpinOrbitShortRange)
        f.set(AttributeArgs.ForceArgs.active, 'False')
        
        f  = forces.find(ForceEnum.Force_From_File)
        f.set(AttributeArgs.ForceArgs.active, 'True')
        f1 = f.find(ForceFromFileParameters.file)
        f1.set(AttributeArgs.name, read_inter_from)
        f2 = f.find(ForceFromFileParameters.options)
        f2.set(AttributeArgs.FileReader.constant,  '1.0')
        f2.set(AttributeArgs.FileReader.ignorelines, '1')
        f2.set(AttributeArgs.FileReader.l_ge_10,  'True')
        
        
    tree.write(fn_xml)
    return fn_xml, out_path_interaction

def __set_up_input_xml_D1Srea(Z, A, MZMin, MZMax, interaction,
                              read_inter_from=None):
    """
    :interaction  ( D1S, READ )
    """
    global USE_COULOMB, USE_COM_2B, ORIGINAL_INTER
    use_coulomb = USE_COULOMB
    global FILENAME_XML_INTER
    
    __fn = FILENAME_XML_INTER.split('/')[-1]
    shutil.copy(FILENAME_XML_INTER, '.')
    root = et.parse(__fn).getroot()
    
    original_inter = ORIGINAL_INTER #'B1' #'D1S' #
    INTER_NC = f'D1S_A{A}_rea.xml'
    fn_xml = INTER_NC
    global A0, Z0
    
    hbaromega = 45*A**(-1/3) - 25*A**(-2/3)
    b_len     = Constants.HBAR_C / np.sqrt(Constants.M_MEAN * hbaromega)
    print("b length   =", b_len)
    print("hbar omega =",hbaromega)
    
    title_ = root.find(InputParts.Interaction_Title)
    out_   = root.find(InputParts.Output_Parameters)
    fn_    = out_.find(Output_Parameters.Output_Filename)
    ht_    = out_.find(Output_Parameters.Hamil_Type)
    com_   = out_.find(Output_Parameters.COM_correction)
    
    sho_ = root.find(InputParts.SHO_Parameters)
    hbo_ = sho_.find(SHO_Parameters.hbar_omega)
    b_   = sho_.find(SHO_Parameters.b_length)
    a_   = sho_.find(SHO_Parameters.A_Mass)
    z_   = sho_.find(SHO_Parameters.Z)
    
    cor_ = root.find(InputParts.Core).find(CoreParameters.innert_core)
    
    
    valenSp  = root.find(InputParts.Valence_Space)
    for elem_ in valenSp.findall(ValenceSpaceParameters.Q_Number):
        valenSp.remove(elem_)
    valenSp  = set_valenceSpace_Subelement(valenSp, MZMin, MZMax)
    forces   = root.find(InputParts.Force_Parameters)
    
    tree = et.ElementTree(root)
    
    ## SHO params and Valence space
    b_.text  = str(b_len)
    hbo_.text= str(hbaromega)
    a_.text  = str(A)
    z_.text  = str(Z)
    
    
    com_.text = '1' if USE_COM_2B else '0'
    _val_str  = "" if MZMax!=MZMin else f"_Nsho{MZMin}"
    
    if MZMax == MZMin:
        nstr = MZMin
        fn_.text = f"D1S_rea_A{A}_Nsho{nstr}".replace('.', '')
        out_path_interaction = 'results/' + fn_.text
    else:
        nstr = f"{MZMin}to{MZMax}"
        fn_.text = f"D1S_rea_A{A}".replace('.', '')
        out_path_interaction = 'results/' + fn_.text
    
    if MZMax == MZMin:
        ht_.text = '3'
        z_c, n_c = __CORES_BY_MZMIN[MZMin - 1]
        cor_.set(CoreParameters.protons,  str(z_c))
        cor_.set(CoreParameters.neutrons, str(n_c)) 
    else:
        ht_.text = '3'
        cor_.set(CoreParameters.protons,  '0')
        cor_.set(CoreParameters.neutrons, '0') 
        
    title_.set(AttributeArgs.name,  
               f"Reduced D1S Hamil with Rearrangement MZ={MZMax} B_LEN={b_len:3.2f}")
    
    f  = forces.find(ForceEnum.Brink_Boeker)
    f.set(AttributeArgs.ForceArgs.active, 'True')
    f  = forces.find(ForceEnum.Coulomb)
    f.set(AttributeArgs.ForceArgs.active, str(use_coulomb))
    f  = forces.find(ForceEnum.SpinOrbitShortRange)
    f.set(AttributeArgs.ForceArgs.active, 'True')
    if MZMax == MZMin:
        ## For the interaction with core, import the interaction from [read_inter_from]
        f  = forces.find(ForceEnum.Brink_Boeker)
        f.set(AttributeArgs.ForceArgs.active, 'False')
        f  = forces.find(ForceEnum.Coulomb)
        f.set(AttributeArgs.ForceArgs.active, 'False')
        f  = forces.find(ForceEnum.SpinOrbitShortRange)
        f.set(AttributeArgs.ForceArgs.active, 'False')
    
    f  = forces.find(ForceEnum.Force_From_File)
    f.set(AttributeArgs.ForceArgs.active, 'True')
    f1 = f.find(ForceFromFileParameters.file)
    f1.set(AttributeArgs.name, read_inter_from)
    f2 = f.find(ForceFromFileParameters.options)
    f2.set(AttributeArgs.FileReader.constant,  '1.0')
    f2.set(AttributeArgs.FileReader.ignorelines, '1')
    f2.set(AttributeArgs.FileReader.l_ge_10,  'True')
    
    tree.write(fn_xml)
    return fn_xml, out_path_interaction

def create_slater_wf(Z, N, inter, MZmin, MZmax):
    wf_file = 'initial_wf_1.txt' if inter!='READ' else 'initial_wf_2.txt'
    if inter=='READ':
        Z = Z - __CORES_BY_MZMIN[MZmin][0]
        N = N - __CORES_BY_MZMIN[MZmin][1]
    
    sh_states = []
    sp_states_sort_order = {}
    ndim = 0
    for M in range(MZmin, MZmax+1):
        for x in valenceSpacesDict_l_ge10_byM[M]:
            sh_states.append(x)
            j = readAntoine(x, l_ge_10=True)[2]
            
            for i in range(0, (j + 1)//2):
                sp_states_sort_order[ndim + 2*i] = ndim + i
                sp_states_sort_order[ndim + 2*i + 1] = ndim + j - i
            ndim += j + 1
            
    ndim *= 2
    U, V = np.zeros((ndim, ndim)), np.zeros((ndim, ndim))
    for i in range(ndim):
        U[i, i] = 1
    for k in range(Z):
        i = sp_states_sort_order[k]
        U[i, i] = 0
        V[i, i] = 1
    for k in range(N):
        i = sp_states_sort_order[k]
        U[i+ndim//2, i+ndim//2] = 0
        V[i+ndim//2, i+ndim//2] = 1
    lines = [len(sh_states), ] + [i for i in sh_states] + [99999999999999999,]
    lines_uv = [[], []]
    for i in range(ndim):
        for j in range(ndim):
            for k in (0, 1):
                txt_ = U[j,i] if k==0 else V[j,i]
                if abs(txt_) < 1e-15:
                    txt_ = 0.0
                txt_ = f"{txt_: >17.15E}"
                lines_uv[k].append(txt_)
    lines = lines + lines_uv[0] + lines_uv[1]
    with open(wf_file, 'w+') as f:
        f.write('\n'.join([str(i) for i in lines]))
    
    ## Test UV -> kappa-rho
    aux = [None, None, None, None]
    aux[0] = np.matmul(np.transpose(U), U) + np.matmul(np.transpose(V), V)
    aux[1] = np.matmul(U, np.transpose(U)) + np.matmul(V, np.transpose(V))
    aux[2] = np.matmul(np.transpose(U), V) + np.matmul(np.transpose(V), U)
    aux[3] = np.matmul(U, np.transpose(V)) + np.matmul(V, np.transpose(U))
    
    for i in range(ndim):
        for j in range(ndim):
            for k in range(2):
                val = "{:15.15f}".format(aux[k][i,j])
                if abs(aux[k][i,j]) < 1.0e-14:
                    if i==j:
                        print("1. Diagonal matrix element non-zero, k,i,j,val=", k,i,j,val)
                elif abs(aux[k][i,j] - 1) < 1.0e-14:
                    if i!=j:
                        print("2.1 Diagonal matrix element != 1, k,i,j,val=", k,i,j, val)
                else:
                    print("2.2 Invalid value for the cond., k,i,j,val=", k,i,j, val)
    
    print("3. Orthogonal conditions to be zero, max/min values of equations:",
          aux[2].max(), aux[2].min(), aux[3].max(), aux[3].min())                
    rho = np.matmul(V, np.transpose(V))
    kap = np.matmul(V, np.transpose(U))
    
    print("Traces of rho, kappa=", np.trace(rho[:ndim//2, :ndim//2]), 
                                   np.trace(rho[ndim//2:, ndim//2:]), np.trace(kap))

def adjust_monopole_hamiltonian_to_core(A, hamil_nocore, hamil_sm, N_min, N_max,
                                        interaction=None,
                                        use_com_corr=False,
                                        export_other_hamils=True):
    """
    Adjust the energy of the 0-body term and the 1-body sp-energies
    from a 2-body hamiltonian
    
    if Interaction == D1S: regenerates the 2B without DD term for true DD evaluation
    """
    
    sh_states = []
    hamil_J   = {}
    hamil_nc_com = {}
    hbaromega = 0
    global A0, Z0
    
    ## Import the complete hamiltonian 2b
    # hamil_nocore = 'results/D1S_dd_16O'
    with open(hamil_nocore+'.sho') as f:
        data   = f.readlines()
        title_ = data[0]
        sh_states = data[2].strip().split()[1:]
        hbaromega = float(data[4].strip().split()[1])
    with open(hamil_nocore+'.2b') as f:
        data = f.readlines()[1:]
        
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
    if use_com_corr:
        com_fac  = hbaromega * ((Constants.M_PROTON + Constants.M_NEUTRON) / 
                                (Constants.HBAR_C**2))
        com_fac /= A
        
        with open(hamil_nocore+'.com', 'r') as f:
            data = f.readlines()[1:]
            
            for line in data:
                line = line.strip()
                if line.startswith('0 5 '):
                    aux = line.split()
                    sh_abcd = tuple(aux[2:6])
                    Jmin, Jmax = int(aux[6]), int(aux[7])
                    J = Jmin
                    
                    if not sh_abcd in hamil_J:
                        hamil_J[sh_abcd] = dict([(j, [0,]*6) for j in range(J, Jmax+1)])
                    continue
                else:
                    assert J <= Jmax, "invalid J"
                    j_vals = [float(x) for x in line.split()]
                    for t in range(6):
                        hamil_J[sh_abcd][J][t] += j_vals[t] * com_fac
                    J += 1
                
    shutil.copy(f'{hamil_sm}.01b', f'{hamil_sm}(before).01b')
    
    hamil_nc_0b = 0.0
    hamil_nc_1b = {}
    with open(hamil_nocore+'.01b', 'r') as f:
        data = f.readlines()[2:]
        for line in data:
            aux = line.strip().split()
            st1, st2 = aux[:2]
            ep, en   = float(aux[2]), float(aux[3])
            hamil_nc_1b[(st1, st2)] = [ep, en]
    
    sh_states_core = []
    sh_states_val  = []
    for N in range(0, N_max+1):
        if N < N_min:
            for s in valenceSpacesDict_l_ge10_byM[N]: sh_states_core.append(s)
        elif N > N_max:
            continue
        else:
            for s in valenceSpacesDict_l_ge10_byM[N]: sh_states_val.append(s)
    
    ## Calculate the core centroids
    ## The com correction should be done for the CORE mass 16 (A0), not the mass
    ## of the full nucleus (the core energy should remain constant for the shell,
    ## therefore, connect with the empty shell case 16O)
    
    final_core_energy = 0.0
    t_c, v_c = 0.0, 0.0
    print(" CORE calculation:")
    for i1, sh_c in enumerate(sh_states_core):
        t_j_p, t_j_n = hamil_nc_1b[(sh_c, sh_c)]
        j = readAntoine(sh_c, l_ge_10=True)[2]
        t_j_p *= j + 1
        t_j_n *= j + 1
        if use_com_corr:
            t_j_p *= (1 - 1/A0) # The 1-body com correction is applied
            t_j_n *= (1 - 1/A0)
        print(f" > {sh_c}  t energy = {t_j_p: >8.5f} {t_j_n: >8.5f}")
        t_c += (t_j_p + t_j_n)
        
        e_j_p, e_j_n, e_j_pn = 0, 0, 0
        for i2, sh_c2 in enumerate(sh_states_core):
            if i2 < i1: continue
            idx_tpl = (sh_c, sh_c2, sh_c, sh_c2)
            if not idx_tpl in hamil_J: idx_tpl = (sh_c2, sh_c, sh_c2, sh_c)
            print(" > >", sh_c2, idx_tpl)
            for J, j_vals in hamil_J[idx_tpl].items():
                fac_J = (2*J + 1)
                
                me_str = ' '.join([f"{x:+6.4f}" for x in j_vals])
                me_str = me_str.replace('+', ' ')
                
                pp = fac_J * j_vals[0]
                nn = fac_J * j_vals[5]
                pn = fac_J * ((1 + (i1!=i2)) * j_vals[1])
                
                e_j_p += pp
                e_j_n += nn
                e_j_pn+= pn
                print(f" > > > {J} {fac_J:5.3f} ::   {me_str}  += {pp: >8.5f}, {pn: >8.5f}, {nn: >8.5f}")
                v_c += pp + pn + nn
        final_core_energy += (t_j_p + t_j_n) + (e_j_p + e_j_pn + e_j_n)
        print(f" > > E_core_sh1 sum = {e_j_p: >8.5f}  {e_j_pn: >8.5f}  {e_j_n: >8.5f}\n")
    print(f" Core energy Total T[{t_c: >9.6f}] + V[{v_c: >9.6f}] = {final_core_energy: >9.6f}")
    
    ## Calculate the sp-energies
    ## The COM correction here affects to upper orbits, includes therefore the
    ## whole A mass for the sp-states to converge into the 16O values
    ##
    ## EDIT, as the USDB scaling do not affect to the sp-energies, I will set them
    ## constant for the shell nuclei: reset the following variable:
    A_sp = A0 # A #
    
    final_sp_energies = []
    for sh_i in sh_states_val:
        e_j_p, e_j_n = hamil_nc_1b[(sh_i, sh_i)]
        j = readAntoine(sh_i, l_ge_10=True)[-1]
        fac_j = 1 / (j + 1)
        if use_com_corr:
            e_j_p *= 1 - 1/A_sp
            e_j_n *= 1 - 1/A_sp
        
        iv = sh_states.index(sh_i)
        for sh_c in sh_states_core:
            ic = sh_states.index(sh_c)
            
            idx_tpl = (sh_c, sh_i) if iv >= ic else (sh_i, sh_c)
            e_j_p2, e_j_n2 = 0, 0
            for J, j_vals in hamil_J[(*idx_tpl, *idx_tpl)].items():
                fac_jJ = fac_j * (2*J + 1) 
                ## Between valence and core states, there is not a==b and therefore,
                ## all elements follow the rule <pppp> + <nnnn> + 2<pnpn> 
                #* (1 / (1 + int(sh_i==sh_c)))
                
                e_j_p2 += fac_jJ * (j_vals[0] + j_vals[4]) #
                e_j_n2 += fac_jJ * (j_vals[5] + j_vals[1]) #
                
            e_j_p += e_j_p2
            e_j_n += e_j_n2 
        final_sp_energies.append( [e_j_p, e_j_n] )
    
    ## export into the 01b file.
    lines = [title_[:-1],]
    with open(hamil_sm+'.01b', 'w+') as f:
        lines.append(final_core_energy)
        for i, sh_i in enumerate(sh_states_val):
            e_p, e_n = final_sp_energies[i]
            lines.append(f"{sh_i: >8} {sh_i: >8} {e_p:12.9f} {e_n:12.9f}")
        lines = [str(x) for x in lines]
        f.write('\n'.join(lines))
    lines_sho = []
    with open(hamil_sm+'.sho', 'r+') as f:
        lines_sho = f.readlines()
        core_ln = lines_sho[3].strip().split()
        Z, N = __CORES_BY_MZMIN[N_min]
        core_ln[0], core_ln[1] = str(Z), str(N)
        lines_sho[3] = "    "+" ".join(core_ln) + '\n'
    with open(hamil_sm+'.sho', 'w+') as f:
        f.write(''.join(lines_sho))
        
    if interaction=='D1S' and export_other_hamils:
        ## Export the BB+LS+Coul terms without the DD term
        inp_fn, out_path_1 = __set_up_input_xml(Z0, A0, N_SHO, N_SHO, 'D1S', 
                                                exclude_DD=True)
    
        runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
        runner_.run()
        
        shutil.copy(hamil_sm+'.01b', out_path_1+'.01b')
        
        ## Export the DD matrix elements (full space)
        inp_fn, out_path_1 = __set_up_input_xml(Z, A, 0, N_SHO, 'D1S', 
                                                only_DD=True)
        runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
        runner_.run()
        
        ## Export the DD matrix elements (full space)
        inp_fn, out_path_1 = __set_up_input_xml(Z, A, N_SHO, N_SHO, 'D1S', 
                                                only_DD=True)
        runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
        runner_.run()
    
    print(" [DONE] exported results into", hamil_sm+'.01b', "/ .sho")

def _convert_JTmatrixElements(filename_, A, adjust_to_mass=False):
    """
    Generates a Matrix Element from an Antoine SHO matrix element format
    hbar_w generated from A (full nucleus, core + valence particles)
    Core and density factor,
    
    Export the J scheme matrix elements, fixed reproducing Antoine mass adjustment
    as hamiltonian type 3: sho, com, 01b energies and .2b files
    """
    hbar_omega = 45*A**(-1/3) - 25*A**(-2/3)
    
    sh_states = []
    sh_energies = []
    coreZ, coreN, densmode, amass = 0, 0, 0, 0
    hamil_type = 1
    hamil_JT = {}
    Nsho = []
    with open(filename_, 'r') as f:
        data = f.readlines()
        
        ## base
        args = data[1].strip().split()
        hamil_type = int(args[0])
        _n = int(args[1])
        e_args = data[2].strip().split()
        for i in range(_n):
            n,l,j = readAntoine(args[2+i], l_ge_10=False)
            Nsho.append(2*n + l)
            sh_states.append(castAntoineFormat2Str((n,l,j), l_ge_10=True))
            sh_states[-1] = int(sh_states[-1])
            ## sp-energies
            sh_energies.append(float(e_args[i]))
        
        ## mass correction factors
        args = data[3].strip().split()
        idens, coreZ, coreN = [int(x) for x in args[:3]]
        amass = float(args[3])
        
        ## list of 2b matrix elements
        for line in data[4:]:
            line = line.strip()
            if line.startswith('0 1'):
                args = line.split()
                _, _, a, b, c, d, Jmin, Jmax = [int(x) for x in args]
                a, b, c, d = [readAntoine(x, l_ge_10=False) for x in (a, b, c, d)]
                sh_abcd = (castAntoineFormat2Str(x, l_ge_10=True) for x in (a, b, c, d))
                sh_abcd = tuple(sh_abcd)
                hamil_JT[sh_abcd] = {0: {}, 1: {}}
                for T in (0, 1):
                    for J in range(Jmin, Jmax+1):
                        hamil_JT[sh_abcd][T][J] = 0
                T = 0
            else:
                jt_vals = [float(x) for x in line.split()]
                for J in range(Jmin, Jmax+1):
                    hamil_JT[sh_abcd][T][J] = jt_vals[J-Jmin]
                T = 1
    
    N_min, N_max = min(Nsho), max(Nsho)
    assert  N_min == N_max , "Why the Shells do not match"
    Zcore, Ncore = __CORES_BY_MZMIN[N_min]
    A_core = Zcore + Ncore
    
    ## Notice the user that he could be over extending the limits for the A given
    assert A >= A_core, f"A[{A}] under the core level [{A_core}]"
    A_max_shell = sum(__CORES_BY_MZMIN[N_max+1])
    assert A <= A_max_shell, f"A[{A}] over the valence space particle dimension Shell[{N_max}]=[{A_max_shell}]"
    
    ## conver the hamiltonian
    scaling_1b = 1
    scaling_2b = 1
    if A != A_core:
        if idens in (1, 2):
            scaling_2b = ((A_core + 2) / (A))**amass
            if idens == 2:
                scaling_1b = scaling_2b
    
    hamil_J = {}
    for sh_abcd, jt_vals in hamil_JT.items():
        hamil_J[sh_abcd] = {}
        for J in jt_vals[0].keys():
            hamil_J[sh_abcd][J] = [0, 0, 0, 0, 0, 0]
            v_t0 = jt_vals[0][J] * scaling_2b
            v_t1 = jt_vals[1][J] * scaling_2b
            ## pppp/nnnn
            hamil_J[sh_abcd][J][0], hamil_J[sh_abcd][J][5]  = v_t1, v_t1
            ## pnpn/npnp
            A1 = ((1 + (sh_abcd[0]==sh_abcd[1])*((-1)**J) ) * 
                  (1 + (sh_abcd[2]==sh_abcd[3])*((-1)**J) )) 
            A0 = ((1 - (sh_abcd[0]==sh_abcd[1])*((-1)**J) ) * 
                  (1 - (sh_abcd[2]==sh_abcd[3])*((-1)**J) )) 
            aux  = ( np.sqrt(A1)*v_t1 + np.sqrt(A0)*v_t0 ) /2
            
            hamil_J[sh_abcd][J][1], hamil_J[sh_abcd][J][4] = aux, aux
            ## pnnp/nppn
            aux  = ( np.sqrt(A1)*v_t1 - np.sqrt(A0)*v_t0 ) /2
            
            hamil_J[sh_abcd][J][2], hamil_J[sh_abcd][J][3] = aux, aux
    
    sh_energies = [scaling_1b * x for x in sh_energies]
    
    file_out = f"results/usdb_J_A{A}"
    title    = f'JT-scheme hamiltonian from [{filename_}] transformed to J and adjusted by A[{A}] mass'
    with open(file_out+'.sho', 'w+') as f:
        lines = [
            title,
            '     3',
            f"    {len(sh_states)} "+" ".join([str(x) for x in sh_states]),
            f"    {idens} {Zcore} {Ncore} {amass}", # idens and amass are ignored for hamil_type/=1
            f"    2 {hbar_omega:17.15f}"]
        f.write('\n'.join(lines))
    
    with open(file_out+'.01b', 'w+') as f:
        lines = []
        for i, sh_ in enumerate(sh_states):
            e_sh = sh_energies[i]
            lines.append(f"{sh_: >6} {sh_: >6}   {e_sh: >12.9f}  {e_sh: >12.9f}")
        lines = [title, '0.0',  *lines]
        f.write('\n'.join(lines))
        
    with open(file_out+'.com', 'w+') as f:
        f.write(title+'\n')
    
    with open(file_out+'.2b', 'w+') as f:
        lines = [title,]
        for sh_abcd, j_vals in hamil_J.items():
            _js = list(j_vals.keys())
            
            lines.append(' 0 5 {} {} {} {} {} {}'.format(*sh_abcd, min(_js), max(_js)))
            for J in _js:
                aux = "    ".join([f"{x: >15.11f}" for x in j_vals[J]])
                lines.append("    "+aux)
        f.write('\n'.join(lines))

def _moveAFilesToFoldersInResults():
    """
    Copying files f
    """
    os.chdir('results')
    all_files = list(os.listdir())
    all_files = list(filter(lambda x: os.path.isfile(x), all_files))
    A_tail    = list(filter(lambda x: x.startswith('D1S_A') and 
                            (x.endswith('.sho') and not 'Nsho' in x), 
                            all_files))
    A_folders = [x.replace('D1S_A','').replace('.sho','') for x in A_tail]
    for A in A_folders:
        if not A.isdigit(): continue
        _fld = f'A{A}'
        if _fld in os.listdir():
            shutil.rmtree(_fld)
        os.mkdir(_fld)
        files2mv = []
        for f in all_files:
            if not _fld in f: continue
            files2mv.append(f)
        
        print(f"  ** Files for {_fld}")
        for f in files2mv:
            shutil.move(f, f'{_fld}/')
            print(f" moving [{f}] into [{_fld}]")
        

if __name__ == '__main__':
    
    # _convert_JTmatrixElements('usdb.sho', A, adjust_to_mass=False)
    
    #===========================================================================
    # clear the result folders into A-folders
    #===========================================================================
    # _moveAFilesToFoldersInResults()
    # 0/0
    #===========================================================================
    # D1S interaction - From Rearrangement matrix elements
    #===========================================================================
    # Interaction definition for no-core space
    N_SHO  = 3
    N_SHO_VS = 2
    Z, A   = 12, 24 # 12, 24 #
    Z0, A0 = 8, 16
    FILENAME_XML_INTER = '../docs/input_D1S.xml'
    USE_COULOMB = False
    USE_COM_2B  = True
    ORIGINAL_INTER = 'D1S'
    
    # for zi in range(0, 12+1, 2):
    #     Z, A   = 8+zi, 16+2*zi # 8, 16
    
    create_slater_wf(Z, A-Z, 'D1S', 0, N_SHO)
    shutil.copy('final_wf.txt', 'initial_wf_1.txt')
    
    # inp_fn, out_path_1 = __set_up_input_xml(Z, A, 0, N_SHO, 'D1S', exclude_DD=False)
    #
    # runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
    # runner_.run()
    
    ## no-core hamiltonian for EDF: D1S-t0 + Q_matrix elements
    inp_fn, out_path_1 = __set_up_input_xml_D1Srea(Z, A, 0, N_SHO, 'D1S', 
                                                   read_inter_from=f'Q_hamilJ_A{A}.2b')
    
    runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
    runner_.run()
    
    ## Truncate the Hamiltonian for a certain shell range
    
    inp_fn, out_path = __set_up_input_xml_D1Srea(Z, A, N_SHO_VS, N_SHO_VS, 'READ', 
                                                 read_inter_from=out_path_1+'.2b')
    
    runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
    runner_.run()
    
    create_slater_wf(Z, A-Z, 'READ', N_SHO, N_SHO)
    
    adjust_monopole_hamiltonian_to_core(A, out_path_1, out_path, N_SHO_VS, N_SHO_VS,
                                        interaction='D1S', use_com_corr=USE_COM_2B,
                                        export_other_hamils=False)
    0/0
    #===========================================================================
    # D1S interaction
    #===========================================================================
    ## Interaction definition for no-core space
    # for zi in range(4, 12+1, 2):
    #     N_SHO  = 3
    #     N_SHO_VS = 2
    #     Z, A   = 8+zi, 16+2*zi # 8, 16
    #     Z0, A0 = 8, 16  # 8, 16
    #     FILENAME_XML_INTER = '../docs/input_D1S.xml'
    #     USE_COULOMB = False
    #     USE_COM_2B  = True
    #     ORIGINAL_INTER = 'D1S'
    #
    #     create_slater_wf(Z, A-Z, 'D1S', 0, N_SHO)
    #     shutil.copy('final_wf.txt', 'initial_wf_1.txt')
    #
    #     inp_fn, out_path_1 = __set_up_input_xml(Z, A, 0, N_SHO, 'D1S', exclude_DD=False)
    #
    #     runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
    #     runner_.run()
    #
    #     ## no-core hamiltonian for EDF
    #     inp_fn, _ = __set_up_input_xml(Z, A, 0, N_SHO, 'D1S', exclude_DD=True)
    #
    #     runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
    #     runner_.run()
    #
    #     ## Truncate the Hamiltonian for a certain shell range
    #     inp_fn, out_path = __set_up_input_xml(Z, A, N_SHO, N_SHO, 'READ', 
    #                                           read_inter_from=out_path_1+'.2b')
    #
    #     runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=False)
    #     runner_.run()
    #
    #     create_slater_wf(Z, A-Z, 'READ', N_SHO, N_SHO)
    #
    #     adjust_monopole_hamiltonian_to_core(A, out_path_1, out_path, N_SHO_VS, N_SHO_VS,
    #                                         interaction='D1S', use_com_corr=USE_COM_2B)
    #

    
    #===========================================================================
    
    #===========================================================================
    # B1 interaction
    #===========================================================================
    ## Interaction definition for no-core space
    # N_SHO = 2
    # Z, A = 8, 16
    # Z0, A0 = 8, 16
    # FILENAME_XML_INTER = '../docs/input_B1.xml'
    # USE_COULOMB = True
    # USE_COM_2B  = True
    # ORIGINAL_INTER = 'B1'
    #
    # inp_fn, out_path_1 = __set_up_input_xml(Z, A, 0, N_SHO, 'B1')
    #
    # runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=True)
    # runner_.run()
    #
    # create_slater_wf(Z, A-Z, 'B1', 0, N_SHO)
    #
    # ## Truncate the Hamiltonian for a certain shell range
    # inp_fn, out_path = __set_up_input_xml(Z, A, N_SHO, N_SHO, 'READ',
    #                                       read_inter_from=out_path_1+'.2b')
    #
    # runner_ = TBME_SpeedRunner(filename=inp_fn, verbose=True)
    # runner_.run()
    #
    # create_slater_wf(Z, A-Z, 'READ', N_SHO, N_SHO)
    #
    # adjust_monopole_hamiltonian_to_core(A, out_path_1, out_path, N_SHO, N_SHO,
    #                                     interaction='B1', use_com_corr=True)
    #
    #===========================================================================
