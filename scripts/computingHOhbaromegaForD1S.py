'''
Created on Oct 16, 2022

@author: Miguel

This module evaluate the D1S elements for different oscillator lengths for 
a certain major shell. The time of computation is reduced by importing b=1 fm 
for Coulomb, LS short range and COM 
  * LS (b) = 130 * LS(b=1) / b**5
  * Coul(b) = Coul(b=1) / b
  * COM are the same for the valence space (cp the hamil filename)


'''
## ensures parent folder importing both Windows/Linux
import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

from helpers.Helpers import  _LINE_2, Constants,\
    valenceSpacesDict_l_ge10_byM, prettyPrintDictionary,\
    getStatesAndOccupationUpToLastOccupied
import xml.etree.ElementTree as et
from helpers.Enums import InputParts, AttributeArgs, ForceEnum,\
    Output_Parameters, SHO_Parameters, ValenceSpaceParameters,\
    CentralMEParameters, PotentialForms, ForceFromFileParameters,\
    BrinkBoekerParameters, CoreParameters, DensityDependentParameters
from pathlib import Path
import subprocess
import time
import numpy as np
from copy import  deepcopy
from helpers.TBME_SpeedRunner import TBME_SpeedRunner
from xml.etree.ElementTree import ElementTree
from helpers.io_manager import zipFilesInFolder


template_xml = \
'''<input>
    <Interaction_Title name="TBME_Runner" details=""/>
    <Output_Parameters>
        <Output_Filename></Output_Filename>
        <Hamil_Type>4</Hamil_Type>
        <COM_correction>0</COM_correction>
    </Output_Parameters>
    <SHO_Parameters>
        <A_mass></A_mass>
        <Z></Z>
        <hbar_omega units='MeV'></hbar_omega>
        <b_length units='fm'></b_length>
    </SHO_Parameters>
    <Valence_Space l_great_than_10='True'>
    </Valence_Space>
    <Core></Core>
    <Force_Parameters>
    </Force_Parameters>
</input>'''
## TODO fix the l_great_than_10='False' and <Hamil_Type>1</Hamil_Type>


_BASE_TEMPLATE_INP_TAU = \
'''Interaction
-----------
Master name hamil. files      hamil
Center-of-mass correction     1
Read reduced hamiltonian      0
No. of MPI proc per H team    0

Particle Number
---------------
Number of active protons      {z}.00
Number of active neutrons     {n}.00
No. of gauge angles protons   1
No. of gauge angles neutrons  1

Wave Function
-------------
Type of seed wave function    {seed}
Number of QP to block         0
No symmetry simplifications   0
Seed random number generation 0
Read/write wf file as text    0
Cutoff occupied s.-p. states  0.00E-00
Include all empty sp states   0
Spatial one-body density      1
Discretization for x/r        100 0.075
Discretization for y/theta    0   0.00
Discretization for z/phi      0   0.00

Iterative Procedure
-------------------
Maximum no. of iterations     700
Step intermediate wf writing  1
More intermediate printing    0
Type of gradient              1
Parameter eta for gradient    0.050E-01
Parameter mu  for gradient    0.200E-01
Tolerance for gradient        0.007E-00

Constraints
-----------
Force constraint N/Z          1
Constraint beta_lm            1
Pair coupling scheme          1'''

_CONSTR_NOCORE = '''
Tolerance for constraints     1.000E-08
Constraint multipole Q10      1   0.000
Constraint multipole Q11      1   0.000
Constraint multipole Q20      {b20}
Constraint multipole Q21      1   0.000
Constraint multipole Q22      1   0.000
Constraint multipole Q30      0   0.000
Constraint multipole Q31      1   0.000
Constraint multipole Q32      1   0.000
Constraint multipole Q33      1   0.000
Constraint multipole Q40      0   0.000
Constraint multipole Q41      1   0.000
Constraint multipole Q42      1   0.000
Constraint multipole Q43      1   0.000
Constraint multipole Q44      1   0.000
Constraint radius sqrt(r^2)   0   0.000
Constraint ang. mom. Jx       0   0.000
Constraint ang. mom. Jy       0   0.000
Constraint ang. mom. Jz       0   0.000
Constraint pair P_T00_J10     1   0.000
Constraint pair P_T00_J1m1    1   0.000
Constraint pair P_T00_J1p1    1   0.000
Constraint pair P_T10_J00     1   0.000
Constraint pair P_T1m1_J00    0   0.000
Constraint pair P_T1p1_J00    0   0.000
Constraint field Delta        0   0.000'''

_CONSTR_VS_WITH_CORE = '''
Tolerance for constraints     1.000E-08
Constraint multipole Q10      0   0.000
Constraint multipole Q11      0   0.000
Constraint multipole Q20      0   0.000
Constraint multipole Q21      1   0.000
Constraint multipole Q22      0   0.000
Constraint multipole Q30      0   0.000
Constraint multipole Q31      0   0.000
Constraint multipole Q32      0   0.000
Constraint multipole Q33      0   0.000
Constraint multipole Q40      0   0.000
Constraint multipole Q41      1   0.000
Constraint multipole Q42      1   0.000
Constraint multipole Q43      1   0.000
Constraint multipole Q44      1   0.000
Constraint radius sqrt(r^2)   0   0.000
Constraint ang. mom. Jx       0   0.000
Constraint ang. mom. Jy       0   0.000
Constraint ang. mom. Jz       0   0.000
Constraint pair P_T00_J10     0   0.000
Constraint pair P_T00_J1m1    0   0.000
Constraint pair P_T00_J1p1    0   0.000
Constraint pair P_T10_J00     0   0.000
Constraint pair P_T1m1_J00    0   0.000
Constraint pair P_T1p1_J00    0   0.000
Constraint field Delta        0   0.000'''

TEMPLATE_INP_TAU = _BASE_TEMPLATE_INP_TAU + _CONSTR_NOCORE
TEMPLATE_INP_TAU_CORE = _BASE_TEMPLATE_INP_TAU + _CONSTR_VS_WITH_CORE

root = et.fromstring(template_xml)
OUTPUT_XML   = 'AUX_INP.xml'

HAMIL_FOLDER = '../savedHamilsBeq1/'
PATH_LSSR = HAMIL_FOLDER + 'LSSR_MZ7_beq1.2b'
PATH_COUL = HAMIL_FOLDER + 'Coul_MZ7_beq1.2b'
PATH_COM2 = HAMIL_FOLDER + 'COM_MZ10.com'

COM2_TRUNC = None # set after COM2 definition (raises error if not done)

BASE_HAMIL_NAME = 'hamil'
TAURUS_EXE_FOLD = 'taurusExe'
TAURUS_INPUT    = TAURUS_EXE_FOLD + '/input_taurus.INP'
TAURUS_OUTPUT   = TAURUS_EXE_FOLD + '/output_taurus.OUT'

SPATIAL_DENS    = 'spatial_density_R.dat'
EXPORT_TBME     = 'hamilD1SnoDDbyB'
BU_FOLDER       = 'BU_outputs'
SUMMARY_FILE    = TAURUS_EXE_FOLD + '/summary_results.txt'

if os.path.exists(SUMMARY_FILE):
    with open(SUMMARY_FILE, 'w') as f:
        f.write('')

Path('./'+TAURUS_EXE_FOLD).mkdir(parents=True, exist_ok=True)
Path('./'+TAURUS_EXE_FOLD+'/'+EXPORT_TBME).mkdir(parents=True, exist_ok=True)
Path('./'+TAURUS_EXE_FOLD+'/'+BU_FOLDER).mkdir(parents=True, exist_ok=True)
EXPORT_TBME = TAURUS_EXE_FOLD+'/'+EXPORT_TBME
BU_FOLDER = TAURUS_EXE_FOLD+'/'+BU_FOLDER 

def _get_title(b_len, Mzmax, Mzmin=0):
    return f"TBME_Suite B_LEN={b_len:3.2f}fm D1S *** MZmax={Mzmax} /min: {Mzmin}"

def _mkdirResults(new_fold):
    path_ = '/'.join(['.', EXPORT_TBME, new_fold])
    Path(path_).mkdir(parents=True, exist_ok=True)
    return path_

def set_valenceSpace_Subelement(elem, Mzmax, Mzmin=0):
    ## TODO: MUY URGENTE, replanteate cambiar esta mierda para distingir entre
    ## hamiltonians no core (por ejemplo haz otro metodo)
    ## Urge en todo caso modificar l_GE_10 en caso de que sigas con el hamilt tipo 1
    ## en VS elem
    
    sp_energies = {}
    if Mzmin == 3:
        sp_energies = {'307': '0.0', '305': '6.5', '1103': '2.0', '1101': '4.0'}
    elif Mzmin == 2:
        sp_energies = {'205 ':'-3.92570', '1001': '-3.2079', '203': '2.1117'}

    elem.tail = '\n\t\t'
    for sh in range(Mzmin, Mzmax+1):
        if (Mzmax == Mzmin):
            #
            if (Mzmin < 4): 
                states_ = {2  : ('205', '1001', '203'),
                           3  : ('307','1103', '305', '1101')}
                sp_states = states_[sh]
        else:
            sp_states = valenceSpacesDict_l_ge10_byM[sh]
        for qn in sp_states:
            e_ = sp_energies.get(qn, '')
            _ = et.SubElement(elem, ValenceSpaceParameters.Q_Number, 
                              attrib={AttributeArgs.ValenceSpaceArgs.sp_state: qn,
                                      AttributeArgs.ValenceSpaceArgs.sp_energy: e_})
            _.tail = '\n\t\t'
    
    return elem

def generateCOMFileFromFile(MZmax, Mzmin, com_filename=None):
    """ 
    Import all states up to MZmax and then filter the results from a file 
    (WARNING, the com file must be in format qqnn with l_ge_10)
    """
    if MZmax > 10:
        raise Exception("There is no COM file larger than 10 and TBME_Runner won't calculate it. Bye.")
    valid_states_ = []
    for M in range(Mzmin, MZmax+1):
        valid_states_ = valid_states_ + list(valenceSpacesDict_l_ge10_byM[M])
    
    with open(PATH_COM2, 'r') as f:
        data = f.readlines()
        
    skip_block = False
    final_com  = [f'Truncated MZ={MZmax} From_ '+data[0], ]
    
    for line in data[1:]:
        l_aux = line.strip()
        header = l_aux.startswith('0 5 ')
        
        if header:
            t0,t1,a,b,c,d, j0,j1 = l_aux.split()
            skip_block = False
            for qn in (a, b, c, d): 
                qn = '001' if qn == '1' else qn 
                
                if qn not in valid_states_:
                    skip_block = True
                    break
            
            if not skip_block:
                final_com.append(line)
            continue
        
        if skip_block: continue
        
        final_com.append(line)
    
    if com_filename == None:  
        com_filename = 'aux_com2_{}.com'.format(MZmax)
    com_text = ''.join(final_com)[:-2]  # omit the last jump /n
    with open(com_filename, 'w+') as f:
        f.write(com_text)
    global COM2_TRUNC
    COM2_TRUNC = com_filename
    
    

def _setForcesOnElement(forces, b_length, do_coulomb=True, do_LS=True, 
                        inner_core=None):
    ## D1S PARAMS:
    W_ls = 130.0
    
    muGL = dict( part_1='0.7',      part_2='1.2',       units='fm')
    Wign = dict( part_1='-1720.3',  part_2='103.639',   units='MeV')
    Bart = dict( part_1='1300',     part_2='-163.483',  units='MeV')
    Heis = dict( part_1='-1813.53', part_2='162.812',   units='MeV')
    Majo = dict( part_1='1397.6',   part_2='-223.934',  units='MeV')
    
    t3_  = dict( value='1390.6',    units='MeV*fm^-4')
    alp_ = dict( value='0.333333')
    x0_  = dict( value='1') 
    _TT = '\n\t\t'
    ## ************************************************************************     
    ls_const = W_ls / (b_length**5)
    if do_LS: print(f" > doing LS m.e.")
    f1  = et.SubElement(forces, ForceEnum.Force_From_File,
                       attrib={AttributeArgs.ForceArgs.active : str(do_LS)})
    f1.text = _TT
    _ = et.SubElement(f1, ForceFromFileParameters.file, 
                      attrib={AttributeArgs.name : PATH_LSSR})
    _.tail=_TT
    _ = et.SubElement(f1, ForceFromFileParameters.options,
                      attrib={AttributeArgs.FileReader.ignorelines : '1',
                              AttributeArgs.FileReader.constant: str(ls_const),
                              AttributeArgs.FileReader.l_ge_10: 'True'})
    _.tail='\n\t'
    f1.tail = '\n\t'
    
    cou_const = 1 / b_length  # e^2 were in the interaction constant
    print(f" > doing Coul m.e.")
    f2  = et.SubElement(forces, ForceEnum.Force_From_File,  
                        attrib={AttributeArgs.ForceArgs.active : str(do_coulomb)})
    f2.text = _TT
    _ = et.SubElement(f2, ForceFromFileParameters.file, 
                      attrib={AttributeArgs.name : PATH_COUL})
    _.tail=_TT
    _ = et.SubElement(f2, ForceFromFileParameters.options,
                      attrib={AttributeArgs.FileReader.ignorelines : '1',
                              AttributeArgs.FileReader.constant: str(cou_const),
                              AttributeArgs.FileReader.l_ge_10: 'True'})
    _.tail='\n\t'
    f2.tail = '\n\t'
    
    print(f" > doing BB m.e.")
    f3  = et.SubElement(forces, ForceEnum.Brink_Boeker, 
                        attrib={AttributeArgs.ForceArgs.active : 'True'})
    f3.text = _TT
    _ = et.SubElement(f3, BrinkBoekerParameters.mu_length, attrib= muGL)
    _.tail = _TT
    _ = et.SubElement(f3, BrinkBoekerParameters.Wigner,    attrib= Wign)
    _.tail = _TT
    _ = et.SubElement(f3, BrinkBoekerParameters.Bartlett,  attrib= Bart)
    _.tail = _TT
    _ = et.SubElement(f3, BrinkBoekerParameters.Heisenberg,attrib= Heis)
    _.tail = _TT
    _ = et.SubElement(f3, BrinkBoekerParameters.Majorana,  attrib= Majo)
    _.tail = '\n\t'
    f3.tail = '\n\t'
    
    print(f" > doing DD m.e. if core. Core=", inner_core)
    do_DD = bool(inner_core!=None)
    f4 = et.SubElement(forces, ForceEnum.Density_Dependent,
            attrib={AttributeArgs.ForceArgs.active : str(do_DD)})
    f4.text = _TT
    _ = et.SubElement(f4, DensityDependentParameters.constant, attrib = t3_ )
    _.tail = _TT
    _ = et.SubElement(f4, DensityDependentParameters.alpha,    attrib = alp_)
    _.tail = _TT
    _ = et.SubElement(f4, DensityDependentParameters.x0,       attrib = x0_ )
    if isinstance(inner_core, tuple):
        _.tail = _TT
        core_ = {AttributeArgs.CoreArgs.protons:  str(inner_core[0]),
                 AttributeArgs.CoreArgs.neutrons: str(inner_core[1])}
        _ = et.SubElement(f4, DensityDependentParameters.core, attrib = core_ )
    _.tail = '\n\t'
    f4.tail = '\n\t'
    return forces, (f1, f2, f3, f4)


def _getCore(core_elem, MZmin, core=None):
    CORES_byMZ = {1: 2, 2: 8, 3: 20}
    inner_core = None
    if core or MZmin != 0:
        core_elem.tail = '\n'
        p = et.SubElement(core_elem, CoreParameters.protons)
        n = et.SubElement(core_elem, CoreParameters.neutrons)
        if core:
            p.text = str(core[0])
            n.text = str(core[1])
            inner_core = core
        else:
            if MZmin > 3:
                print("[Warning] Core for HO major not specified, ommiting DD.")
                return core_elem, None
            core_ = CORES_byMZ.get(MZmin, 0)
            p.text = str(core_)
            n.text = str(core_)
            inner_core = (core_, core_)
        p.tail = '\n\t\t'
        n.tail = '\n\t\t'
        core_elem.tail = '\n\t'
    return core_elem, inner_core

def generateD1SHamil(Mzmax, b_length, 
                     Mzmin=0, core=None, do_coulomb=True, do_LS=True):
    """
    Generates the input file for the D1S interaction depending on a b_length
    
    :return output_filename for the .2b, .sho matrix elements to be stored
    """    
    global BASE_HAMIL_NAME
    global OUTPUT_XML
    global COM2_TRUNC
    global root
    ## MODIFY ELEM.
    title_ = root.find(InputParts.Interaction_Title)
    out_   = root.find(InputParts.Output_Parameters)
    outfn_ = out_.find(Output_Parameters.Output_Filename)
    htype_ = out_.find(Output_Parameters.Hamil_Type)
    core_  = root.find(InputParts.Core)
    
    sho_ = root.find(InputParts.SHO_Parameters)
    hbo_ = sho_.find(SHO_Parameters.hbar_omega)
    b_   = sho_.find(SHO_Parameters.b_length)
    
    forces = root.find(InputParts.Force_Parameters)
    
    tree = et.ElementTree(root)
    
    #%% Setting of interactions and SHO values ********************************
    hbaromega = ((Constants.HBAR_C / b_length)**2) / Constants.M_MEAN
    
    b_.text  = str(b_length)
    hbo_.text = str(hbaromega)    
    core_, inner_core = _getCore(core_, Mzmin)
    
    forces, f_subelems = _setForcesOnElement(forces, b_length, 
                                             do_coulomb, do_LS,  inner_core)
    if Mzmin == 0:
        htype_.text = '4'
        hamil_filename = f"D1s_B{b_length:05.3f}_Mz{Mzmax}".replace('.', '_')
    else:
        htype_.text = '3'
        tail = f"{Mzmin}-{Mzmax}" if Mzmin != Mzmax else f"{Mzmax}"
        hamil_filename = f"D1s_B{b_length:05.3f}_ShellMz{tail}".replace('.', '_')
    outfn_.text    = BASE_HAMIL_NAME #hamil_filename 
    title_.set(AttributeArgs.name, _get_title(b_length, Mzmax, Mzmin))
        
    tree.write(OUTPUT_XML)
    
    ## RUN THE SPEEDRUNNER
    TBME_SpeedRunner.setFolderToSaveResults('.')
    tbme_ = TBME_SpeedRunner(filename=OUTPUT_XML, verbose=False)
    tbme_.run()
    
    ## Reset Forces for B,
    for f in f_subelems:
        forces.remove(f)
    
    ## copy the COM file
    with open(BASE_HAMIL_NAME+'.com', 'w+') as f: # #hamil_filename +'.com'
        with open(COM2_TRUNC, 'r') as f2:
            txt_ = f2.read()
        f.write(txt_)
    
    return hamil_filename
        

def _getProperties():
    aux = {
        'E_HFB': None,
        'Kin'  : None,
        'HF'   : None,
        'pair' : None,
        'parity': None,
        'b20'  : None,
        'b22'  : None,
        'r2'   : None,
        'Jz'   : None,
        'var_n' : None,
        'var_p' : None,
        'pc_t0' : None,
        'pc_t1' : None,}

    with open(TAURUS_OUTPUT, 'r') as f:
        data = f.readlines()
        for line in data:
            if   line.startswith('Parity         '):
                aux['parity']   = (float(line.split()[1]), )
            elif line.startswith('Number of protons '):
                aux['var_p']    = (float(line.split()[-1]), )
            elif line.startswith('Number of neutrons'):
                aux['var_n']    = (float(line.split()[-1]), )
            elif line.startswith('One-body'):
                aux['Kin']      = tuple(float(x) for x in line.split()[1:] )
            elif line.startswith(' ph part'):
                aux['HF']       = tuple(float(x) for x in line.split()[2:] )
            elif line.startswith(' pp part'):
                aux['pair']     = tuple(float(x) for x in line.split()[2:] )
            elif line.startswith('Full H'):
                aux['E_HFB']    = tuple(float(x) for x in line.split()[2:] )
            elif line.startswith('Beta_20'):
                aux['b20']      = tuple(float(x) for x in line.split()[1:] )
            elif line.startswith('Beta_22'):
                aux['b22']      = tuple(float(x) for x in line.split()[1:] )
            elif line.startswith('  r^2  '):
                aux['r2']       = tuple(float(x) for x in line.split()[1:] )
            elif line.startswith('  Z   '):
                aux['Jz']       = tuple(float(x) for x in line.split()[1:] )
            elif line.startswith('T = 0 ; J = 1 '):
                aux['pc_t0']    = tuple(float(x) for x in line.split()[-3:] )
            elif line.startswith('T = 1 ; J = 0 '):
                aux['pc_t1']    = tuple(float(x) for x in line.split()[-3:] )
    
        aux['var'] = aux['var_p'] + aux['var_n']
        del aux['var_n']
        del aux['var_p']
    with open(TAURUS_OUTPUT, 'r') as f:
        data = f.read()
        properly_finished = not 'Maximum number of iterations reached' in data
        
    return aux, properly_finished

def _extract_dictOfTuples(dict_vals):
    """ Aux method to extract the format of the dictionary"""
    dict_vals = dict_vals.strip()
    assert dict_vals.startswith('{'), " ! This is not a dictionary. STOP"
    assert dict_vals.endswith('}'),   " ! This is not a dictionary. STOP"
    dict_vals = dict_vals[1:-2] ## rm { and )}
    vals = dict_vals.split('),')
    out_dict = {}
    for line in vals:
        if ')' in line: 
            line = line.replace(')', '')
        # line = line.replace(': (')
        k, vv = line.split(': (')
        vv = vv.split(',')
        if vv[-1] == '': vv = vv[:-1]
        k = k.replace("'", '').strip()
        out_dict[k] = tuple(float(v) for v in vv)
    
    return out_dict

def plot_summaryResults(title_calc = '', export_figs=False):
    """ 
    Plots from a file SUMMARY file for matplotlib.
    """
    ## Group and extract fields  ***** 
    global SUMMARY_FILE, BU_FOLDER
    results_byB = []
    b_lengths   = []
    state_result = {}
    
    _ff = SUMMARY_FILE
    with open(_ff, 'r') as f:
        data = f.readlines()
        for line in data:
            b, properly_finished, dict_vals = line.split('##')
            
            properly_finished = properly_finished.lower()=='true'
            if not properly_finished:
                state_result[b] = properly_finished
                continue
            
            b_lengths.append(float(b))
            
            results_byB.append({})
            dict_vals = _extract_dictOfTuples(dict_vals) #json.loads(dict_vals)
            for k, vals_ in dict_vals.items():
                results_byB[-1][k] = tuple(float(v) for v in vals_)
    
    b_lengths   = np.array(b_lengths)
    hbar_omegas = 41.425 / np.power(b_lengths, 2)
    
    rms, pc_t0, parity = [],[],[]
    e_hfb = [[],[],[],[]]
    kin, pair, hf = deepcopy(e_hfb), deepcopy(e_hfb), deepcopy(e_hfb)
    pc_t1 = [[], [], []]
    for dict_vals in results_byB:
        for i in range(4):
            e_hfb[i].append(dict_vals['E_HFB'][i])
            pair [i].append(dict_vals['pair'][i])
            hf   [i].append(dict_vals['HF'][i])
            if i < 3:
                kin[i].  append(dict_vals['Kin'][i])
                pc_t1[i].append(dict_vals['pc_t1'][i])
        
        rms.  append(dict_vals['r2'][-1])
        pc_t0.append(dict_vals['pc_t0'][1])
        
        parity.append(dict_vals['parity'][-1])
    
    ### PLOTS  ************************
    # 
    ### *******************************
    import matplotlib.pyplot as plt
    plt.style.use('bmh')
    
    fig1, ax1 = plt.subplots() # hfb
    fig2, ax2 = plt.subplots() # hf
    fig3, ax3 = plt.subplots() # pa
    fig4, ax4 = plt.subplots() # kin
    for i, lab_ in enumerate(('pp', 'nn', 'pn', 'total')):
        marker = '.-' if i < 3 else '^-'
        ax1.plot(hbar_omegas, e_hfb[i], marker, label=lab_)
        ax2.plot(hbar_omegas, hf[i],    marker, label=lab_)
        ax3.plot(hbar_omegas, pair[i],  marker, label=lab_)
        
        if   i < 2: ax4.plot(hbar_omegas, kin[i], marker, label=lab_)
        elif i < 3: continue
        else:       ax4.plot(hbar_omegas, kin[i-1], marker, label=lab_)
    
    for ax in (ax1, ax2, ax3, ax4):
        axt = ax.twiny()
        axt.set_xlim(ax.get_xlim())
        axt.set_xticks(hbar_omegas)
        axt.set_xticklabels(b_lengths)
        # axt.set_xlim(ax.get_xlim())
        # axt.set_xticks(ax.get_xticks())
        # bb = [(41.425 / float(hhww))**.5 for hhww in ax.get_xticks()]
        # axt.set_xticklabels([f'{bbb:4.2f}' for bbb in bb])
    
    ax1.set_title("E HFB % "+title_calc)
    ax2.set_title("E HF  % "+title_calc)
    ax3.set_title("E pairing % "+title_calc)
    ax4.set_title("Kin  % "+title_calc)
    for i, ax_ in enumerate((ax1, ax2, ax3, ax4)):
        ax_.set_ylabel('MeV')
        ax_.set_xlabel('hbar omega (MeV)')
        ax_.legend() #if i == 3:
            #ax_.legend(frameon=False, loc='upper center', ncol=3)
        
        if export_figs:
            plt.tight_layout()
            txt_ = ax_.title.get_text().split('%')[0].strip().lower().replace(' ','-')
            ax_.figure.savefig("".join([BU_FOLDER, '/', txt_, ".pdf"]))
    
    fig, ax = plt.subplots()
    ax.set_title(title_calc+"\nPair Properties (Agnostic coupling)")
    ax.plot(hbar_omegas, pc_t0, '.-', label='pc_t00')
    ax.plot(hbar_omegas, pc_t1[0], 'v-', label='pc_t1 pp')
    ax.plot(hbar_omegas, pc_t1[1], 'o-', label='pc_t1 pn')
    ax.plot(hbar_omegas, pc_t1[2], '^-', label='pc_t1 nn')
    ax.set_xlabel('hbar omega (MeV)')
    plt.legend()
    plt.tight_layout()
    if export_figs:
        ax.figure.savefig(BU_FOLDER+'/pair-couplings'+".pdf")
    plt.show()

# _DEF_BASE = {(12,10): 0.4, (12,12):0.36, (12,14):0.3, (12,16):0.25}
_DEF_BASE = {( 8, 8): 0.0, ( 8,10): 0.0, ( 8,12): 0.0,
             (10,10): 0.3, (10,12):0.35, (10,14):0.25, (10,16):0.2,
             (14,12): 0.3, (14,12): 0.3,}
def _executeTaurus(zz, nn, outfile_tailtext, seed=3, reduced_vs_calc=True):
    """ 
    This is an extension to run Taurus from the exe_scripts by making the input
    
    It do not export DataTaurus Result
    """
    
    try:        
        ## ----- execution ----
        if not reduced_vs_calc:
            b20_ = '0   0.000'
            if seed == 3:
                b20_ = f'1   {_DEF_BASE[(zz,nn)]:5.3f}'
            text = TEMPLATE_INP_TAU.format(z=zz, n=nn, seed=seed, b20=b20_)
        else:
            text = TEMPLATE_INP_TAU_CORE.format(z=zz, n=nn, seed=seed)
        
        
        with open(TAURUS_INPUT, 'w+') as f:
            f.write(text)
               
        _e = subprocess.call('./{}/taurus_vap.exe < {} > {}'
                             .format(TAURUS_EXE_FOLD, TAURUS_INPUT, TAURUS_OUTPUT), 
                             shell=True, timeout=43200) # 1/2 day timeout
        
        properties, properly_finished = _getProperties()
        # return properties # TODO: Remove Windows
        
        # move shit to the folder
        ## TODO: change for LINUX
        details = '_{}_Z{}N{}'.format(outfile_tailtext,zz,nn)
        _e = subprocess.call('mv {} {}'.format(TAURUS_OUTPUT, 
                              BU_FOLDER+'/output'+details+'.OUT'),
                              shell=True, timeout=8640) # 2.4h timeout
        _e = subprocess.call('cp final_wf.bin {}/seed_{}.bin'
                             .format(BU_FOLDER, details), 
                             shell=True)
        if os.path.exists(SPATIAL_DENS):
            _e = subprocess.call('cp {} {}/{}{}.dat'
                                 .format(SPATIAL_DENS,
                                         BU_FOLDER, SPATIAL_DENS[:-4], details), 
                                 shell=True)
        
        # _e = subprocess.call('rm *.dat *.red *.txt', shell=True)
    
    except Exception as e:
        print("\n 1>> EXCEP (exe_q20pes_taurus._executeProgram) >>>>>>>  ")
        print(" 1>> current b20 =", outfile_tailtext)
        print(" 1>> OUTPUT FILE from TAURUS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if os.path.exists(TAURUS_OUTPUT):
            with open(TAURUS_OUTPUT, 'r') as f:
                text = f.read()
                print(text)
        else:
            print(" 1 ::: [WARNING] Cannot open output file !!")
        print(" 1<< OUTPUT FILE from TAURUS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(" 1> [",e.__class__.__name__,"]:", e, "<1")
        
        #print(res)
        print(" 1<< EXCEP (exe_q20pes_taurus._executeProgram) <<<<<<<  ")
        return None
    
    return properties, properly_finished   






################################################################################    
#      MAIN EXECUTIOM   
#
################################################################################



if __name__ == '__main__':
    
    print("ITERATING TBME_RUNNER for valence spaces and interactions:"+_LINE_2)
    #% First, iterate over b_lengths for the larger space
    
    # TODO: set parameters for the execution. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    MZmax = 4
    MZmin = 0
    # b_lengths = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.4, 2.6]
    b_lengths = [1.5 + (0.05*i) for i in range(22)]
    zz, nn = 12, 12
    # nucleus = [(2,nn) for nn in range(0,14,2)] + [(4,nn) for nn in range(0,14,2)]
    # nucleus = [(12,nn) for nn in range(10,17,2)]# 
    nucleus = [( 8, 8), ( 8,10), ( 8,12), (10,10), (10,12), (10,14), (10,16), 
               (14,14), (14,16) ]
    seed_base = 3
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    seed = seed_base # final seed will be reused to speed the minimiztaion
    for zz, nn in nucleus:
        SUMMARY_FILE =  TAURUS_EXE_FOLD + f'/summary_results_z{zz}n{nn}_MZ{MZmax}.txt'
        
        calculation_title = f"D1S on Z,N=({zz},{nn})  MZmax={MZmax}  seed_type:{seed}"
        if MZmin == 0:
            calculation_title = f"D1S on Z,N=({zz},{nn})  Shells={MZmin}:{MZmax} seed_type:{seed}"
        # if os.getcwd().startswith('C:'):
        #     SUMMARY_FILE = f'BU_outputs_54Cr_s{seed}/'+SUMMARY_FILE.split('/')[-1]
        #     BU_FOLDER    = f'BU_outputs_54Cr_s{seed}'
        #     plot_summaryResults(title_calc=calculation_title, export_figs=True)
        #     raise Exception("Stopping here, You're in windows")
        
        print(" Details:", calculation_title) # prompt details
        print(" bs to compute:", b_lengths)
        print()
        
        generateCOMFileFromFile(MZmax, MZmin)
        args = (COM2_TRUNC, TAURUS_EXE_FOLD+'/'+BASE_HAMIL_NAME)
        _e = subprocess.call('cp {} {}.com'.format(*args), shell=True)
        _e = subprocess.call('cp {} {}.com'.format(COM2_TRUNC, BASE_HAMIL_NAME), shell=True)
        # args = (COM2_TRUNC, TAURUS_EXE_FOLD+'\\'+BASE_HAMIL_NAME) #TODO: windows 
        # _e = subprocess.call('copy {} {}.com'.format(*args), shell=True)
        
        ## Only set valencespace once in the XML
        valenSp = root.find(InputParts.Valence_Space)
        valenSp = set_valenceSpace_Subelement(valenSp, MZmax, MZmin)
        
        ## Verify if there is a taurus_vap.exe
        assert os.path.exists('taurus_vap.exe'), "taurus_vap.exe required to run the script. STOP!"
        _e = subprocess.call(f'cp taurus_vap.exe {TAURUS_EXE_FOLD}', shell=True)
        
        summary = {}
        for b, b_len in enumerate(b_lengths[::-1]): ## start from the larger b
            print(f"* b_length = {b_len:3.2f} [fm]  [{b:2}/{len(b_lengths):2}]")
            
            hamil_filename = generateD1SHamil(MZmax, b_len, Mzmin=MZmin,
                                              do_coulomb=True, do_LS=True)
            
            print(" ... executing taurus %")
            _start =  time.time()
            seed = 1 if b>0 else seed_base ## start from 
            properties, prop_finished = _executeTaurus(zz, nn, hamil_filename, 
                                                       seed, MZmin==MZmax)
            _end =  time.time()
            print(" [DONE] executing taurus. TIME:", _end -_start,"\n RESULTS:")
            prettyPrintDictionary({'_data':properties})
            print()
            ## move the hamiltonian to its folder ()
            args = (BASE_HAMIL_NAME, EXPORT_TBME+'/'+hamil_filename)
            ## TODO: change for LINUX
            _e = subprocess.call('mv {}.sho {}.sho'.format(*args), shell=True)
            _e = subprocess.call('mv {}.2b {}.2b'  .format(*args), shell=True)
            
            # args = (BASE_HAMIL_NAME, EXPORT_TBME.replace('/', '\\')+'\\'+hamil_filename)
            # _e = subprocess.call('copy {}.sho {}.sho'.format(*args), shell=True)
            # _e = subprocess.call('copy {}.2b {}.2b'  .format(*args), shell=True)
            
            ## Export properties of interest:
            summary[b_len] = properties['E_HFB'][-1]
            with open(SUMMARY_FILE, 'a') as f:
                f.write("{:4.3f} ##{}## {}\n".format(b_len, prop_finished, properties))
            
            ## remove the hamil files to avoid confilcts
            _e = subprocess.call('rm *.dat *.red', shell=True)
            
            seed = seed_base
            if prop_finished:
                seed = 1
                _e = subprocess.call('cp final_wf.bin initial_wf.bin', shell=True)
            ## 
        print(summary)
        
        _order = f'cp -r {HAMIL_FOLDER} {BU_FOLDER}'
        _e = subprocess.call(_order, shell=True)
        zipFilesInFolder(BU_FOLDER, f"{TAURUS_EXE_FOLD}/BU_z{zz}n{nn}-Mz{MZmax}")
        plot_summaryResults(title_calc=calculation_title, export_figs=True)
        print(" FINISHED all B Lengths")
    
    ## =======================================================================
    #%% Then, use the previous results to evaluate smaller valence spaces
    
