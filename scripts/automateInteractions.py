'''
Created on Apr 4, 2022

@author: Miguel
'''
## ensures parent folder importing both Windows/Linux
import os
import sys

sys.path.insert(1, os.path.realpath(os.path.pardir))

from helpers.Helpers import valenceSpacesDict_l_ge10, _LINE_2, Constants, _LINE_1
import xml.etree.ElementTree as et
from helpers.TBME_Runner import TBME_Runner
from helpers.Enums import InputParts, AttributeArgs, ForceEnum,\
    Output_Parameters, SHO_Parameters, ValenceSpaceParameters,\
    CentralMEParameters, PotentialForms
from pathlib import Path
import os, shutil
import subprocess
from scripts.computingHOhbaromegaForD1S import generateCOMFileFromFile, \
    COM2_TRUNC, BASE_HAMIL_NAME, generateD1SHamil
from scripts.computingHOhbaromegaForD1S import TEMPLATE_INP_TAU

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
    <Core/>
    <Force_Parameters>
    </Force_Parameters>
</input>'''

root = et.fromstring(template_xml)
OUTPUT_XML   = 'AUX_INP.xml'


def _get_title(b_len, param):
    return f"TBME_Suite B_LEN={b_len:3.2f} PARAM={param} V0=1MeV"

def _mkdirResults(fold):
    Path('./results/HamilsJ/'+fold).mkdir(parents=True, exist_ok=True)
    return 'HamilsJ/'+fold

def _get_filename(inter, b_len, param, shells):
    if  shells == None:
        shells = '_MZ5'
    FOLD = ''
    BVAL = str(round(100*b_len)).zfill(3)
    if inter == ForceEnum.Central:
        FOLD = f'Gaussians/{BVAL}/'
        FOLD = _mkdirResults(FOLD)
        MVAL = str(round(100*param)).zfill(3)
        return f"{FOLD}gaussianV1mu{MVAL}_B{BVAL}{shells}"
    elif inter == ForceEnum.SpinOrbitShortRange:
        FOLD = f'LS_shortRange/{BVAL}/'
        FOLD = _mkdirResults(FOLD)
        return f"{FOLD}LSV1mu_B{BVAL}{shells}"
    elif inter == ForceEnum.Coulomb:
        FOLD = f'Coul/{BVAL}/'
        FOLD = _mkdirResults(FOLD)
        return f"{FOLD}Coulomb_B{BVAL}{shells}"
    elif inter == ForceEnum.Brink_Boeker:
        FOLD = f'BB/{BVAL}/'
        FOLD = _mkdirResults(FOLD)
        return f"{FOLD}BB_D1S_B{BVAL}{shells}"
    elif inter == ForceEnum.Kinetic_2Body:
        FOLD = f'Kin2B/{BVAL}/'
        FOLD = _mkdirResults(FOLD)
        return f"{FOLD}Kin2B_B{BVAL}{shells}"

def set_valenceSpace_Subelement(elem, *shells):
    
    for sh in shells:
        sp_states = valenceSpacesDict_l_ge10[sh]
        for qn in sp_states:
            _ = et.SubElement(elem, ValenceSpaceParameters.Q_Number, 
                              attrib={AttributeArgs.ValenceSpaceArgs.sp_state: qn})
            _.tail = '\n\t\t'
    
    return elem
    
def runInteractions(interactions, b_lengths, gaussian_lengths):
    title_ = root.find(InputParts.Interaction_Title)
    out_    = root.find(InputParts.Output_Parameters)
    fn_    = out_.find(Output_Parameters.Output_Filename)
    
    sho_ = root.find(InputParts.SHO_Parameters)
    hbo_ = sho_.find(SHO_Parameters.hbar_omega)
    b_  = sho_.find(SHO_Parameters.b_length)
    
    valenSp  = root.find(InputParts.Valence_Space)
    valenSp = set_valenceSpace_Subelement(valenSp,
                                          'S','P','SD','PF','SDG','PFH')
    forces = root.find(InputParts.Force_Parameters)
    
    tree = et.ElementTree(root)
    
    ## do KIN 2B (omits b steps, Taurus set 1/(b**2 A) ====================
    if ForceEnum.Kinetic_2Body in interactions:
        print(f" > doing Kinetic_ 2B m.e.")
        b_.text  = str(1.0)
        hbaromega = ((Constants.HBAR_C)**2) / Constants.M_MEAN
        hbo_.text = str(hbaromega)
        ## Title and FileName
        fn_.text = _get_filename(ForceEnum.Kinetic_2Body, 1.0, None,None)
        title_.set(AttributeArgs.name, _get_title(1.0, '(A=1)'))
        f  = et.SubElement(forces, ForceEnum.Kinetic_2Body, 
                           attrib={AttributeArgs.ForceArgs.active : 'True'})
        f.tail = '\n\t\t'
        
        tree.write(OUTPUT_XML)
        tbme_ = TBME_Runner(filename=OUTPUT_XML, verbose=False)
        tbme_.run()
        forces.remove(f)
            
    for b, b_len in enumerate(b_lengths):
        hbaromega = ((Constants.HBAR_C / b_len)**2) / Constants.M_MEAN
        print(f"* b_length = {b_len:3.2f} [fm]  [{b:2}/{len(b_lengths):2}]")
        
        b_.text  = str(b_len)
        hbo_.text = str(hbaromega)
        
        ## do LS ==============================================================
        if ForceEnum.SpinOrbitShortRange in interactions:
            print(f" > doing LS m.e.")
            ## Title and FileName
            fn_.text = _get_filename(ForceEnum.SpinOrbitShortRange, b_len, 
                                     None,None)
            title_.set(AttributeArgs.name, _get_title(b_len, '***'))
            f  = et.SubElement(forces, ForceEnum.SpinOrbitShortRange, 
                               attrib={AttributeArgs.ForceArgs.active : 'True'})
            _ = et.SubElement(f, CentralMEParameters.potential, 
                              attrib={AttributeArgs.name : PotentialForms.Power})
            _.tail = '\n\t\t'
            _ = et.SubElement(f, CentralMEParameters.constant, 
                              attrib={AttributeArgs.value : '1.0'})
            _.tail = '\n\t\t'
            _ = et.SubElement(f, CentralMEParameters.mu_length, 
                              attrib={AttributeArgs.value:  '1.0'})
            _.tail = '\n\t\t'
            _ = et.SubElement(f, CentralMEParameters.n_power, 
                              attrib={AttributeArgs.value : '0'})
            _.tail = '\n\t\t'
            
            f.tail = '\n\t\t'
            
            tree.write(OUTPUT_XML)
            tbme_ = TBME_Runner(filename=OUTPUT_XML, verbose=False)
            tbme_.run()
            forces.remove(f)
            
        ## do Coulomb_ ========================================================
        if ForceEnum.Coulomb in interactions:
            print(f" > doing Coulomb m.e.")
            ## Title and FileName
            fn_.text = _get_filename(ForceEnum.Coulomb, b_len, None,None)
            title_.set(AttributeArgs.name, _get_title(b_len, '***'))
            f  = et.SubElement(forces, ForceEnum.Coulomb, 
                               attrib={AttributeArgs.ForceArgs.active : 'True'})
            f.tail = '\n\t\t'
            
            tree.write(OUTPUT_XML)
            tbme_ = TBME_Runner(filename=OUTPUT_XML, verbose=False)
            tbme_.run()
            forces.remove(f)
        
        ## do Gaussians_ ======================================================
        if ForceEnum.Central in interactions:
            print(" > Iteration gaussians")
            for i, G_len in enumerate(gaussian_lengths):
                print(f"  ** Gaussian {G_len:3.2f} fm [{i:2}/{len(gaussian_lengths):2}]")
                
                ## Title and FileName
                fn_.text = _get_filename(ForceEnum.Central, b_len, 
                                         G_len, None)
                title_.set(AttributeArgs.name, _get_title(b_len, G_len))
                f  = et.SubElement(forces, ForceEnum.Central, 
                                   attrib={AttributeArgs.ForceArgs.active : 'True'})
                _ = et.SubElement(f, CentralMEParameters.potential, 
                                  attrib={AttributeArgs.name : PotentialForms.Gaussian})
                _.tail = '\n\t\t'
                _ = et.SubElement(f, CentralMEParameters.constant, 
                                  attrib={AttributeArgs.value : '1.0'})
                _.tail = '\n\t\t'
                _ = et.SubElement(f, CentralMEParameters.mu_length, 
                                  attrib={AttributeArgs.value:  str(G_len)})
                _.tail = '\n\t\t'
                _ = et.SubElement(f, CentralMEParameters.n_power, 
                                  attrib={AttributeArgs.value : '0'})
                _.tail = '\n\t\t'
                
                tree.write(OUTPUT_XML)
                tbme_ = TBME_Runner(filename=OUTPUT_XML, verbose=False)
                tbme_.run()
                forces.remove(f)
        
        ## ===================================================================

def _runTaurusBaseSeedForHamil(zz, nn, seed, Mz, folder2dump):    
    
    b20_ = '0   0.000'
    text = TEMPLATE_INP_TAU.format(z=zz, n=nn, seed=seed, b20=b20_)
    
    with open('aux.INP', 'w+') as f:
        f.write(text)
    extension = f'z{zz}n{nn}'
    _e = subprocess.call('./taurus_vap.exe < aux.INP > aux.OUT', 
                         shell=True, timeout=43200) # 1/2 day timeout
    shutil.move(f'final_wf.bin', folder2dump+f"/final_{extension}.bin")
    shutil.move(f'aux.OUT', folder2dump+f"/aux_{extension}.OUT")
    

def run_D1S_Interaction():
    """ Function to produce a list of hamiltonians D1S using process from the
    hbar omega """
    # TODO: set parameters for the execution. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # MZmax = 4
    # nucleus = {
    #     (10, 16): 1.82, ( 8, 8) : 1.84, (8, 10) : 1.66, (8, 12) : 1.68, #(12, 12): 1.76, 
    #     (10, 10): 1.68, (10, 12): 1.78, (10, 14): 1.74, 
    #     # (12, 6): 1.88, (12, 8): 1.74, (12,10): 1.78, (12,14): 1.75, 
    #     # (12,12): 1.75, (12,16): 1.80, (12,18): 1.87, (12,20): 1.94, 
    #     # (12, 22): 1.94, (12, 24): 1.96, (12, 26): 1.92, (12, 28): 1.98, 
    #     (14, 12): 1.76, (14, 14): 1.68,
    # }
    MZmax = 5
    nucleus = {
        ( 6, 6) : 1.75, ( 6, 8) : 2.00,
        ( 8, 8) : 2.12, (8, 10) : 1.66, #(8, 12) : 1.68, (12, 12): 1.76, 
        #(10, 10): 1.68, (10, 12): 1.78, (10, 14): 1.74, (10, 16): 1.82, 
        # (12, 6): 1.88, (12, 8): 1.74, (12,10): 1.78, (12,14): 1.75, 
        # (12,12): 1.75, (12,16): 1.80, (12,18): 1.87, (12,20): 1.94, 
        # (12, 22): 1.94, (12, 24): 1.96, (12, 26): 1.92, (12, 28): 1.98, 
        # (14, 12): 1.76, 
        (14, 14): 1.62, (14, 16): 1.80,
        (16, 16): 1.82, (16, 18): 1.76,
        (18, 18): 1.80, (18, 20): 1.78,
        (20, 20): 1.76,
    }
    
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    iter_ = 0
    HAMIL_FOLDER = 'hamil_folder' 
    Path('./'+HAMIL_FOLDER).mkdir(parents=True, exist_ok=True)
    
    for zn, b_len in nucleus.items():
        zz, nn = zn
        iter_ += 1
        
        calc_title = f"D1S on Z,N=({zz},{nn})  MZmax={MZmax}"
        hamil_name = f'D1S_t0_z{zz}n{nn}_MZ{MZmax}'
        global BASE_HAMIL_NAME    
        #BASE_HAMIL_NAME =  hamil_name   
        print(" Details:", calc_title) # prompt details
        print()
        
        generateCOMFileFromFile(MZmax, 0, com_filename=f"{hamil_name}.com")
        # args = (COM2_TRUNC, TAURUS_EXE_FOLD+'\\'+BASE_HAMIL_NAME) #TODO: windows 
        # _e = subprocess.call('copy {} {}.com'.format(*args), shell=True)
        
        ## Only set valencespace once in the XML
        ## NOTE: this function and XML tree object must be imported and make global
        from scripts.computingHOhbaromegaForD1S import set_valenceSpace_Subelement, root
        global root
        valenSp = root.find(InputParts.Valence_Space)
        valenSp = set_valenceSpace_Subelement(valenSp, MZmax)
        
        ## Verify if there is a taurus_vap.exe        
        print(f"* b_length = {b_len:3.2f} [fm]  [{iter_:2}/{len(nucleus):2}]")
        
        _ = generateD1SHamil(MZmax, b_len, do_coulomb=True, do_LS=True)
        
        if os.path.exists('taurus_vap.exe') and os.path.exists('input_DD_PARAMS.txt'):
            _runTaurusBaseSeedForHamil(zz, nn, 5, MZmax, HAMIL_FOLDER)
        ## TODO: move resultant hamiltonians to folder with name identifying nucleus
        shutil.move(f'{BASE_HAMIL_NAME}.2b', HAMIL_FOLDER+f"/{hamil_name}.2b")
        shutil.move(f'{hamil_name}.com', HAMIL_FOLDER+f"/{hamil_name}.com")
        shutil.move(f'{BASE_HAMIL_NAME}.sho', HAMIL_FOLDER+f"/{hamil_name}.sho")
        
    

if __name__ == '__main__':
    print()
    print(_LINE_1+"ITERATING TBME_RUNNER for valence spaces and interactions:"+_LINE_2)
    #%$ First, iterate over b_lengths for the larger space
    b_lengths = [#0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
                 #2.0, 2.2, 
                 2.4]
    
    gaussian_lengths = [#0.5, 0.7, 0.9, 1.1, 1.3, 
                        1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
    
    # runInteractions((ForceEnum.Central,       #ForceEnum.Coulomb, 
    #                  #ForceEnum.SpinOrbitShortRange #,ForceEnum.Kinetic_2Body
    #                  ), 
    #                  b_lengths, gaussian_lengths)
    
    run_D1S_Interaction()
    
    print(" FINISHED all B Lengths")
    
    ## =======================================================================
    #%% Then, use the previous results to evaluate smaller valence spaces
    
    