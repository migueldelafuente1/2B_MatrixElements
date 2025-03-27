
import os, shutil
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))

import xml.etree.ElementTree as et

from matrix_elements.ArgonePotential import ElectromagneticAv18TermsInteraction_JScheme,\
    NucleonAv14TermsInteraction_JTScheme, NucleonAv18TermsInteraction_JTScheme
from helpers.TBME_Runner import TBME_Runner
from scripts.automateInteractions import OUTPUT_XML, template_xml
from helpers.Helpers import valenceSpacesDict_l_ge10_byM, ConstantsV18,\
    ORDER_GEN_LAGUERRE
from helpers.Enums import ValenceSpaceParameters, AttributeArgs, InputParts,\
    Output_Parameters, SHO_Parameters, ForceEnum

    
def set_valenceSpace_Subelement(elem, MZMin, MZMax):
    for sh in range(MZMin, MZMax +1):
        sp_states = valenceSpacesDict_l_ge10_byM[sh]
        for qn in sp_states:
            _ = et.SubElement(elem, ValenceSpaceParameters.Q_Number, 
                              attrib={AttributeArgs.ValenceSpaceArgs.sp_state: qn})
            _.tail = '\n\t\t'
    return elem


root = et.fromstring(template_xml)

def __set_up_input_xml(term, b_len, MZMin, MZMax, inters=[]):
    """
    Prepare the input shells, b, term de Argone interactions.
    """
    assert inters.__len__() > 0, "Insert an interaction"
    fn_xml = OUTPUT_XML
    
    title_ = root.find(InputParts.Interaction_Title)
    out_    = root.find(InputParts.Output_Parameters)
    fn_    = out_.find(Output_Parameters.Output_Filename)
    
    sho_ = root.find(InputParts.SHO_Parameters)
    hbo_ = sho_.find(SHO_Parameters.hbar_omega)
    b_  = sho_.find(SHO_Parameters.b_length)
    
    valenSp  = root.find(InputParts.Valence_Space)
    valenSp = set_valenceSpace_Subelement(valenSp, MZMin, MZMax)
    forces = root.find(InputParts.Force_Parameters)
    
    tree = et.ElementTree(root)
    
    ## SHO params and Valence space
    hbaromega = 0.5 * ((ConstantsV18.HBAR_C / b_len)**2 / 
                       (ConstantsV18.M_NEUTRON + ConstantsV18.M_PROTON))
    b_.text  = str(b_len)
    hbo_.text = str(hbaromega)
    
    
    ## Switch on the interaction.
    for inter in inters:
        if inter == ForceEnum.Argone18Electromagetic:
            fn_.text = f"EMAV18_{term}_b{b_len:4.2f}".replace('.', '')
            title_.set(AttributeArgs.name,  
                       f"Integral term [{inter}] ELECTRO [{term}] B_LEN={b_len:3.2f} V0=1MeV")
            f  = et.SubElement(forces, ForceEnum.Argone18Electromagetic, 
                               attrib={AttributeArgs.ForceArgs.active : 'True'})
        elif inter == ForceEnum.Argone14NuclearTerms:
            fn_.text = f"AV14_{term}_b{b_len:4.2f}".replace('.', '')
            title_.set(AttributeArgs.name,  
                       f"Integral term [{inter}] NUCLEAR [{term}] B_LEN={b_len:3.2f} V0=1MeV")
            f  = et.SubElement(forces, ForceEnum.Argone14NuclearTerms, 
                               attrib={AttributeArgs.ForceArgs.active : 'True'})
        elif inter == ForceEnum.Argone18NuclearTerms:
            fn_.text = f"AV18_{term}_b{b_len:4.2f}".replace('.', '')
            title_.set(AttributeArgs.name,  
                       f"Integral term [{inter}] NUCLEAR [{term}] B_LEN={b_len:3.2f} V0=1MeV")
            f  = et.SubElement(forces, ForceEnum.Argone18NuclearTerms, 
                               attrib={AttributeArgs.ForceArgs.active : 'True'})
        else:
            raise Exception(f" [STOP] Invalid interaction force [{inter}]")
        
        f.tail = '\n\t\t'
        
        tree.write(fn_xml)
        forces.remove(f)
            
    
    
    return fn_xml

def __exportTalmiIntegrals(force, talmiInts, b_len):
    """
    export, in the folder result/ the Talmi integrals
    """
    b_len = f"{b_len:3.2f}".replace('.', '')
    fn_   = f"results/talmiInt_{force}_b{b_len}.txt"
    
    txt_ = []
    p_max = 0
    for k, vals in talmiInts.items():
        txt_.append(f'{k: >6}   '+ '   '.join([f"{x:5.9f}" for x in vals]))
        p_max = max(len(vals), p_max)
    
    txt_ = f"TALMI INTEGRALS {force} p max=[{p_max-1}]:\n" + '\n'.join(txt_)
    with open(fn_, 'w+') as f:
        f.write(txt_)




if __name__ == '__main__':
    
    __TEST_CASE = 1
    global ORDER_GEN_LAGUERRE
    if __TEST_CASE == 1:
        ## Visualize the interaction without constants, seeing the radial and
        ## angular constants.
        
        Inter_, force = NucleonAv14TermsInteraction_JTScheme, ForceEnum.Argone14NuclearTerms
        
        b_len = 1.50
        MZMin = 0
        MZMax = 2
        ORDER_GEN_LAGUERRE = 350
        
        Inter_._SWITCH_OFF_CONSTANTS = False
        for term in Inter_._SWITCH_OFF_TERMS.keys():
            Inter_._SWITCH_OFF_TERMS[term] = True
        
        talmiInts = {}
        for term in ('lS2_NN_T2', ): #  Inter_._SWITCH_OFF_TERMS.keys(): # 
            print(f" Evaluating [{term}] ...")
            Inter_._SWITCH_OFF_TERMS[term] = False
            
            inp_fn = __set_up_input_xml(term, b_len, MZMin, MZMax, inters=[force,])
            
            runner_ = TBME_Runner(filename=inp_fn, verbose=True)
            runner_.run()
            
            Inter_._SWITCH_OFF_TERMS[term] = True
            
            talmiInts[term] = Inter_._talmi_integrals[term]
            
        __exportTalmiIntegrals(force, talmiInts, b_len)
        print(" All terms evaluated! Bye.")

    elif __TEST_CASE == 2:
        ## Visualize the interaction without constants, seeing the radial and
        ## angular constants.
        
        Inter_, force = ElectromagneticAv18TermsInteraction_JScheme, ForceEnum.Argone18Electromagetic
        
        b_len = 1.50
        MZMin = 4
        MZMax = 4
        ORDER_GEN_LAGUERRE = 350
        
        Inter_._SWITCH_OFF_CONSTANTS = True
        Inter_.USE_EXACT_VACUUM_POLARIZATION = True
        for term in Inter_._SWITCH_OFF_TERMS.keys():
            Inter_._SWITCH_OFF_TERMS[term] = True
        
        talmiInts = {}
        for term in Inter_._SWITCH_OFF_TERMS.keys():
            print(f" Evaluating [{term}] ...")
            Inter_._SWITCH_OFF_TERMS[term] = False
            
            inp_fn = __set_up_input_xml(term, b_len, MZMin, MZMax, inters=[force,])
            
            runner_ = TBME_Runner(filename=inp_fn, verbose=False)
            runner_.run()
            
            Inter_._SWITCH_OFF_TERMS[term] = True
            
            term2 = term if term!='s' else 'df'
            talmiInts[term] = Inter_._talmi_integrals[term2]
            
        __exportTalmiIntegrals(force, talmiInts, b_len)
        print(" All terms evaluated! Bye.")
