'''
Created on Feb 23, 2021

@author: Miguel
'''
from helpers.Enums import InputParts, SHO_Parameters, ForceEnum,\
    AttributeArgs, ValenceSpaceParameters, Output_Parameters
from helpers.Enums import ForceVariablesDict
import xml.etree.ElementTree as et
from helpers.WaveFunctions import QN_1body_jj
from helpers.Helpers import readAntoine, shellSHO_Notation, valenceSpacesDict_l_ge10


def castAntoineFormat2Str(state, l_ge_10=False):
    """ 
    return a string with the state index in Antoine format. Check state.
    
    :l_ge_10 <bool> [default=False] format for l>10. Converts l<10 to this format
    """
    if isinstance(state, QN_1body_jj):
        if l_ge_10:
            return state.AntoineStrIndex_l_greatThan10
        return state.AntoineStrIndex
    
    if isinstance(state, int):
        if l_ge_10:
            state = readAntoine(state, l_ge_10)
            state = str(10000*state[0] + 100*state[1] + state[2])
        state = str(state)
    elif isinstance(state, str):
        if l_ge_10:
            state = readAntoine(state, l_ge_10)
            state = str(10000*state[0] + 100*state[1] + state[2])
    elif isinstance(state, (tuple, list)):
        if l_ge_10:
            state = str(10000*state[0] + 100*state[1] + state[2])
        else:
            state = str(1000*state[0] + 100*state[1] + state[2])
    else:
        raise Exception('invalid type for the state to read (onlu str, int or list)')
    
    if state == '1':
        return '001'
    return state

def valenceSpaceShellNames(valence_space, l_ge_10=False):
    """
    Join the SHO major shells defining the valence space (without sense care).
        Arg:
    :valence_space Quantum numbers array (Antoine format)
    :l_ge_10 <bool> [default=False] format for l>10.
    """
    _space = []
    states_accepted = 0
    for shell, qqnn_shell in valenceSpacesDict_l_ge10.items():
        
        aux = [(sp, shellSHO_Notation(*readAntoine(sp, True))) for sp in qqnn_shell]
        aux = [(sho_st , sp in valence_space) for sp, sho_st in aux]
        aux = dict(aux)
        if not False in aux.values():
            _space.append(shell)
            states_accepted += len(aux)
        elif True in aux.values():
            aux = dict(filter(lambda x: x[1], aux.items())).keys()
            _space.append('({}: {})'.format(shell, list(aux)))
            states_accepted += len(aux)
        
        if states_accepted == len(valence_space):
            break
    
    return list(set(_space))

class _Parser:
    
    class Type:
        json = 'json'
        xml  = 'xml'
    
    filetype = None
    
    def __init__(self, data):
        
        self._data     = None
        raise ParserException('abstract method, implement me!')
    
    def getInteractionTitle(self):
        raise ParserException('abstract method, implement me!')
    
    def getOutputArgs(self):
        raise ParserException('abstract method, implement me!')
    
    def getSHOArgs(self):
        raise ParserException('abstract method, implement me!')
    
    def getValenceSpaceArgs(self):
        raise ParserException('abstract method, implement me!')
    
    def getCore(self):
        raise ParserException('abstract method, implement me!')
    
    def getForceEnum(self):
        raise ParserException('abstract method, implement me!')
        
class ParserException(Exception):
    pass


class _JsonParser(_Parser):
    
    filetype = _Parser.Type.json
    
    def __init__(self, json_data):
        
        self._data = json_data
    
    def getInteractionTitle(self):
        vals = {AttributeArgs.name : self._data[InputParts.Interaction_Title],
                AttributeArgs.details : ''}
        return vals
    
    def getOutputArgs(self):
        # TODO: Not tested
        vals_dict = self._data.get(InputParts.Output_Parameters)
        
        for param in Output_Parameters.members():
            if param not in vals_dict:
                if param == Output_Parameters.Output_Filename:
                    raise ParserException("Missing filename in {} json dictionary"
                                          .format(InputParts.Output_Parameters))
                vals_dict[param] = None
                
            else:
                vals_dict[param] = str(vals_dict[param])
                    
        return vals_dict
    
    def getSHOArgs(self):
        vals_dict = self._data.get(InputParts.SHO_Parameters)
        mandatory = SHO_Parameters.members()
        keys_ = [key_ in mandatory for key_ in vals_dict.keys()]
        
        if False in keys_ or len(keys_)!=len(mandatory):
            missing = set(mandatory).symmetric_difference(set(vals_dict.keys()))
            raise ParserException("missing or wrong parameter/s: [{}] in {}"
                                  .format(missing, InputParts.SHO_Parameters))
                        
        return vals_dict
    
    def getValenceSpaceArgs(self):
        _data = self._data.get(InputParts.Valence_Space)
        qn_Sts  = _data[ValenceSpaceParameters.Q_Number]
        qn_Ens  = _data.get(ValenceSpaceParameters.QN_Energies)
        antoine_l_ge_10 = False # invalid in json format
        
        if (qn_Ens == None) or (qn_Ens == []):
            qn_Ens = ['0.0' for _ in range(len(qn_Sts))]
        elif len(qn_Ens) != len(qn_Sts):
            raise ParserException("Missing single particle energy, QN_Energies"
                                  "[{}] mismatch with the number of states given"
                                  "[{}]".format(len(qn_Ens), len(qn_Sts)))
        
        vals_dict = {}
        for i in range(len(qn_Sts)):
            vals_dict[str(qn_Sts[i])] = str(qn_Ens[i])  
            
        return vals_dict, antoine_l_ge_10
    
    def getCore(self):
        
        vals_dict = self._data.get(InputParts.Core)
        if not (AttributeArgs.CoreArgs.protons  in vals_dict and
                AttributeArgs.CoreArgs.neutrons in vals_dict):
            raise ParserException("missing 'protons' or 'neutrons' in {}"
                                  .format(InputParts.Core))
        
        if not AttributeArgs.CoreArgs.innert_core in vals_dict:
            vals_dict[AttributeArgs.CoreArgs.innert_core] = 'None'
        
        return vals_dict
    
    def getForceEnum(self):
        force_elems = self._data.get(InputParts.Force_Parameters)
        force_dict = {}
        
        for force, force_params in force_elems.items():
            if not force_params:
                continue
                        
            assert force in ForceVariablesDict, ParserException(
                "Unimplemented force and arguments for [{}], see "
                "'ForceVariablesDict'".format(force))
            
            
            for param in ForceVariablesDict[force].members():
                param_elem = force_params.get(param)
                if param_elem == None:
                    raise ParserException("missing parameter [{}] in Force:"
                                          " [{}]".format(param, force))
                # TODO: List implementation will mismatch with the attribute 
                # fix it in the processing class
                force_params[param] = [str(_elem) for _elem in param_elem]
            
            force_dict[force] = force_params
                
            #skip non active forces
        
        if len(force_dict) == 0:
            raise ParserException("No Force given, must be one of these: {}"
                                  .format(ForceEnum.members()))
        return force_dict


class _XMLParser(_Parser):
    
    filetype = _Parser.Type.xml
    
    def __init__(self, xml_data):
        
        self._data = xml_data
    
    def getInteractionTitle(self):
        elem = self._data.find(InputParts.Interaction_Title)        
        return elem.attrib
    
    def getOutputArgs(self):
        elem = self._data.find(InputParts.Output_Parameters)
        
        vals_dict = {}
        for param in Output_Parameters.members():
            val = elem.find(param)
            if val == None:
                if param == Output_Parameters.Output_Filename:
                    raise ParserException("Missing filename in {} tag for XML input"
                                          .format(InputParts.Output_Parameters))
            else:
                val = val.text
            vals_dict[param] = val
        
        return vals_dict
    
    def getSHOArgs(self):
        elem = self._data.find(InputParts.SHO_Parameters)
        vals_dict = {}
        for param in SHO_Parameters.members():
            _aux = elem.find(param)
            
            if _aux == None:
                raise ParserException("missing parameter [{}] in {}"
                                      .format(param, InputParts.SHO_Parameters))
            vals_dict[param] = _aux.text
            
        return vals_dict
    
    def getValenceSpaceArgs(self):
        """ 
        The Antoine_ format for l < 10 or l > 10 in the sho states is assumed to 
        be correct, attribute "antoine_l_ge_10" is given to the main program to 
        interpret the quantum numbers.
        """
        elem = self._data.find(InputParts.Valence_Space)
        l_ge_10 = elem.attrib.get(ValenceSpaceParameters.l_great_than_10)
        antoine_l_ge_10 = False if l_ge_10 in ('False', None) else True
        
        vals_dict = {}
        for _item in elem.iter(ValenceSpaceParameters.Q_Number):
            key_ = _item.attrib.get(AttributeArgs.ValenceSpaceArgs.sp_state)
            val_ = _item.attrib.get(AttributeArgs.ValenceSpaceArgs.sp_energy)
            
            if val_ in (None, ''):
                val_ = None
                
            vals_dict[key_] = val_ 
            
        return vals_dict, antoine_l_ge_10
    
    def getCore(self):
        elem = self._data.find(InputParts.Core)
        vals_dict = {}
        
        for param in AttributeArgs.CoreArgs.members():
            val_ = elem.find(param)
                    
            if val_ in (None, ''):
                if param == AttributeArgs.CoreArgs.innert_core:
                    val_ = 0
                else:
                    raise ParserException("missing parameter [{}] in {}"
                                          .format(param, InputParts.Core))
            if param == AttributeArgs.CoreArgs.innert_core:
                # TODO: What to do with innert_core, append a
                val_ = val_.get(AttributeArgs.name)
            else:
                val_ = val_.text
            
            vals_dict[param] = val_
        
        # TODO: put a Core Nucleus Name
        
        return vals_dict
    
    def getForceEnum(self):
        force_elems = self._data.find(InputParts.Force_Parameters)
        force_dict = {}
        
        for force_elem in force_elems.getchildren():
            active_ = force_elem.attrib.get(AttributeArgs.ForceArgs.active)
            force = force_elem.tag
            
            if active_ in ('1', 'True', 'true', None):                
                assert force in ForceVariablesDict, ParserException(
                    "Unimplemented force and arguments for [{}], see "
                    "'ForceVariablesDict'".format(force))
                
                params = {}
                
                for param in ForceVariablesDict[force].members():
                    param_elem = force_elem.find(param)
                    if param_elem == None:
                        print("WARNING {}!: missing parameter [{}] in Force [{}]"
                              .format(self.__class__, param, force))
                        continue
                    params[param] = param_elem.attrib
                
                if len(params) == 0 and ForceVariablesDict[force].members()!=[]:
                    # If no parameters for a tag with not 'active' attribute, skip
                    continue
                
                # Assume there are different types of the same type of interaction
                # m.e (Central: Coulomb + Yukawa + ...)
                if force in force_dict:
                    force_dict[force].append(params)
                else:
                    force_dict[force] = [params]
                
            #skip non active forces
        
        if len(force_dict) == 0:
            raise ParserException("No Force given, must be one of these: {}"
                                  .format(ForceEnum.members()))
        return force_dict
        




class CalculationArgs(object):
    '''
    input 
    '''
    
    def __init__(self, input_data):
        
        self._data = None
        if isinstance(input_data, et.Element):
            self._data = _XMLParser(input_data)
        elif isinstance(input_data, dict):
            self._data = _JsonParser(input_data)
        else:
            raise ParserException("Invalid input data type: [{}], must be "
                                  "<xml.etree.ElementTree.Element> or <dict>"
                                  .format(input_data.__class__))
        
        self.Interaction_Title  = None
        self.Output_Parameters  = None
        self.SHO_Parameters     = None
        self.Valence_Space      = None
        self.Core               = None
        self.Force_Parameters   = None
        
        self.formatAntoine_l_ge_10 = False 
        
        self._settingParameters()
    
    
    def _settingParameters(self):
        setattr(self,
                InputParts.Interaction_Title, 
                self._data.getInteractionTitle())
        setattr(self,
                InputParts.Output_Parameters, 
                self._data.getOutputArgs())
        setattr(self,
                InputParts.SHO_Parameters, 
                self._data.getSHOArgs())
        setattr(self,
                InputParts.Core, 
                self._data.getCore())
        setattr(self,
                InputParts.Force_Parameters, 
                self._data.getForceEnum())
        
        val_dict, self.formatAntoine_l_ge_10 = self._data.getValenceSpaceArgs()
        setattr(self, InputParts.Valence_Space, val_dict)
        
        self._defineForces()
        
    def _defineForces(self):
        pass
    
    
    def getFilename(self):
        
        outp_   = getattr(self, InputParts.Output_Parameters)
        outp_fn = outp_.get(Output_Parameters.Output_Filename)
        
        if outp_fn:
            return outp_fn
        
        # The next part won't run because output_filename is required for the input
        spc = getattr(self, InputParts.Valence_Space).keys()
        spc = '_'.join(valenceSpaceShellNames(spc))
        
        outp_fn = spc + ''.join(getattr(self, InputParts.Force_Parameters).keys())
        
        return outp_fn
        
        
        
        
    






