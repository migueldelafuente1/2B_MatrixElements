'''
Created on Feb 23, 2021

@author: Miguel
'''
from helpers.Enums import InputParts, SHO_Parameters, ForceParameters, Enum,\
    AttributeArgs, ValenceSpaceParameters
from helpers.Enums import ForceVariablesDict
import xml.etree.ElementTree as et
from helpers.WaveFunctions import QN_1body_jj
from helpers.Helpers import valenceSpacesDict, shellSHO_Notation

def readAntoine(index):
    """ 
    returns the Quantum numbers from string Antoine's format:
        :return: [n, l, j], None if invalid
    """
    if isinstance(index, str):
        index = int(index)
    
    if(index == 1):
        return[0, 0, 1]
    else:
        if index % 2 == 1:
            n = int((index)/1000)
            l = int((index - (n*1000))/100)
            j = int(index - (n*1000) - (l*100))# is 2j 
            
            if (n >= 0) and (l >= 0) and (j > 0):
                return [n, l, j]
    
    raise Exception("Invalid state index for Antoine Format [{}]".format(index))

def castAntoineFormat2Str(state):
    """ 
    return a string with the state index in Antoine format. Check state.
    """
    if isinstance(state, QN_1body_jj):
        return state.AntoineStrIndex
    elif isinstance(state, int):
        readAntoine(state)
        if state == 1:
            return '001'
        return str(state)
    elif isinstance(state, str):
        if state == '1':
            return '001'
        readAntoine(state)
        return state

def valenceSpaceShellNames(valence_space):
    """
    Join the SHO major shells defining the valence space (without sense care).
    Arg:
    :valence_space Quantum numbers array (Antoine format)
    """
    _space = []
    states_accepted = 0
    for shell, qqnn_shell in valenceSpacesDict.items():
        
        aux = [(sp, shellSHO_Notation(*readAntoine(sp)))  for sp in qqnn_shell]
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
    
    def getOutputFilename(self):
        raise ParserException('abstract method, implement me!')
    
    def getSHOArgs(self):
        raise ParserException('abstract method, implement me!')
    
    def getValenceSpaceArgs(self):
        raise ParserException('abstract method, implement me!')
    
    def getCore(self):
        raise ParserException('abstract method, implement me!')
    
    def getForceParameters(self):
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
    
    def getOutputFilename(self):
        return self._data.get(InputParts.Output_Filename)
    
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
        
        if (qn_Ens == None) or (qn_Ens == []):
            qn_Ens = ['0.0' for _ in range(len(qn_Sts))]
        elif len(qn_Ens) != len(qn_Sts):
            raise ParserException("Missing single particle energy, QN_Energies"
                                  "[{}] mismatch with the number of states given"
                                  "[{}]".format(len(qn_Ens), len(qn_Sts)))
        
        vals_dict = {}
        for i in range(len(qn_Sts)):
            vals_dict[str(qn_Sts[i])] = str(qn_Ens[i])  
            
        return vals_dict
    
    def getCore(self):
        
        vals_dict = self._data.get(InputParts.Core)
        if not (AttributeArgs.CoreArgs.protons  in vals_dict and
                AttributeArgs.CoreArgs.neutrons in vals_dict):
            raise ParserException("missing 'protons' or 'neutrons' in {}"
                                  .format(InputParts.Core))
        
        if not AttributeArgs.CoreArgs.innert_core in vals_dict:
            vals_dict[AttributeArgs.CoreArgs.innert_core] = 'None'
        
        return vals_dict
    
    def getForceParameters(self):
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
                                  .format(ForceParameters.members()))
        return force_dict


class _XMLParser(_Parser):
    
    filetype = _Parser.Type.xml
    
    def __init__(self, xml_data):
        
        self._data = xml_data
    
    def getInteractionTitle(self):
        elem = self._data.find(InputParts.Interaction_Title)        
        return elem.attrib
    
    def getOutputFilename(self):
        elem = self._data.find(InputParts.Output_Filename)
        return elem.text
    
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
        elem = self._data.find(InputParts.Valence_Space)
        vals_dict = {}
        for _item in elem.iter(ValenceSpaceParameters.Q_Number):
            key_ = _item.attrib.get(AttributeArgs.ValenceSpaceArgs.sp_state)
            val_ = _item.attrib.get(AttributeArgs.ValenceSpaceArgs.sp_energy)
            
            if val_ in (None, ''):
                val_ = 0.0
                
            vals_dict[key_] = val_ 
            
        return vals_dict
    
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
    
    def getForceParameters(self):
        force_elems = self._data.find(InputParts.Force_Parameters)
        force_dict = {}
        
        for force_elem in force_elems.getchildren():
            active_ = force_elem.attrib.get(AttributeArgs.ForceArgs.active)
            force = force_elem.tag
            
            if active_  in ('1', 'True', 'true', None):                
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
                
                if len(params) == 0:
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
                                  .format(ForceParameters.members()))
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
        
        
        setattr(self,
                InputParts.Interaction_Title, 
                self._data.getInteractionTitle())
        setattr(self,
                InputParts.Output_Filename, 
                self._data.getOutputFilename())
        setattr(self,
                InputParts.SHO_Parameters, 
                self._data.getSHOArgs())
        setattr(self,
                InputParts.Core, 
                self._data.getCore())
        setattr(self,
                InputParts.Valence_Space, 
                self._data.getValenceSpaceArgs())
        setattr(self,
                InputParts.Force_Parameters, 
                self._data.getForceParameters())
        
        self._defineForces()
    
    def _defineForces(self):
        pass
    
    
    def getFilename(self):
        
        outp_fn = getattr(self, InputParts.Output_Filename)
        if outp_fn:
            return outp_fn
        
        spc = getattr(self, InputParts.Valence_Space).keys()
        spc = '_'.join(valenceSpaceShellNames(spc))
        
        outp_fn = spc + ''.join(getattr(self, InputParts.Force_Parameters).keys())
        
        return outp_fn
        
        
        
        
    






