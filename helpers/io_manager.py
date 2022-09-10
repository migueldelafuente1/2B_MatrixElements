'''
Created on Feb 23, 2021

@author: Miguel
'''
from helpers.Enums import InputParts, SHO_Parameters, ForceEnum,\
    AttributeArgs, ValenceSpaceParameters, Output_Parameters,\
    ForcesWithRepeatedParametersList, CoreParameters, OutputFileTypes
from helpers.Enums import ForceVariablesDict
import xml.etree.ElementTree as et
from helpers.WaveFunctions import QN_1body_jj
from helpers.Helpers import readAntoine, shellSHO_Notation, valenceSpacesDict_l_ge10,\
    getCoreNucleus
from copy import deepcopy


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
        raise Exception('invalid type for the state to read (only str, int or list)')
    
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
        
        if not CoreParameters.innert_core in vals_dict:
            vals_dict[CoreParameters.innert_core] = 'None'
        
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
                vals_dict[param] = None
                continue
                # if param != SHO_Parameters.Z:
                #     raise ParserException("missing parameter [{}] in {}"
                #                           .format(param, InputParts.SHO_Parameters))
                # else:
                #     continue
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
        
        for param in CoreParameters.members():
            val_ = elem.find(param)
                    
            if val_ in (None, ''):
                # if param == AttributeArgs.CoreArgs.innert_core:
                #     val_ = 0
                # else:
                pass#val_ = 0
                    # raise ParserException("missing parameter [{}] in {}"
                    #                       .format(param, InputParts.Core))
            if param == CoreParameters.innert_core:
                val_ = {}
                for attr_ in AttributeArgs.CoreArgs.members():
                    value = val_.get(attr_)
                    val_[attr_] = value if value else 0
                    if attr_ == AttributeArgs.name and not value:
                        val_[attr_] = getCoreNucleus(0, 0)
            else:
                val_ = '0' if val_ == None  else val_.text
            
            vals_dict[param] = val_
                
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
                
                params = self._getParamsForce(force, force_elem)
                
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
    
    def _getParamsForce(self, force, force_elem):
        """ Forces can have unique or multiple entries in the parameters """
        if force_elem.tag in ForcesWithRepeatedParametersList:
            return self._getForceParams_MultipleEntries(force, force_elem)
        else:
            return self._getForceParams_FixedEntries(force, force_elem)
    
    def _getForceParams_FixedEntries(self, force, force_elem):
        """ Entry tags are fixed and unique, read them from VariablesDict """
        params = {}
        
        for param in ForceVariablesDict[force].members():
            param_elem = force_elem.find(param)
            if param_elem == None:
                print("WARNING [{}]!: missing parameter [{}] in Force [{}] (might be optional)"
                      .format(self.__class__.__name__, param, force))
                continue
            params[param] = param_elem.attrib
        
        return params
    
    def _getForceParams_MultipleEntries(self, force, force_elem):
        """ 
        Entry tags are fixed but can be multiple (must have the same name ) 
        """
        params = {}
        
        for param in ForceVariablesDict[force].members():
            
            param_elems = force_elem.findall(param)
            
            if len(param_elems) > 1:
                for i, param_elem in enumerate(param_elems):
                    params[param + str(i)] = param_elem.attrib
            else:
                param_elem = param_elems[0]
                if param_elem == None:
                    print("WARNING [{}]!: missing parameter [{}] in Force [{}] (might be optional)"
                          .format(self.__class__.__name__, param, force))
                    continue
                params[param] = param_elem.attrib
        
        return params


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



class TBME_Reader():
    
    """ 
    Parser for output two-body interaction matrix element files to be imported
    both for Hamiltonian reading in TBME_Runner or for testing purpuses
    
    matrix elements are stored according the scheme.
    """
    
    class _Scheme:
        ## Only for inner use
        J  = 'J'
        JT = 'JT'
    
    _JSchemeIndexing = {
        0: (1,  1,  1,  1),  # pppp 
        1: (1, -1,  1, -1),  # pnpn 
        2: (1, -1, -1,  1),  # pnnp 
        3: (-1, 1,  1, -1),  # nppn 
        4: (-1, 1, -1,  1),  # npnp 
        5: (-1, -1,-1, -1),  # nnnn
    }
    
    def __init__(self, filename, 
                 ignorelines=None, constant=None, val_space=None, l_ge_10=True):
        """
        Args:
            :filename
        Optional arguments:
            :ignorelines <int>, Number of lines to skip before the JT blocks 
            :constant    <float>, Factor to multiply all the matrix elements
            :val_space   <list of Antoine indexes>, when given, it skips matrix
                          elements that contain other quantum numbers.
        """
        with open(filename, 'r') as f:
            data = f.readlines()
            if ignorelines:
                data = data[int(ignorelines):]
        if constant:
            self.const = float(constant)
        self.l_ge_10 = l_ge_10
        
        if filename.endswith(OutputFileTypes.sho):
            self.scheme = self._Scheme.JT
        elif filename.endswith(OutputFileTypes.twoBody):
            self.scheme = self._Scheme.J
        else:
            raise ParserException("Unidentified Scheme for importing")
        
        self._elements = {}
        self._valence_space = []
        self._strict_val_space = False
        if val_space:
            self._strict_val_space = True
            self._valence_space = [int(st) for st in val_space]
        
        self._importMatrixElements(data)
    
    def getMatrixElemnts(self, sorting_order=None):
        """ Get the imported matrix elements,
        : sorting order: <list> of tuples with the qqnn in """
        
        if sorting_order:
            dict_2 = {}
            q_numbs = sorting_order
            for i in range(len(q_numbs)):
                bra = q_numbs[i]
                if bra == (205, 205):
                    _=0
                if not bra in self._elements: 
                    continue
                else:
                    if not bra in dict_2:
                        dict_2[q_numbs[i]] = {}
            
                for j in range(0, len(q_numbs)): # start i = 0
                    ket = q_numbs[j]            
                    if not ket in self._elements[bra]:
                        continue
                    else:
                        if not ket in dict_2[bra]:
                            dict_2[bra][ket] = {}
                    
                    block = self._elements[q_numbs[i]][q_numbs[j]]
                    dict_2[q_numbs[i]][q_numbs[j]] = block
            
            self._elements = deepcopy(dict_2)
                
        return deepcopy(self._elements)
    
    def _importMatrixElements(self, data):
        if self.scheme == self._Scheme.JT:
            self._getJTSchemeMatrixElements(data)
        elif self.scheme == self._Scheme.J:
            self._getJSchemeMatrixElements(data)
    
    
    def _getJTSchemeMatrixElements(self, data):
        
        JT_block = {}
        bra, ket = None, None
        index    = 0
        j_min, j_max, T = 0, 0, 0
        
        for line in data:
            line = line.strip()
            if index == 0:
                bra, ket, j_min, j_max, skip = self._getBlockQN_HamilType(line)

                if skip:
                    index += 1
                    continue
                # JT_block[bra] = {0: {}, 1: {}}
                phs_bra, phs_ket, bra, ket = self._getPermutationPhase(bra, ket)
                if bra in JT_block:
                    if ket in JT_block[bra]:
                        print("WARNING, while reading the file found repeated "
                              "matrix element block:", bra, ket)
                    JT_block[bra][ket] = {0: {}, 1: {}}
                else:
                    JT_block[bra] = {ket : {0: {}, 1: {}}}
            elif index > 0 and (not skip):
                T = index - 1
                
                line = line.split()
                for i in range(j_max - j_min +1):
                    J = j_min + i
                    
                    phs = 1
                    if self.bra_exch:
                        phs *= (-1)**(phs_bra + J + T)
                    if self.ket_exch:
                        phs *= (-1)**(phs_ket + J + T)
                    
                    JT_block[bra][ket][T][J] = self.const * phs * float(line[i])
            
            index = index + 1 if index < 2 else 0
        
        self._elements = JT_block
    
    
    def _getJSchemeMatrixElements(self, data):
        
        J_block = {}
        bra, ket = None, None
        index = 0
        j_min, j_max, J = 0, 0, 0
        
        for line in data:
            line = line.strip()
            if index == 0:
                bra, ket, j_min, j_max, skip = self._getBlockQN_HamilType(line)
                j_length = j_max - j_min + 1
                if skip:
                    index += 1
                    continue
                
                j_dict = dict([(j, {}) for j in range(j_min, j_max+1)])
                
                phs_bra, phs_ket, bra, ket = self._getPermutationPhase(bra, ket)
                if bra in J_block:
                    if ket in J_block[bra]:
                        print("WARNING, while reading the file found repeated "
                              "matrix element block:", bra, ket)
                    J_block[bra][ket] = deepcopy(j_dict)
                else:
                    J_block[bra] = {ket : deepcopy(j_dict)}
                
                t_indexing = self._getParticleLabelIndexingAfterPermutation()
                
            elif index > 0 and (not skip):
                J = j_min + index - 1
                
                line = line.split()
                phs = 1
                if self.bra_exch:
                    phs *= (-1)**(phs_bra + J + 1)
                if self.ket_exch:
                    phs *= (-1)**(phs_ket + J + 1)
                
                for T in t_indexing:
                    J_block[bra][ket][J][T] = self.const * phs * float(line[T])
            
            index = index + 1 if index < j_length else 0
        
        self._elements = J_block
    
    def _getParticleLabelIndexingAfterPermutation(self):
        """ When bra or ket are exchanged, the index  of the particle
         lables have to be reorganized """
        t_indexing = [0, 1, 2, 3, 4, 5]
        #         pppp pnpn pnnp nppn npnp nnnn
        if self.bra_exch:
            if self.ket_exch:
                # pppp npnp nppn pnnp pnpn nnnn
                t_indexing = [0, 4, 3, 2, 1, 5]
            else:
                # pppp nppn npnp pnpn pnnp nnnn
                t_indexing = [0, 3, 4, 1, 2, 5]
        if self.ket_exch:
            #     pppp pnnp pnpn npnp nppn nnnn
            t_indexing = [0, 2, 1, 4, 3, 5]
        
        return t_indexing
    
    def _getBlockQN_HamilType(self, header):
        """ 
        :header <str>  with the sp states and JT values:
               0       1       1     103     103     205       1       2
               Tmin    Tmax    st1   st2     st3     st4       Jmin    Jmax
        """ 
        header = header.strip().split()
        # spss = [int(sp) for sp in header[2:6]]
        spss = [int(castAntoineFormat2Str(readAntoine(sp, self.l_ge_10), 
                                          True)) for sp in header[2:6]]
        # The valence space internally is always l_gt_10=True (convert)
        
        skip = False
        for st in spss:
            if st not in self._valence_space:
                if self._strict_val_space:
                    skip = True
                else:
                    self._valence_space.append(st)
        
        bra = min(tuple(spss[0:2]), tuple(spss[2:4]))
        ket = max(tuple(spss[0:2]), tuple(spss[2:4]))
        
        return bra, ket, int(header[-2]), int(header[-1]), skip
    
    def _getPermutationPhase(self, bra, ket):
        """ 
        return the bra and the ket in increasing order (avoids repetition).
        
        phases do not have the (J + 1) or (J + T) part (append afterwards), 
        just j1 + j2
        """
        
        self.bra_exch  = False
        self.ket_exch  = False
        phs_bra, phs_ket    = 0, 0
        
        if bra[0] > bra[1]:
            self.bra_exch = True
            bra = (bra[1], bra[0])
            phs_bra = (readAntoine(bra[0], self.l_ge_10)[2] + 
                       readAntoine(bra[1], self.l_ge_10)[2]) // 2
        if ket[0] > ket[1]:
            self.ket_exch = True
            ket = (ket[1], ket[0])
            phs_ket = (readAntoine(ket[0], self.l_ge_10)[2] + 
                       readAntoine(ket[1], self.l_ge_10)[2]) // 2
            
        return phs_bra, phs_ket, bra, ket


