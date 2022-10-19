'''
Created on Feb 23, 2021

@author: Miguel
'''
from helpers.io_manager import CalculationArgs, readAntoine,\
    castAntoineFormat2Str, valenceSpaceShellNames, ParserException, TBME_Reader
import json
import time

from helpers.Enums import InputParts as ip, AttributeArgs, OUTPUT_FOLDER,\
    SHO_Parameters, Output_Parameters, OutputFileTypes,\
    CouplingSchemeEnum, ForceEnum, ForceFromFileParameters
from matrix_elements import switchMatrixElementType
from matrix_elements.MatrixElement import _TwoBodyMatrixElement
from itertools import combinations_with_replacement
from helpers.WaveFunctions import QN_1body_jj, QN_2body_jj_JT_Coupling,\
    QN_2body_jj_J_Coupling
from copy import deepcopy
from helpers.Helpers import recursiveSumOnDictionaries, getCoreNucleus,\
    Constants, almostEqual, printProgressBar, _LINE_1, _LINE_2

import os
import numpy as np
import traceback 

class TBME_Runner(object):
    '''
    The two body matrix element runner evaluate all matrix elements for a given
    valence space and Force parameters.
    '''
    NULL_TOLERANCE = _TwoBodyMatrixElement.NULL_TOLERANCE
    PRINT_LOG = True
    
    RESULT_FOLDER = OUTPUT_FOLDER
    
    _Scheme = TBME_Reader._Scheme
        # ## Only for inner use
        # J  = 'J'
        # JT = 'JT'
    
    _JSchemeIndexing = TBME_Reader._JSchemeIndexing
        # 0: (1,  1,  1,  1),  # pppp 
        # 1: (1, -1,  1, -1),  # pnpn 
        # 2: (1, -1, -1,  1),  # pnnp 
        # 3: (-1, 1,  1, -1),  # nppn 
        # 4: (-1, 1, -1,  1),  # npnp 
        # 5: (-1, -1,-1, -1),  # nnnn
    
    def __init__(self, filename='', verbose=True, manual_input={}):
        
        self.filename   = filename
        self.PRINT_LOG  = verbose
        self.input_obj  = None
        self.tbme_class = None
        self.l_ge_10    = False
        self.valence_space   = []
        self._hamil_type     = '1'
        self._com_correction = False
        
        self.results = {}
        self.resultsByInteraction = {}
        self.resultsSummedUp    = {}
        self.com_2bme           = {}
        self.interactionSchemes = {}
        self._brokeInteractions = {}
        self._forceNameFromForceEnum = {}
        
        self.filename_output = 'out'
        
        if filename.endswith('.json'):
            self._readJsonInputFile()
        elif filename.endswith('.xml'):
            self._readXMLInputFile()
        else:
            raise TBME_RunnerException(
                'Invalid input file [{}], must be xml or json'.format(filename))
        
        self.filename_output  = self.RESULT_FOLDER +'/'+ self.input_obj.getFilename()
        self._setHamilTypeAndCOMCorrection()
    
    def _setHamilTypeAndCOMCorrection(self, hamil_type=None, com_correct=None):
        """
        Method to define the global output, hamil_types for TAURUS include options
        that require extra files (1body, 2body, sho) or define in different ways
        those files. The method is mind to be use automatically from the input 
        file but the keyword arguments let us define directly for testing.
        
        :hamil_type <int> 
            1,2: just return one file .sho (2 differences neutron/proton 
                energies in JT scheme (ANTOINE format)
            3:  general hamiltonian_ J scheme, returns a file for the sho 
                definition (.sho), the tbme in the file (.2b) and the one body 
                matrix elements in another (.01b)
            4:  'bare' hamiltonian_ in the J scheme
            
            more information in TAURUS manual
        """
        type_= self.input_obj.Output_Parameters.get(Output_Parameters.Hamil_Type)            
        com_ = self.input_obj.Output_Parameters.get(Output_Parameters.COM_correction)            
        
        if type_ != None:
            if type_ in list('1234'):
                self._hamil_type = type_
            else:
                ParserException('Hamil_type value must be an integer in range 1 to 4')
        if com_ != None:
            if com_ in ('0','1'):
                self._com_correction = bool(int(com_))
            else:
                ParserException('COM_correction value must be an integer: 0 (off) or 1 (on)')
        
    
    def _readJsonInputFile(self):  
        print("Warning! input by json is DEPRECATED in the program")
        
        with open(self.filename, 'r') as jf:
            data = json.load(jf)
        
        self.input_obj = CalculationArgs(data)
    
    def _readXMLInputFile(self):
        """ Read """
        import xml.etree.ElementTree as et
        
        tree  = et.parse(self.filename)
        _root = tree.getroot()
        
        self.input_obj = CalculationArgs(_root)
    
    __advertence_Kin2Body = """
    WARNING: [Kinetic_2Body] in not an available force, the interaction is only 
to evaluate the 2Body center of mass correction. Set in the input the entry:
    In <Output_Parameters> : <COM_correction>1</COM_correction>
to get these matrix elements in a separated file (.com extension). 
The program will exclude it from the interaction file and will produce the .com file.
"""
    
    def _checkHamilTypeAndForces(self):
        """ 
        Certain interactions cannot be expressed in JT formalism if they break
        isospin_ symmetry (f.e Coulomb_). That means the global scheme for the
        matrix elements must be J-scheme, and then, only hamilType=3/4 are 
        accepted to print the elements.
        
        This method checks that if HamilType is 0/1, then it must not be any 
        isospin_ breaking interactions (before any calculations).
        
        Also fill the interaction schemes.
        If com correction, add to the 
        """
        
        _forcesAttr = ip.Force_Parameters
        J_schemeForces = []
        JT_schemeForces = []
        T_breaking = []
        self._forces2ReadFromFile = False
        
        for force in getattr(self.input_obj, _forcesAttr):
            
            me = switchMatrixElementType(force)
            
            if me._BREAK_ISOSPIN: # cannot use property, m.e. not instanced
                T_breaking.append(force)
            
            f_scheme = me.COUPLING
            f_scheme = [f_scheme,] if isinstance(f_scheme, str) else f_scheme
            
            if CouplingSchemeEnum.L in f_scheme or CouplingSchemeEnum.S in f_scheme:
                raise TBME_RunnerException("Matrix Element in L or S scheme,"
                                           "class can only run in J or JT scheme")
            
            if (CouplingSchemeEnum.JJ in f_scheme):
                if force == ForceEnum.Force_From_File: 
                    self._forces2ReadFromFile = True
                    continue
                elif force == ForceEnum.Kinetic_2Body:
                    print(self.__advertence_Kin2Body)
                    self._com_correction = True
                    continue
                
                if (CouplingSchemeEnum.T in f_scheme):
                    JT_schemeForces.append(force)
                    self.interactionSchemes[force] = self._Scheme.JT
                    if me._BREAK_ISOSPIN:
                        raise TBME_RunnerException(
                            "Error in force [{}] <class>:[{}] definition"
                            .format(force, me.__class__.__name__) +
                            ", Isospin based (reduced) matrix elements breaks T.")
                else:
                    J_schemeForces.append(force)
                    self.interactionSchemes[force] = self._Scheme.J
        
        if self._com_correction:
            # append the K 2Body to the input force to compute (remove after)
            getattr(self.input_obj, _forcesAttr)[ForceEnum.Kinetic_2Body] = [dict()]
            JT_schemeForces.append(ForceEnum.Kinetic_2Body)
            self.interactionSchemes[ForceEnum.Kinetic_2Body] = self._Scheme.JT
        
        if self._hamil_type in '12':
            if len(T_breaking) > 0:
                raise TBME_RunnerException("Cannot compute isospin-breaking interactions"
                    "{} to write in hamilType format 1-2 (JT scheme)".format(T_breaking))
            if len(JT_schemeForces) == 0 and (not self._forces2ReadFromFile):
                # TODO: might convert to JT scheme J forces.
                raise TBME_RunnerException("Cannot compute J forces JT scheme")
        
    
    def _defineSHOParameters(self):
        """
        Hierarchy:
            hbar*omega not given, extract it from A aproximation: 41*A^{-1/3} MeV
            then, if b_param is not given set it by hbar_omega
            
            if A is not given, set hbar omega to 0 
            
            if b_param is given, it remains (might not be comparible with default
            h_bar_omega approximation for A_mass)
            
            if b_param not given (neither A and hbar*omega) raise error (at least
            raise an error to set it as 1)
        """
        sho_params = self.input_obj.SHO_Parameters
        
        A_mass      = sho_params.get(SHO_Parameters.A_Mass)
        hbar_omega  = sho_params.get(SHO_Parameters.hbar_omega)
        b_length    = sho_params.get(SHO_Parameters.b_length)
        
        if b_length != None:
            b_length = float(b_length)
        
        if hbar_omega == None:
            if A_mass == None:
                if b_length == None:
                    raise TBME_RunnerException("SHO Parameters Incomplete, "
                            "b_length is required. Got".format(sho_params))
                hbar_omega = 0
            else:
                A_mass = int(A_mass)
                hbar_omega = 41 / (A_mass**(1/3))
                if b_length == None:
                    b_length = Constants.HBAR_C 
                    b_length /= np.sqrt(Constants.M_NUCLEON * hbar_omega)
                else:
                    hbar_omega  = Constants.HBAR_C**2 
                    hbar_omega /= (Constants.M_MEAN * (b_length**2))
        else:
            hbar_omega = float(hbar_omega)
            if b_length == None:
                b_length = Constants.HBAR_C 
                b_length /= np.sqrt(Constants.M_NUCLEON * hbar_omega)
            if A_mass == None:
                A_mass = 0
        
        self.input_obj.SHO_Parameters[SHO_Parameters.A_Mass]     = A_mass
        self.input_obj.SHO_Parameters[SHO_Parameters.hbar_omega] = hbar_omega
        self.input_obj.SHO_Parameters[SHO_Parameters.b_length]   = b_length
        
    
    def _defineValenceSpaceEnergies(self):
        """ 
        Method define the default single particle energies for the valence 
        space based in the quantity h_bar*omega given in the SHO parameters.
        
        Check inconsistency in energy definitions (all settled or none of it)
        """
        self._defineSHOParameters()
        
        hbar_omega = self.input_obj.SHO_Parameters.get(SHO_Parameters.hbar_omega)
        hbar_omega = float(hbar_omega)
        
        valence_space_aux = self.input_obj.Valence_Space
        l_ge_10 = self.input_obj.formatAntoine_l_ge_10
        self.l_ge_10 = l_ge_10
        ## !! All q.n converted to l_ge_10 for TBME_Runner to use
        valence_space = {}
        for spst_ant, spe_ener in valence_space_aux.items():
            n, l, _j = readAntoine(spst_ant, l_ge_10)
            spst_ant = castAntoineFormat2Str((n, l, _j), l_ge_10=True)
            valence_space[spst_ant] = spe_ener
        self.input_obj.Valence_Space = valence_space
        
        null_type_elems = [energ == None for energ in valence_space.values()]
        if not False in null_type_elems:
            # none were defined (settled as '' in input)
            for spst_ant in valence_space:
                
                n, l, _j = readAntoine(spst_ant, l_ge_10)
                energ = hbar_omega * (2*n + l + 1.5)
                
                self.input_obj.Valence_Space[spst_ant] = float(energ)
                 
        elif True in null_type_elems:
            #error, some sp_energies given but not all
            raise TBME_RunnerException("Single particle energies given and others"
                    " settled by default (''), must set all or none. Got: [{}]"
                    .format(valence_space))
        else:
            # all were defined
            pass
    
    def _sortQQNNFromTheValenceSpace(self):
        """ 
        sort and combine in order q. numbers for the two body wave functions
        access by attribute self._twoBodyQuantumNumbersSorted
        """
        q_numbs = getattr(self.input_obj, ip.Valence_Space)
        self.valence_space = valenceSpaceShellNames(q_numbs)
        
        q_numbs = map(lambda qn: int(qn), q_numbs)
        q_numbs = sorted(q_numbs)#, reverse=True)
        q_numbs = list(combinations_with_replacement(q_numbs, 2))
        
        self._twoBodyQuantumNumbersSorted = q_numbs
    
    def _computeForValenceSpaceJTCoupled(self, force=''):
        """ 
        method to run the whole valence space m.e. in the JT scheme
        """
        q_numbs = self._twoBodyQuantumNumbersSorted
        
        for i in range(len(q_numbs)):
            n1_bra = QN_1body_jj(*readAntoine(q_numbs[i][0], l_ge_10=True))
            n2_bra = QN_1body_jj(*readAntoine(q_numbs[i][1], True))
            
            self.results[q_numbs[i]] = {}
            
            J_bra_min = abs(n1_bra.j - n2_bra.j) // 2
            J_bra_max = (n1_bra.j + n2_bra.j) // 2
            
            for j in range(i, len(q_numbs)):
                n1_ket = QN_1body_jj(*readAntoine(q_numbs[j][0], True))
                n2_ket = QN_1body_jj(*readAntoine(q_numbs[j][1], True))
                
                self.results[q_numbs[i]][q_numbs[j]] = {0: {},  1: {}}
                
                J_min = max(abs(n1_ket.j - n2_ket.j) // 2, J_bra_min)
                J_max = min((n1_ket.j + n2_ket.j) // 2,    J_bra_max)
                
                self._count += 1
                for T in (0, 1):
                    # assume M.E. cannot couple <(JT)|V|J'T'> if J', T' != J, T
                    for J in range(J_min, J_max +1):
                        tic = time.time()
                        
                        bra = QN_2body_jj_JT_Coupling(n1_bra, n2_bra, J, T)
                        ket = QN_2body_jj_JT_Coupling(n1_ket, n2_ket, J, T)

                        me = self.tbme_class(bra, ket)
                        self.results[q_numbs[i]][q_numbs[j]][T][J] = me.value
                        
                        if me.value and self.PRINT_LOG:
                            print(' * me[{}/{}]_({:.4}s): <{}|V|{} (J:{}T:{})> {}'
                                  .format(self._count, self._total_me,
                                          time.time()-tic,
                                          bra.shellStatesNotation, 
                                          ket.shellStatesNotation, J, T, force))
                            print('\t= {:.8} '.format(me.value))                        
    
    def _computeForValenceSpaceJCoupled(self, force=''):
        """ 
        method to run the whole valence space m.e. in the J scheme
        Indexing for J row:
            {0: pppp, 1:pnpn, 2:pnnp, 3:nppn, 4:npnp, 5:nnnn}
        """
        q_numbs = self._twoBodyQuantumNumbersSorted
        
        for i in range(len(q_numbs)):
            n1_bra = QN_1body_jj(*readAntoine(q_numbs[i][0], l_ge_10=True))
            n2_bra = QN_1body_jj(*readAntoine(q_numbs[i][1], True))
            
            self.results[q_numbs[i]] = {}
            
            J_bra_min = abs(n1_bra.j - n2_bra.j) // 2
            J_bra_max = (n1_bra.j + n2_bra.j) // 2
            
            for j in range(i, len(q_numbs)):
                n1_ket = QN_1body_jj(*readAntoine(q_numbs[j][0], True))
                n2_ket = QN_1body_jj(*readAntoine(q_numbs[j][1], True))
                
                J_min = max(abs(n1_ket.j - n2_ket.j) // 2, J_bra_min)
                J_max = min((n1_ket.j + n2_ket.j) // 2,    J_bra_max)
                
                aux = [(J, {}) for J in range(J_min, J_max +1)]
                self.results[q_numbs[i]][q_numbs[j]] = deepcopy(dict(aux))
                
                self._count += 1
                for J in range(J_min, J_max +1):
                    for m, mts in self._JSchemeIndexing.items():
                        tic = time.time()
                        
                        n1_bra.m_t = mts[0]
                        n2_bra.m_t = mts[1]
                        n1_ket.m_t = mts[2]
                        n2_ket.m_t = mts[3]
                        
                        bra = QN_2body_jj_J_Coupling(n1_bra, n2_bra, J)
                        ket = QN_2body_jj_J_Coupling(n1_ket, n2_ket, J)
                        
                        me = self.tbme_class(bra, ket)
                        self.results[q_numbs[i]][q_numbs[j]][J][m] = me.value
                        
                        if me.value and self.PRINT_LOG:
                            print(' * me[{}/{}]_({:.4}s): <{}|V|{} (J:{})> {}'
                                  .format(self._count, self._total_me,
                                          time.time()-tic, 
                                          bra, ket, J, force))
                            print('\t= {:.8} '.format(me.value))
                if not self.PRINT_LOG:
                    printProgressBar(self._count, self._total_me, 
                                     prefix='Progress '+force+':')
    
    def _computeForValenceSpace(self, force):
        len_q_numbs = len(self._twoBodyQuantumNumbersSorted)
        # start the count for printing
        self._count = 0
        self._total_me = (len_q_numbs * (len_q_numbs + 1)) // 2
        
        if self.interactionSchemes[force] == self._Scheme.J:
            self._computeForValenceSpaceJCoupled(force)
        elif self.interactionSchemes[force] == self._Scheme.JT:
            self._computeForValenceSpaceJTCoupled(force)
        else:
            raise TBME_RunnerException("force [{}] invalid scheme: {}".format(
                                       force, self.interactionSchemes[force]))
            
    
    def _readMatrixElementsFromFile(self, force_str, **params):
        """ 
        TODO: implement in an external reading io_manager.
        
        Proceed to read all the matrix elements from a file (J or JT scheme)
        filename is mandatory.
        """
        filename = params.get(ForceFromFileParameters.file)[AttributeArgs.name]
        options  = params.get(ForceFromFileParameters.options)
        ## Options
        ignorelines = options.get(AttributeArgs.FileReader.ignorelines)
        if ignorelines:
            ignorelines = int(ignorelines)
        constant = options.get(AttributeArgs.FileReader.constant)
        if constant:
            constant = float(constant)
        l_ge_10 = options.get(AttributeArgs.FileReader.l_ge_10)
        if l_ge_10:
            l_ge_10 = False if l_ge_10.lower() == 'false' else True
        
        valence_space = list(self.input_obj.Valence_Space.keys())
        
        data_ = TBME_Reader(filename, ignorelines, 
                            constant, valence_space, l_ge_10)
        
        if data_.scheme == self._Scheme.J:
            self.interactionSchemes[force_str] = self._Scheme.J
            if self._hamil_type in '12':
                raise TBME_RunnerException("imported matrix elements [{}] are in"
                    " the J scheme, but Hamiltype 1-2 (JT scheme)".format(filename))
        elif data_.scheme == self._Scheme.JT:
            self.interactionSchemes[force_str] = self._Scheme.JT
        
        self.results = data_.getMatrixElemnts(self._twoBodyQuantumNumbersSorted)
        self.resultsByInteraction[force_str] = deepcopy(self.results)
    
    
    def combineAllResults(self):
        
        kin_key = ForceEnum.Kinetic_2Body
        final_J  = {}
        final_JT = {}
        for force, results in self.resultsByInteraction.items():
            if self.__class__ == TBME_Runner:
                force = self._forceNameFromForceEnum[force]
            
            if self.interactionSchemes[force] == self._Scheme.J:
                # Kin 2Body is a JT scheme interaction
                recursiveSumOnDictionaries(results, final_J)
            else:
                if force == kin_key:
                    conv_2J = self._hamil_type in '34', False
                    results = self._convertMatrixElemensScheme(results, *conv_2J)
                    self.com_2bme = results
                    continue # do not add the Kin 2Body
                recursiveSumOnDictionaries(results, final_JT)
        
        if self._hamil_type in '12':
            final = final_JT
            # TODO: convert J dictionary to JT dictionary
            final_J = self._convertMatrixElemensScheme(final_J, False, True)
            recursiveSumOnDictionaries(final_J, final)
        elif self._hamil_type in '34':
            final = final_J
            # TODO: convert JT dictionary to J dictionary
            final_JT = self._convertMatrixElemensScheme(final_JT, True, False)
            recursiveSumOnDictionaries(final_JT, final)
        self.resultsSummedUp = final
        
        # remove the Kin Entry in thee Forces interaction
        if kin_key in getattr(self.input_obj, ip.Force_Parameters):
            del getattr(self.input_obj, ip.Force_Parameters)[kin_key]
                            
    
    def run(self):
        """
        Calculate all the matrix elements for all the interactions, and 
        print its combination in a file.
        """
        self._defineValenceSpaceEnergies()
        self._sortQQNNFromTheValenceSpace()      
        self._checkHamilTypeAndForces()
        
        _forcesAttr = ip.Force_Parameters
        times_ = {}
        for force, force_list in getattr(self.input_obj, _forcesAttr).items():
            i = 0
            for params in force_list:
                force_str = force+str(i) if len(force_list) > 1 else force
                # update arrays and variables
                i += 1
                self.results = {}
                self._forceNameFromForceEnum[force_str] = force
                tic_ = time.time()
                if force == ForceEnum.Force_From_File:
                    ## read from file case
                    self._readMatrixElementsFromFile(force_str, **params)
                    times_[force_str] = round(time.time() - tic_, 4)
                    continue
                ## computable interaction
                try:
                    sho_params = getattr(self.input_obj, ip.SHO_Parameters)
                    
                    self.tbme_class = switchMatrixElementType(force)
                    self.tbme_class.resetInteractionParameters(also_SHO=True)
                    self.tbme_class.setInteractionParameters(**params, 
                                                             **sho_params)
                    self._computeForValenceSpace(force)
                    
                    self.resultsByInteraction[force_str] = deepcopy(self.results)
                    times_[force_str] = round(time.time() - tic_, 4)
                    print(" Force [{}] m.e. calculated: [{}]s"
                                .format(force, times_[force_str]))
                except BaseException:
                    trace = traceback.format_exc()
                    self._procedureToSkipBrokenInteraction(force_str, trace)
        
        print("Finished computation, Total time (s): [", sum(times_.values()),"]=")
        print("\n".join(["\t"+str(t)+"s" for t in times_.items()]))
        if len(self._brokeInteractions) > 0:
            print(_LINE_1, "ERROR !! Interactions have errors"
                  " and where skipped with these exceptions")
            for f, excep in self._brokeInteractions.items():
                print("Interaction [{}], Exception ::\n{}\n{}" .format(f, excep, _LINE_2))
        
        self.combineAllResults()
        self.printMatrixElementsFile()
        self.printComMatrixElements()
    
    def _procedureToSkipBrokenInteraction(self, force_str, exception_trace):
        """
        Remove matrix elements from results and append Warning at the end.
        File name is changed to _RECOVERY_[filename].2b
        """
        if force_str != ForceEnum.Kinetic_2Body:
            fn = self.input_obj.Output_Parameters.get(Output_Parameters.Output_Filename)
            self.filename_output = self.RESULT_FOLDER + '/' + '_RECOVERY_' + fn
        else:
            self._com_correction = False
        
        self._brokeInteractions[force_str] = str(exception_trace)
    
    def _valenceSpaceLine(self):
        
        valence = getattr(self.input_obj, ip.Valence_Space)
        valence = sorted([tuple(x) for x in valence.items()], key=lambda x: x[1])
        
        if self._hamil_type in '34':            
            aux = [str(len(valence))]
            # convert to l_ge_10 for taurus input
            for i in range(len(valence)):
                stt, _en = valence[i]
                stt = castAntoineFormat2Str(readAntoine(stt, True), True)
                valence[i] = (stt, _en)
        else:
            aux = [self._hamil_type, str(len(valence))]
            if self.l_ge_10:
                # translate l > 10 to l < 10 in case states given in l > 10
                len_ = len(valence)
                aux_valence = []
                for i in range(len_):
                    stt, _en = valence[i]
                    stt = castAntoineFormat2Str(readAntoine(stt, True), False)
                    aux_valence.append((stt, _en))
                valence = aux_valence
        
        valen = '\t' + ' '.join(aux + [x[0] for x in valence])
        energ = '\t' + ' '.join([str(x[1]) for x in valence])
                
        return valen, energ
    
    def _interactionStringInHeader(self):
        """ 
        Auxiliary method to warning in the title that M.E  has been skipped.
        """
        str_ok = '. ME evaluated: ' + ' + '.join(
                            getattr(self.input_obj, ip.Force_Parameters).keys())
        ## ! Append warning if broken interactions.
        if len(self._brokeInteractions) > 0:
            if len(self._brokeInteractions) == 1:
                if ForceEnum.Kinetic_2Body in self._brokeInteractions:
                    return str_ok
            fcs = getattr(self.input_obj, ip.Force_Parameters).keys()
            str_fails = []
            for f in fcs:
                is_f_ko = [f in key_ for key_ in self._brokeInteractions.keys()]
                if True in is_f_ko:
                    f = "({} ({}/{}) SKIPPED)".format(f, is_f_ko.count(True), 
                                        len(self.input_obj.Force_Parameters[f]))
                str_fails.append(f)
            str_fails = '. ME PARTIALLY! evaluated: ' + ' + '.join(str_fails)
            return str_fails
        
        return str_ok
    
    def _headerFileWriting(self):
        """ 
        Writing Header of the Antoine file: title / valence space / Energies 
        """
        
        title_args = getattr(self.input_obj, ip.Interaction_Title)
        title = '  '
        title += title_args.get(AttributeArgs.name)
        if title_args.get(AttributeArgs.details):
            title += ' :({})'.title_args.get(AttributeArgs.details)
        title += self._interactionStringInHeader()
          
        title += '. Shell({})'.format('+'.join(self.valence_space))
        b_len = self.input_obj.SHO_Parameters.get(SHO_Parameters.b_length, None)
        if b_len and b_len > self.NULL_TOLERANCE:
            title += "(B={:06.4f}fm)".format(b_len)
        
        valen, energ = self._valenceSpaceLine()
        
        
        core = getattr(self.input_obj, ip.Core) 
        
        if self._hamil_type in '34':
            core_args = [core.get(AttributeArgs.CoreArgs.protons, '0'),
                         core.get(AttributeArgs.CoreArgs.neutrons, '0')]
        else:
            _apply_density_correction = '0'
            core_args = [
                _apply_density_correction,
                core.get(AttributeArgs.CoreArgs.protons, '0'),
                core.get(AttributeArgs.CoreArgs.neutrons, '0'), 
                '0.300000']#, '0.000000'
        #title += ' (Core: {})'.format(getCoreNucleus(*core_args[1:3]))
        core_args = '\t' + ' '.join(core_args) 
        self.title = title
        
        return [title, valen, energ, core_args]
    
    
    def _formatValues2Standard(self, J_vals):
        """ 
        Cast all numerical values to standard form: all with 6 digits, 
        scientific notation if lower than 1.e-6, number under NULL_TOLERANCE 
        are rounded to 0.000000
        :J_vals <dict> {[j ]: [<ab|V|cd>(j) ]}
        
        0.000000153 -> 1.530e-7
        0.00000153  -> 0.000002
        0.0         -> 0.000000
        """
        
        all_null = True
        values = []
        
        for mat_elem in J_vals.values():
            
            if abs(mat_elem) > self.NULL_TOLERANCE:
                all_null = False
                
                if abs(mat_elem) < 1.e-10:
                    values.append("{: 12.10e}".format(mat_elem))
                else:
                    values.append("{: 12.10f}".format(mat_elem))
            else:
                try:
                    values.append("{: .10f}".format(mat_elem))
                except TypeError as tp:
                    # TODO: 
                    pass
        return all_null, values
    
    def _getJvaluesFromIsospin(self, val_t0, val_t1, J, bra1, bra2, ket1, ket2):
        """ Compute JT decomposition into the J scheme proton/neutron labeled """
        #=======================================================================
        # pppp = val_t1 / 1.7320508075688772 # sqrt(3)
        # nnnn = pppp
        #
        # pnpn = ( 0.5 * val_t0) + (0.28867513459481287 * val_t1)  # sqrt(3)/6
        # pnnp = (-0.5 * val_t0) + (0.28867513459481287 * val_t1)
        # nppn = pnnp
        # npnp = pnpn
        #
        # return {0: pppp, 1:pnpn, 2:pnnp, 3:nppn, 4:npnp, 5:nnnn}
        #=======================================================================
        
        ## Suhonen_ definition
        pppp = val_t1
        nnnn = pppp
        
        phs_J = (-1)**J
        aux_0 = np.sqrt((1 - (phs_J*(bra1==bra2)))*(1 - (phs_J*(ket1==ket2))))
        aux_1 = np.sqrt((1 + (phs_J*(bra1==bra2)))*(1 + (phs_J*(ket1==ket2))))
        
        pnpn = 0.5*((aux_1 * val_t1) + (aux_0 * val_t0))
        npnp = pnpn
        pnnp = 0.5*((aux_1 * val_t1) - (aux_0 * val_t0))
        nppn = pnnp
        
        return {0: pppp, 1:pnpn, 2:pnnp, 3:nppn, 4:npnp, 5:nnnn}
    
    def _getIsospinvaluesFromLabels(self, j_block, J, bra1, bra2, ket1, ket2):
        """ Reversed of the previous method, also checks <nnnn>=<pppp>, etc. """
        pppp, nnnn = j_block[0], j_block[5]
        pnpn, npnp = j_block[1], j_block[4]
        pnnp, nppn = j_block[2], j_block[3]
        
        assert almostEqual(pppp, nnnn, tolerance=1e-9), "nnnn m.e != pppp m.e."
        assert almostEqual(pnpn, npnp, tolerance=1e-9), "pnpn m.e != npnp m.e."
        assert almostEqual(pnnp, nppn, tolerance=1e-9), "nppn m.e != pnnp m.e."
        
        v_T1 = pppp
        v_T0 = (pnpn - pnnp)
        
        if bra1==bra2 or ket1==ket2:
            if J % 2 == 0:
                v_T0 = 0.0
            else:
                v_T0 /= 2   ## Aux0 = 2
                v_T1 = 0.0
        #else:  ## Aux0 = 1
        
        return v_T0, v_T1
    
    def printMatrixElementsFile(self):
        
        if self._hamil_type in '12':
            # Basic Antoine Format (2 difference between neutrons-protons)
            self._printHamilType_12_File()
        elif self._hamil_type == '3':
            # J scheme 3 file output  (0, 1 & 2 body Hamiltonian_)
            print("WARNING :: hamil_type=3 not jet implemented, running hamil_type=4")
            self._printHamilType_34_Files()            
            # raise TBME_RunnerException("hamil_type=3 not jet implemented")
        else:
            # J scheme "bare" hamiltonian, return .sho header and .2b files
            self._printHamilType_34_Files()
    
    
    def _getJBoundsForT0And1Match(self, J_vals_T0, J_vals_T1):
        """ Test method for block printing. """
        jmin_0, jmax_0 = min(J_vals_T0), max(J_vals_T0)
        if jmin_0 != min(J_vals_T1):
            raise TBME_RunnerException("T0 T1 jmin doesn't match: [{}]T=0 [{}]T=1"
                                       .format(J_vals_T0, J_vals_T1))
        if jmax_0 != max(J_vals_T1):
            raise TBME_RunnerException("T0 T1 jmax doesn't match: [{}]T=0 [{}]T=1"
                                       .format(J_vals_T0, J_vals_T1))
        
        return jmin_0, jmax_0
    
    def _write2BME_JTBlock(self, bra1, bra2, ket1, ket2, T_vals):
        """
        Return list with 3 strings for a T0,1 block of the qqnn if there are 
        non null values, empty list if doesn't.
        """       
        spss_str = ' '.join([bra1, bra2, ket1, ket2])
        str_me_space = []
        
        all_null = True
        
        J_vals_T0 = T_vals[0]
        J_vals_T1 = T_vals[1]
        
        if J_vals_T0 and J_vals_T1: 
            jmin, jmax = self._getJBoundsForT0And1Match(J_vals_T0, J_vals_T1)
            
            str_me_space.append(' 0 1 {} {} {}'.format(spss_str, jmin, jmax))
            
            t0_null, t0_vals_str = self._formatValues2Standard(J_vals_T0)
            t1_null, t1_vals_str = self._formatValues2Standard(J_vals_T1)
            
            all_null = t0_null and t1_null
            str_me_space.append('\t' + '\t'.join(t0_vals_str))
            str_me_space.append('\t' + '\t'.join(t1_vals_str))
                
        if all_null:
            return []
        return str_me_space 
    
    def _printHamilType_12_File(self):
        ## create directory for the output
        if not os.path.exists(self.RESULT_FOLDER):
            os.mkdir(self.RESULT_FOLDER)
        
        strings_ = self._headerFileWriting()
        
        for bra, kets in self.resultsSummedUp.items():
            bra1 = readAntoine(bra[0], l_ge_10=self.l_ge_10)
            bra2 = readAntoine(bra[1], self.l_ge_10)
            ## The first step converts to the real n, l; then cast as l < 10
            bra1 = castAntoineFormat2Str(bra1, l_ge_10 = False)
            bra2 = castAntoineFormat2Str(bra2, False)
            
            for ket, T_vals in kets.items():
                ket1 = readAntoine(ket[0], self.l_ge_10)
                ket2 = readAntoine(ket[1], self.l_ge_10)
                
                ket1 = castAntoineFormat2Str(ket1, False)
                ket2 = castAntoineFormat2Str(ket2, False)
                
                spss = (bra1, bra2, ket1, ket2)
                
                strings_ += self._write2BME_JTBlock(*spss, T_vals)
        
        strings_ = '\n'.join(strings_)
        
        with open(self.filename_output + OutputFileTypes.sho, 'w+') as f:
            f.write(strings_)
            print (">> Antoine format 2body m.e. & valence space saved in [{}]"
                   .format(self.filename_output + OutputFileTypes.sho))
    
    def _write2BME_JParticleLabeledBlock(self, bra1, bra2, ket1, ket2, block_vals):
        """ 
                TODO: CHANGE to the J scheme (omit conversion)
                
        This method prints the JT block in a proton-neutron label matrix elements
        (the case of the .2b Taurus input file), block format:
        
        t_min(=0) t_max(=5) a b c d (ANTOINE states) jmin jmax
        (6 pp-pp, pn-pn, pn-np, ... columns for jmin)
        (6 pp-pp, pn-pn, pn-np, ... columns for jmin+1) ... an so on until jmax
        
        Column index:
        0: pp_pp, 1: pn_pn, 2: pn_np, 3: np_pn, 4: np_np, 5:nn_nn
        
        values come from T=0, 1 combination 
        """
        spss_str = ' '.join([bra1, bra2, ket1, ket2])
        str_me_space = []
        
        all_null = True
        
        if not block_vals:
            return []
        
        jmin, jmax = min(block_vals.keys()), max(block_vals.keys()) 
        #jmin, jmax = self._getJBoundsForT0And1Match(J_vals_T0, J_vals_T1)
        str_me_space.append(' 0 5 {} {} {}'.format(spss_str, jmin, jmax))
        
        for j in range(jmin, jmax +1):
            j_null, j_vals_str = self._formatValues2Standard(block_vals[j])
            
            all_null = all_null and j_null
            str_me_space.append('\t' + '\t'.join(j_vals_str))
        
        if all_null:
            return []
        return str_me_space 
        
    
    def _printHamilType_34_Files(self):
        ## create directory for the output
        if not os.path.exists(self.RESULT_FOLDER):
            os.mkdir(self.RESULT_FOLDER)
        
        strings_sho_ = self._headerFileWriting()
        
        # write .sho file
        del strings_sho_[2] # dismiss the sp energies
        strings_sho_.insert(1, self._hamil_type)
        _hw = self.input_obj.SHO_Parameters.get(SHO_Parameters.hbar_omega)
        strings_sho_.append('2 '+str(_hw))
        
        with open(self.filename_output + OutputFileTypes.sho, 'w+') as f:
            f.write('\n'.join(strings_sho_))
            print (">> s.h.o valence space & parameters saved in [{}]"
                   .format(self.filename_output + OutputFileTypes.sho))
        
        # write .01 file
        if self._hamil_type == '3':
            raise TBME_RunnerException("Not implemented for hamil_type=3")
        
        # write .2b file
        strings_ = [strings_sho_[0]] # title
        for bra, kets in self.resultsSummedUp.items():
            bra1 = castAntoineFormat2Str(bra[0], l_ge_10=True)
            bra2 = castAntoineFormat2Str(bra[1], True)
            
            for ket, block in kets.items():
                ket1 = castAntoineFormat2Str(ket[0], True)
                ket2 = castAntoineFormat2Str(ket[1], True)
                
                spss = (bra1, bra2, ket1, ket2)
                
                strings_ += self._write2BME_JParticleLabeledBlock(*spss, block)
        
        strings_ = '\n'.join(strings_)
        
        with open(self.filename_output + OutputFileTypes.twoBody, 'w+') as f:
            f.write(strings_)
            print (">> 2body m.e. saved in [{}]"
                   .format(self.filename_output + OutputFileTypes.twoBody))
    
    
    def printComMatrixElements(self):
        """
        Print the matrix elements for the 2 body center of mas correction
        in the corresponding scheme. First line of the file is ignored.
        """
        if not self._com_correction:
            return
        
        l_fmt = self.l_ge_10
        # write .com file
        strings_ = [self.title]
        for bra, kets in self.com_2bme.items():
            ## The first step converts to the real n, l; then cast as l < 10
            bra1 = castAntoineFormat2Str(readAntoine(bra[0], l_ge_10=l_fmt), l_fmt)
            bra2 = castAntoineFormat2Str(readAntoine(bra[1], l_fmt), l_fmt)
            
            for ket, block in kets.items():
                ket1 = castAntoineFormat2Str(readAntoine(ket[0], l_fmt), l_fmt)
                ket2 = castAntoineFormat2Str(readAntoine(ket[1], l_fmt), l_fmt)
                
                spss = (bra1, bra2, ket1, ket2)
                
                if self._hamil_type in '12':
                    strings_ += self._write2BME_JTBlock(*spss, block)
                else:
                    strings_ += self._write2BME_JParticleLabeledBlock(*spss, block)
        
        strings_ = '\n'.join(strings_)
        
        with open(self.filename_output + OutputFileTypes.centerOfMass, 'w+') as f:
            f.write(strings_)
            print (">> Center of mass correction m.e. saved in [{}]"
                   .format(self.filename_output + OutputFileTypes.centerOfMass))
        
    
    def _convertMatrixElemensScheme(self, me_dict, conv_2J, conv_2JT):
        """ 
        Converter, take dictionary of the valence space computation and change 
        to the other scheme. It crashes if the current dictionary is in that scheme.
        :conv_2J / conv_2JT explicit boolean only
        
        Input:
        {(bra1 <int>, bra2 <int> ) : {
            (ket1 <int>, ket2 <int> ) : {
              *** JT scheme block:
                T <int> =0: {J <int>: me_val,  ....},
                T <int> =1: {J <int>: me_val,  ....}
              *** J scheme block:
                J <int> (jmin): {mt <int>=0: me_val, 2: me_val, ..., 5: me_val},
                ...
                J <int> (jmax): {0: me_val, 2: me_val, ..., 5: me_val}
            }
        }
        
        
        """
        assert (type(conv_2J), type(conv_2JT)) == (bool, bool), "give me explicit boolean  >:("
        
        if not (conv_2J or conv_2JT):
            return me_dict
        
        for bra, kets in me_dict.items():
            b1 = castAntoineFormat2Str(bra[0], l_ge_10=True)
            b2 = castAntoineFormat2Str(bra[1], True)
            
            for ket, block_vals in kets.items():
                k1 = castAntoineFormat2Str(ket[0], True)
                k2 = castAntoineFormat2Str(ket[1], True)
                
                if conv_2J:
                    ## JT scheme 2 J scheme
                    J_vals_T0 = block_vals[0]
                    J_vals_T1 = block_vals[1]
                    if not (J_vals_T0 or J_vals_T1):
                        continue
                    
                    jmin, jmax = self._getJBoundsForT0And1Match(J_vals_T0, J_vals_T1)
                    
                    vals_J = {}
                    for j in range(jmin, jmax +1):
                        
                        vals_J[j] = self._getJvaluesFromIsospin(J_vals_T0[j], 
                                                                J_vals_T1[j], j,
                                                                b1, b2, k1, k2)
                    me_dict[bra][ket] = vals_J
                elif conv_2JT:
                    ## J scheme 2
                    #TODO: ("WARNING !! Converting from J scheme to JT TODO: verify")
                    
                    vals_T = {0 : {}, 1: {}}
                    for j, j_block in block_vals.items():
                        v_t0, v_t1 = self._getIsospinvaluesFromLabels(j_block, j,
                                                                      b1, b2, k1, k2)
                        vals_T[0][j] = v_t0
                        vals_T[1][j] = v_t1
                    me_dict[bra][ket] = vals_T
        
        return me_dict
        

class TBME_RunnerException(BaseException):
    pass
