'''
Created on Feb 23, 2021

@author: Miguel
'''
from helpers.io_manager import CalculationArgs, readAntoine,\
    castAntoineFormat2Str, valenceSpaceShellNames, ParserException
import json
import time

from helpers.Enums import InputParts as ip, AttributeArgs, OUTPUT_FOLDER,\
    SHO_Parameters, InputParts, Output_Parameters, OutputFileTypes
from matrix_elements import switchMatrixElementType
from itertools import combinations_with_replacement
from helpers.WaveFunctions import QN_1body_jj, QN_2body_jj_JT_Coupling
from copy import deepcopy
from helpers.Helpers import recursiveSumOnDictionaries, getCoreNucleus,\
    Constants
import os
import numpy as np


class TBME_Runner(object):
    '''
    The two body matrix element runner evaluate all matrix elements for a given
    valence space and Force parameters.
    '''
    NULL_TOLERANCE = 1.e-12
    PRINT_LOG = True
    
    RESULT_FOLDER = OUTPUT_FOLDER
    
    def __init__(self, filename='', manual_input={}):
        
        self.filename   = filename
        self.input_obj  = None
        self.tbme_class    = None
        self.valence_space = []
        self._hamil_type     = '1'
        self._com_correction = '0'
        
        self.results = {}
        self.resultsByInteraction = {}
        self.resultsSummedUp = {}
        
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
                self._com_correction = com_
            else:
                ParserException('COM_correction value must be an integer: 0 (off) or 1 (on)')
        
    
    def _readJsonInputFile(self):  
        print("Warning! input by json is deprecated in the program")
        
        with open(self.filename, 'r') as jf:
            data = json.load(jf)
        
        self.input_obj = CalculationArgs(data)
    
    def _readXMLInputFile(self):
        """ Read """
        import xml.etree.ElementTree as et
        
        tree  = et.parse(self.filename)
        _root = tree.getroot()
        
        self.input_obj = CalculationArgs(_root)
    
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
        # convert to l_ge_10 for the code to use
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
    
    def _computeForValenceSpaceJTCoupled(self, force=''):
        """ """
        q_numbs = getattr(self.input_obj, ip.Valence_Space)
        self.valence_space = valenceSpaceShellNames(q_numbs)
        
        q_numbs = map(lambda qn: int(qn), q_numbs)
        q_numbs = sorted(q_numbs)#, reverse=True)
        q_numbs = list(combinations_with_replacement(q_numbs, 2))
        
        count_ = 0
        total_me_ = len(q_numbs)*(len(q_numbs)+1)//2
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
                
                J_ket_min = abs(n1_ket.j - n2_ket.j) // 2
                J_ket_max = (n1_ket.j + n2_ket.j) // 2
                
                count_ += 1
                for T in (0, 1):
                    # assume M.E. cannot couple <(JT)|V|J'T'> if J', T' != J, T
                    for J in range(max(J_bra_min, J_ket_min), 
                                   min(J_bra_max, J_ket_max)+1):
                        tic = time.time()
                        
                        bra = QN_2body_jj_JT_Coupling(n1_bra, n2_bra, J, T)
                        ket = QN_2body_jj_JT_Coupling(n1_ket, n2_ket, J, T)
                                                
                        me = self.tbme_class(bra, ket)
                        self.results[q_numbs[i]][q_numbs[j]][T][J] = me.value
                        
                        if me.value and self.PRINT_LOG:
                            print(' * me[{}/{}]_({:.4}s): <{}|V|{} (J:{}T:{})> {}'
                                  .format(count_, total_me_,
                                          time.time()-tic, 
                                          bra.shellStatesNotation, 
                                          ket.shellStatesNotation, J, T, force))
                            print('\t= {:.8} '.format(me.value))                        
        
    
    def combineAllResults(self):
        
        final = {}
        for _force, results in self.resultsByInteraction.items():
            recursiveSumOnDictionaries(results, final)
        
        self.resultsSummedUp = final
                            
    
    def run(self):
        """
        Calculate all the matrix elements for all the interactions, and 
        print its combination in a file.
        """
        self._defineValenceSpaceEnergies()        
        
        _forcesAttr = ip.Force_Parameters
        times_ = {}
        for force, force_list in getattr(self.input_obj, _forcesAttr).items():
            i = 0
            for params in force_list:
                force_str = force+str(i) if len(force_list) > 1 else force
                
                tic_ = time.time()
                sho_params = getattr(self.input_obj, ip.SHO_Parameters)
                
                self.tbme_class = switchMatrixElementType(force)
                self.tbme_class.resetInteractionParameters(also_SHO=True)
                self.tbme_class.setInteractionParameters(**params, **sho_params)
                
                self._computeForValenceSpaceJTCoupled(force)
                times_[force_str] = round(time.time() - tic_, 4)
                print(" Force [{}] m.e. calculated: [{}]s"
                      .format(force, times_[force_str]))
                
                self.resultsByInteraction[force_str] = deepcopy(self.results)
                i += 1
        
        print("Finished computation, Total time (s): [", sum(times_.values()),"]=")
        print("\n".join(["\t"+str(t)+"s" for t in times_.items()]))
        self.combineAllResults()
        
        self.printMatrixElementsFile()
    
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
            
        valen = '\t' + ' '.join(aux + [x[0] for x in valence])
        energ = '\t' + ' '.join([str(x[1]) for x in valence])
                
        return valen, energ
    
    def _headerFileWriting(self):
        """ 
        Writing Header of the Antoine file: title / valence space / Energies 
        """
        
        title_args = getattr(self.input_obj, ip.Interaction_Title)
        title = '  '
        title += title_args.get(AttributeArgs.name)
        if title_args.get(AttributeArgs.details):
            title += ' :({})'.title_args.get(AttributeArgs.details)
        title += '. ME evaluated: ' + ' + '.join(
            getattr(self.input_obj, ip.Force_Parameters).keys())
        title += '. Shell({})'.format('+'.join(self.valence_space))
        
        valen, energ = self._valenceSpaceLine()
        
        core = getattr(self.input_obj, ip.Core) 
        
        # TODO: Define it in input file, how it works?
        _apply_density_correction = '0'
        
        core_args = [
            _apply_density_correction,
            core.get(AttributeArgs.CoreArgs.protons, '0'),
            core.get(AttributeArgs.CoreArgs.neutrons, '0'), 
            '0.300000', '0.000000']
        title += ' (Core: {})'.format(getCoreNucleus(*core_args[1:3]))
        core_args = '\t' + ' '.join(core_args) 
        
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
                
                if abs(mat_elem) < 1.e-6:
                    values.append("{: .3e}".format(mat_elem))
                else:
                    values.append("{: .6f}".format(mat_elem))
            else:
                try:
                    values.append("{: .6f}".format(mat_elem))
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
        
        
        
    
    def printMatrixElementsFile(self):
        
        if self._hamil_type in '12':
            # Basic Antoine Format (2 difference between neutrons-protons)
            self._printHamilType_12_File()
        elif self._hamil_type == '3':
            # J scheme 3 file output  (0, 1 & 2 body Hamiltonian_)
            # TODO: Implement
            raise TBME_RunnerException("hamil_type=3 not jet implemented")
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
            bra1 = castAntoineFormat2Str(bra[0])
            bra2 = castAntoineFormat2Str(bra[1])
            
            for ket, T_vals in kets.items():
                ket1 = castAntoineFormat2Str(ket[0])
                ket2 = castAntoineFormat2Str(ket[1])
                
                spss = (bra1, bra2, ket1, ket2)
                
                strings_ += self._write2BME_JTBlock(*spss, T_vals)
        
        strings_ = '\n'.join(strings_)
        
        with open(self.filename_output + OutputFileTypes.sho, 'w+') as f:
            f.write(strings_)
    
    def _write2BME_JParticleLabeledBlock(self, bra1, bra2, ket1, ket2, T_vals):
        """ 
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
        
        J_vals_T0 = T_vals[0]
        J_vals_T1 = T_vals[1]
        if not (J_vals_T0 or J_vals_T1):
            return []
        
        jmin, jmax = self._getJBoundsForT0And1Match(J_vals_T0, J_vals_T1)
        str_me_space.append(' 0 5 {} {} {}'.format(spss_str, jmin, jmax))
        
        for j in range(jmin, jmax +1):
            
            vals_j = self._getJvaluesFromIsospin(J_vals_T0[j], J_vals_T1[j], j,
                                                 bra1, bra2, ket1, ket2)
            
            j_null, j_vals_str = self._formatValues2Standard(vals_j)
            
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
        strings_sho_ = strings_sho_[:2] #dismiss core and sp energies
        strings_sho_.insert(1, self._hamil_type)
        _hw = self.input_obj.SHO_Parameters.get(SHO_Parameters.hbar_omega)
        strings_sho_.append(str(_hw))
        
        with open(self.filename_output + OutputFileTypes.sho, 'w+') as f:
            f.write('\n'.join(strings_sho_))
        
        # write .01 file
        if self._hamil_type == '3':
            raise TBME_RunnerException("Not implemented for hamil_type=3")
        
        # write .2b file
        strings_ = [strings_sho_[0]] # title
        for bra, kets in self.resultsSummedUp.items():
            bra1 = castAntoineFormat2Str(bra[0], l_ge_10=True)
            bra2 = castAntoineFormat2Str(bra[1], True)
            
            for ket, T_vals in kets.items():
                ket1 = castAntoineFormat2Str(ket[0], True)
                ket2 = castAntoineFormat2Str(ket[1], True)
                
                spss = (bra1, bra2, ket1, ket2)
                
                strings_ += self._write2BME_JParticleLabeledBlock(*spss, T_vals)
        
        strings_ = '\n'.join(strings_)
        
        with open(self.filename_output + OutputFileTypes.twoBody, 'w+') as f:
            f.write(strings_)
        
        
        

class TBME_RunnerException(BaseException):
    pass
