'''
Created on Feb 23, 2021

@author: Miguel
'''
from helpers.io_manager import CalculationArgs, readAntoine,\
    castAntoineFormat2Str, valenceSpaceShellNames
import json
import time

from helpers.Enums import InputParts as ip, AttributeArgs, OUTPUT_FOLDER
from matrix_elements import switchMatrixElementType
from itertools import combinations_with_replacement
from helpers.WaveFunctions import QN_1body_jj, QN_2body_jj_JT_Coupling
from copy import deepcopy
from helpers.Helpers import recursiveSumOnDictionaries, getCoreNucleus
import os


class TBME_Runner(object):
    '''
    The two body matrix element runner evaluate all matrix elements for a given
    valence space and Force parameters.
    '''
    NULL_TOLERANCE = 1.e-12
    PRINT_LOG = True
    
    RESULT_FOLDER = OUTPUT_FOLDER
    
    def __init__(self, filename='', manual_input={}):
        
        self.filename = filename
        self.input_obj  = None
        self.tbme_class    = None
        self.valence_space = []
        
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
        self.filename_output  = self.RESULT_FOLDER +'/'+ self.input_obj.getFilename()
        
        
    def _computeForValenceSpaceJTCoupled(self, force=''):
        """ """
        q_numbs = getattr(self.input_obj, ip.Valence_Space)
        self.valence_space = valenceSpaceShellNames(q_numbs)
        
        q_numbs = map(lambda qn: int(qn), q_numbs)
        q_numbs = sorted(q_numbs)#, reverse=True)
        q_numbs = list(combinations_with_replacement(q_numbs, 2))
        
        count_ = 0
        for i in range(len(q_numbs)):
            n1_bra = QN_1body_jj(*readAntoine(q_numbs[i][0]))
            n2_bra = QN_1body_jj(*readAntoine(q_numbs[i][1]))
            
            self.results[q_numbs[i]] = {}
            
            J_bra_min = abs(n1_bra.j - n2_bra.j) // 2
            J_bra_max = (n1_bra.j + n2_bra.j) // 2
            
            for j in range(i, len(q_numbs)):
                n1_ket = QN_1body_jj(*readAntoine(q_numbs[j][0]))
                n2_ket = QN_1body_jj(*readAntoine(q_numbs[j][1]))
                
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
                                  .format(count_, 
                                          len(q_numbs)*(len(q_numbs)+1)//2,
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
        # TODO: Read the self.input_obj.Fore_Parameters and implement the 
        # parameters in the Matrix Element Class.
        _forcesAttr = ip.Force_Parameters
        times_ = {}
        for force, force_list in getattr(self.input_obj, _forcesAttr).items():
            i = 0
            for params in force_list:
                force_str = force+str(i) if len(force_list) > 1 else force
                
                tic_ = time.time()
                sho_params = getattr(self.input_obj, ip.SHO_Parameters)
                
                self.tbme_class = switchMatrixElementType(force)
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
            
    def _headerFileWriting(self):
        """ Writing Header of the Antoine file: title / valence space / Energies """
                
        title_args = getattr(self.input_obj, ip.Interaction_Title)
        title = '  '
        title += title_args.get(AttributeArgs.name)
        if title_args.get(AttributeArgs.details):
            title += ' :({})'.title_args.get(AttributeArgs.details)
        title += '. ME evaluated: ' + ' + '.join(
            getattr(self.input_obj, ip.Force_Parameters).keys())
        title += '. Shell({})'.format('+'.join(self.valence_space))
        
        valence = getattr(self.input_obj, ip.Valence_Space)
        valence = sorted([tuple(x) for x in valence.items()], key=lambda x: x[1])
        valen = '\t' + ' '.join(['1', str(len(valence))] + [x[0] for x in valence])
        
        energ = '\t' + ' '.join([str(x[1]) for x in valence])
        
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
            jmin, jmax = min(J_vals_T0), max(J_vals_T0)
            assert jmin == min(J_vals_T1), TBME_RunnerException("T0 T1 jmin doesn't match:",
                                                                J_vals_T0, J_vals_T1)
            assert jmax == max(J_vals_T1), TBME_RunnerException("T0 T1 jmax doesn't match",
                                                                J_vals_T0, J_vals_T1)
            
            str_me_space.append(' 0 1 {} {} {}'.format(spss_str, jmin, jmax))
            
            t0_null, t0_vals_str = self._formatValues2Standard(J_vals_T0)
            t1_null, t1_vals_str = self._formatValues2Standard(J_vals_T1)
            
            all_null = t0_null and t1_null
            str_me_space.append('\t' + '\t'.join(t0_vals_str))
            str_me_space.append('\t' + '\t'.join(t1_vals_str))
                
        if all_null:
            return []
        return str_me_space 
                
    def printMatrixElementsFile(self):
        

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
        
        with open(self.filename_output, 'w+') as f:
            f.write(strings_)
        
        
        
        

class TBME_RunnerException(BaseException):
    pass
