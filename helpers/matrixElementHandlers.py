'''
Created on Jul 2, 2021

@author: Miguel
'''

from helpers.Helpers import prettyPrintDictionary
from helpers.io_manager import readAntoine
from copy import deepcopy

class MatrixElementFilesComparator:
    
    """ Read matrix elements from Antoine format J and T and check their values
    (script takes care of exchange sing), 
    Can return a dictionary with invalid matrix (getFailedME()) and a summary of
    the failures/miss/oks (getResults())
    
    Usage:
    test = MatrixElementFilesComparator('../results/central_SPSDPF_bench.sho', 
                                        '../results/central_SPSDPF_2.sho')
    
    _result = test.compareDictionaries()
    
    print(" === TEST RESULTS:    =================================\n")
    prettyPrintDictionary(test.getResults())
    prettyPrintDictionary(test.getFailedME())
    
    """
    
    class File:
        bench = 'bench'
        test  = 'test'
    class ME:
        diagonal = 'DIAGONAL'
        off_diag = 'OFF_DIAG'
        
    _aux_indent = {1: 'J', 2: ''}
    
    _results_initial = {
        'TOTAL': 0, 'FAIL': 0, 'PASS': 0,
        'DIAGONAL': {
            'ZERO': {
                'MISS' : 0,
                'TOTAL': 0,
                'OK':  0},
            'NON_ZERO': {
                'MISS' : 0,
                'TOTAL': 0,
                'OK':  0}
            },
        'OFF_DIAG': {
            'ZERO': {
                'MISS' : 0,
                'TOTAL': 0,
                'OK':  0},
            'NON_ZERO': {
                'MISS' : 0,
                'TOTAL': 0,
                'OK':  0}
            }
        }
    
    def __init__(self, file_bench, file_test, ignorelines=(4,4), verbose=False):
        """
        file paths to the files for bench and to test
        :ignorelines = <tuple> (<int> starting line in bench, in 2test)
        :verbose <bool>, print all the analysis
        """
        
        self._TESTING_DIAG = None
        self._verbose = verbose
        
        self._results = deepcopy(self._results_initial)
        self._failed_me = {}#{self.ME.diagonal: {}, self.ME.off_diag: {}}
        
        self.b_bench_diag = None
        self.b_bench_off  = None
        self.b_2test_diag = None
        self.b_2test_off  = None
        
        self._getJTSchemeMatrixElements(self.File.bench, file_bench, ignorelines[0])
        self._getJTSchemeMatrixElements(self.File.test, file_test, ignorelines[1])
    
    def _getBlockQN_HamilType1(self, header):
        """ 
        :header <str>  with the sp states and JT values:
               0       1       1     103     103     205       1       2
               Tmin    Tmax    st1   st2     st3     st4       Jmin    Jmax
        """ 
        header = header.strip().split()
        spss = [int(sp) for sp in header[2:6]]
        return tuple(spss), int(header[-2]), int(header[-1])
    
    def _isDiagonalMe(self, spss):
        bra, ket = sorted(spss[:2]), sorted(spss[2:])
        if tuple(bra) == tuple(ket):
            return True
        return False
    
    def _getJTSchemeMatrixElements(self, file_, filename, ignorelines=0):
        
        with open(filename, 'r') as f:
            data = f.readlines()
        
        data = data[ignorelines:]
        
        JT_block_diag, JT_block_off_diag = {}, {}
        me_states = None
        index = 0
        j_min, j_max, T = 0, 0, 0
        for line in data:
            line = line.strip()
            if index == 0:
                me_states, j_min, j_max = self._getBlockQN_HamilType1(line)
                if self._isDiagonalMe(me_states):
                    JT_block_diag[me_states] = {0: {}, 1: {}}
                    # JT_block_diag[me_states] = {}
                    diagonal = True
                else:
                    JT_block_off_diag[me_states] = {0: {}, 1: {}}
                    # JT_block_off_diag[me_states] = {}
                    diagonal = False
            else:
                T = index - 1
                
                line = line.split() 
                for j in range(j_max - j_min +1):
                    if diagonal: 
                        JT_block_diag[me_states][T][j_min + j] = float(line[j])
                    else:
                        JT_block_off_diag[me_states][T][j_min + j] = float(line[j])
            
            index = index + 1 if index < 2 else 0
        
        if file_ == self.File.bench:
            self.b_bench_diag = JT_block_diag
            self.b_bench_off  = JT_block_off_diag
        if file_ == self.File.test:
            self.b_2test_diag = JT_block_diag
            self.b_2test_off  = JT_block_off_diag
    
    
    def _countStatusFail(self, fail, value=None, missing=False):
            
        self._results['TOTAL'] += 1
        valueIsNonZero = value and abs(value) > 1.e-6
        
        if self._TESTING_DIAG:
            if valueIsNonZero:
                self._results[self.ME.diagonal]['NON_ZERO']['TOTAL'] += 1
                if missing:
                    self._results[self.ME.diagonal]['NON_ZERO']['MISS'] += 1
            else:
                self._results[self.ME.diagonal]['ZERO']['TOTAL'] += 1
        else:
            if valueIsNonZero:
                self._results[self.ME.off_diag]['NON_ZERO']['TOTAL'] += 1
                if missing:
                    self._results[self.ME.off_diag]['NON_ZERO']['MISS'] += 1
            else:
                self._results[self.ME.off_diag]['ZERO']['TOTAL'] += 1
        
        if fail:
            self._results['FAIL'] += 1
        else:
            self._results['PASS'] += 1
            if self._TESTING_DIAG:
                if valueIsNonZero:
                    self._results[self.ME.diagonal]['NON_ZERO']['OK'] += 1
                else:
                    self._results[self.ME.diagonal]['ZERO']['OK'] += 1
            else:
                if valueIsNonZero:
                    self._results[self.ME.off_diag]['NON_ZERO']['OK'] += 1
                else:
                    self._results[self.ME.off_diag]['ZERO']['OK'] += 1
    
    def _appendFailureDetails(self, spss, jt, val, val_bench):
        
        diff_abs, diff_rel = "[{}]t != [{}]b".format(val, val_bench), 'Non Zero'
        
        if val != 0:
            diff_rel = "[{:+5.4f}={:+5.3f}%]".format(val_bench/val, 
                                                     100*(val_bench/val))
        if self._verbose: 
            print(spss, jt)
            print("... Not equal:", val, "must be", val_bench)
            print("... ", diff_abs, '\t', diff_rel)
        
        if not spss in self._failed_me:
            self._failed_me[spss] = {}
        
        fail = {'abs': diff_abs, 'rel': diff_rel}
        if jt in self._failed_me[spss]:
            self._failed_me[spss][jt] = fail
        else:
            self._failed_me[spss] = {jt: fail}
    
    def getResults(self):
        return deepcopy(self._results)
    
    def getFailedME(self):
        return deepcopy(self._failed_me)
    
    def _meIndexInDict(self, spss, dict_):
        """ Return:
        :in_dict <bool>
        :keys[i] Is the coret tuple in the new dictionary
        :p      The power of -1 not the phase itself, negative if double permutation
        """
        b, k = (spss[0], spss[1]), (spss[2], spss[3])
        b_exch = (spss[1], spss[0])
        k_exch = (spss[3], spss[2])
        
        keys = [None]*8
        keys[0], keys[1] = spss, (*k, *b)                   # no   phase exch
        keys[2], keys[3] = (*b_exch, *k), (*k, *b_exch)     # bra  phase exch
        keys[4], keys[5] = (*k_exch, *b), (*b, *k_exch)     # ket_ phase exch
        keys[6], keys[7] = (*k_exch, *b_exch), (*b_exch, *k_exch) # both phase exch
        
        for i in range(8):
            if keys[i] in dict_:
                if i < 2:
                    return True, keys[i], 0
                elif i < 4:
                    p = (readAntoine(b[0], True)[2]+readAntoine(b[1], True)[2]) // 2
                    return True, keys[i], p
                elif i < 6:
                    p = (readAntoine(k[0], True)[2]+readAntoine(k[1], True)[2]) // 2
                    return True, keys[i], p
                else:
                    p = (readAntoine(b[0], True)[2]+readAntoine(b[1], True)[2]) // 2
                    p+= (readAntoine(k[0], True)[2]+readAntoine(k[1], True)[2]) // 2
                    return True, keys[i], -p
        return False, spss, 0
            
    
    def compareDictionaries(self):
        
        self._TESTING_DIAG = False
        self._compareDictionaries(self.b_bench_off,  self.b_2test_off)
        self._TESTING_DIAG = True
        self._compareDictionaries(self.b_bench_diag, self.b_2test_diag)
    
    def _compareDictionaries(self, d_bench, d_test):
        
        for spss_0, jt_block in d_bench.items():
            
            in_dict, spss, phs_pow = self._meIndexInDict(spss_0, d_test)
            
            head = spss_0 if spss_0==spss else str(spss_0)+" ->> "+str(spss)
            if self._verbose: print("\n", head)
            
            if not in_dict:
                if self._verbose: print("\n[Missing]", spss, "in m.e. to test:")
                self._countStatusFail(True, 1, missing=True)
                continue
            
            for T, j_block in jt_block.items():
                if self._verbose: print(". T=", T)
                for J, val_bench in j_block.items():
                    if self._verbose: print(".. J=", J)
                    double_perm = 1 if phs_pow < 0 else 0
                    phs = (-1)**((double_perm + 1)*(T + J) + phs_pow) if phs_pow else 0
                    val = d_test[spss][T][J]
                    if phs != 0:
                        val *= phs
                    if abs(val_bench - val) > 0.00001:
                        str_jt = "JT:{},{}".format(J,T)
                                                
                        self._appendFailureDetails(spss, str_jt, val, val_bench)
                        self._countStatusFail(True, val)
                    else:
                        if self._verbose: 
                            print("... Equal [OK]",  val, "=", val_bench, "(bench)")
                        self._countStatusFail(False, val)
    