'''
Created on Jul 2, 2021

@author: Miguel
'''

from helpers.Helpers import prettyPrintDictionary, shell_filling_order,\
    shell_filling_order_ge_10
from helpers.io_manager import readAntoine
from copy import deepcopy
from matplotlib.pyplot import tight_layout


class MatrixElementFilesComparator:
    
    """ Read matrix elements from Antoine format J and T and check their values
    (script takes care of exchange sing), 
    Can return a dictionary with invalid matrix (getFailedME()), a summary of
    the failures/miss/oks (getResults()) and a the list of bench/test missing
    elements with getMissingME().
    
    Usage:
    test = MatrixElementFilesComparator('../results/central_SPSDPF_bench.sho', 
                                        '../results/central_SPSDPF_2.sho')
    
    test.compareDictionaries()
    
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
    
    def __init__(self, file_bench, file_test, 
                 ignorelines=(4,4), l_ge_10=True, verbose=False):
        """
        file paths to the files for bench and to test
        :ignorelines = <tuple> (<int> starting line in bench, in 2test)
        :verbose <bool>, print all the analysis
        """
        
        self._TESTING_DIAG = None
        self._verbose = verbose
        self.l_ge_10 = l_ge_10
        
        self._results = deepcopy(self._results_initial)
        self._passed_me = []
        self._failed_me = {}#{self.ME.diagonal: {}, self.ME.off_diag: {}}
        self._failed_me_num = {}
        self._missing_me = {'in_'+self.File.bench: [],
                            'in_'+self.File.test: []}
        
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
        
        diff_abs_str ="[{}]t != [{}]b".format(val, val_bench)
        rel_str, diff_rel_str = 'Non Zero', 'Non Zero'
        diff_abs, rel, diff_rel = 0, 1, 1
        
        if val_bench != 0:
            rel =  val/val_bench
            diff_abs = val - val_bench
            diff_rel = diff_abs / val_bench
            rel_str = "[{:+5.4f}={:+5.3f}%]".format(rel, 100*rel)
            diff_rel_str = "[{:+5.4e}={:+5.3f}%]".format(diff_abs, 100*diff_rel)
            
        if self._verbose: 
            print(spss, jt)
            print("... Not equal:", val, "must be", val_bench)
            print("... ", diff_abs, '\t', diff_rel, '\t rel=', rel)
        
        if not spss in self._failed_me:
            self._failed_me[spss] = {}
            self._failed_me_num[spss] = {}
        
        fail = {'abs': diff_abs_str, 'rel': rel_str, 'diff_rel': diff_rel_str}
        fail_num = {'abs': diff_abs, 'rel': rel, 'diff_rel': diff_rel}
        if jt in self._failed_me[spss]:
            self._failed_me[spss][jt] = fail
            self._failed_me_num[spss][jt] = fail_num
        else:
            self._failed_me[spss] = {jt: fail}
            self._failed_me_num[spss] = {jt: fail_num}
    
    def _appendMissingDetails(self, in_file, spss):
        
        self._missing_me['in_'+in_file].append(spss)            
        
    def getResults(self):
        return deepcopy(self._results)
    
    def getFailedME(self):
        return deepcopy(self._failed_me)
    
    def getMissingME(self):
        return deepcopy(self._missing_me)
    
    def plotFailedDifferences(self):
        import numpy as np
        import matplotlib
        #matplotlib.rcParams['text.usetex'] = True
        import matplotlib.pylab as plt
        
        rel = []
        x = []
        index = {}
        i = 0
        for spss, val_jt in self._failed_me_num.items():
            # if not(10001 in spss):
            #     continue
            #     if spss.count(10001) >= 1:
            #         continue
            for jt, val in val_jt.items():
                # if abs(val['diff_rel']) < 0.4:
                #     continue
                rel.append(val['diff_rel'])
                x.append(i)
                index[i] = "{}\t {}\t = {:.6f}".format(spss, jt, rel[-1])
                i += 1
        
        
        prettyPrintDictionary(index)
        
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        ax.grid(True)
        ticks = max(float(len(x) // 30), 1.0)
        ax.set_xticks(np.arange(min(x), max(x)+1, ticks))
        ax.scatter(x, rel)
        # ax.set_ylabel(r"$\frac{me - me_{bench}}{me_{bench}}$")
        ax.set_ylabel(r"(x - bench) / bench")
        ax.set_title("TBME relative differences")
        #plt.show()
        
        fig2, ax2 = plt.subplots(tight_layout=True)
        n, bins, patches = plt.hist(x=rel, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel(r"(x - bench) / bench")
        plt.ylabel('Frequency')
        plt.title('Distribution of difference')
        plt.show()
        
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
        
        lge10 = self.l_ge_10
        
        for i in range(8):
            if keys[i] in dict_:
                if i < 2:
                    return True, keys[i], 0
                elif i < 4:
                    p = (readAntoine(b[0],lge10)[2]+readAntoine(b[1],lge10)[2]) // 2
                    return True, keys[i], p
                elif i < 6:
                    p = (readAntoine(k[0],lge10)[2]+readAntoine(k[1],lge10)[2]) // 2
                    return True, keys[i], p
                else:
                    p = (readAntoine(b[0],lge10)[2]+readAntoine(b[1],lge10)[2]) // 2
                    p+= (readAntoine(k[0],lge10)[2]+readAntoine(k[1],lge10)[2]) // 2
                    return True, keys[i], -p
        return False, spss, 0
            
    
    def compareDictionaries(self):
        
        self._TESTING_DIAG = False
        self._compareDictionaries(self.b_bench_off,  self.b_2test_off)
        self._TESTING_DIAG = True
        self._compareDictionaries(self.b_bench_diag, self.b_2test_diag)
    
    def _compareDictionaries(self, d_bench, d_test):
        
        for spss_0, jt_block in d_bench.items():
            if spss_0 == (1,1,1,1):
                _=0
            in_dict, spss, phs_pow = self._meIndexInDict(spss_0, d_test)
            
            head = spss_0 if spss_0==spss else str(spss_0)+" ->> "+str(spss)
            if self._verbose: print("\n", head)
            
            if not in_dict:
                if self._verbose: print("\n[Missing]", spss, "in m.e. to test:")
                self._appendMissingDetails(self.File.test, spss)
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
                    # if the difference is 0.1% greater than the bench value
                    if abs(val_bench - val) > abs(0.00000001 * val_bench):
                        str_jt = "JT:{},{}".format(J,T)
                                                
                        self._appendFailureDetails(spss, str_jt, val, val_bench)
                        self._countStatusFail(True, val)
                    else:
                        if self._verbose: 
                            print("... Equal [OK]",  val, "=", val_bench, "(bench)")
                        self._countStatusFail(False, val)
                        self._passed_me.append(spss)
        
        for spss_t in d_test.keys():
            if spss_t not in d_bench:
                self._appendMissingDetails(self.File.bench, spss_t)
                self._countStatusFail(True, 1, missing=True)
        
#===============================================================================
# PLOTING M.E (T channel) to be compared
#===============================================================================~

class MatrixElements_PlotComparator():
    
    """ 
    This class compare several matrix element interactions in the JT scheme 
    by plotting the T=0/1 TBME energy vs TBME (s1,s2,s3,s4) J index
    
    Kwargs: force name = filepath
    
    Example:
        MatrixElements_PlotComparator(
            Central_gaussian1 = '../results/central_g1.sho',
            Central_gaussian2 = '../results/central_g2.sho', ... )
        
    
    """
    
    def __init__(self, **files):
        
        self.l_ge_10 = True
        
        self.tbms    = {}
        self.tbms_T0 = {}
        self.tbms_T1 = {}
        
        self.valenceSpace = []
        
        for force, filepath in files.items():
            self._getJTSchemeMatrixElements(force, filepath, ignorelines=4)
            self.tbms_T0[force] = {}
            self.tbms_T1[force] = {}
        
        self._fillAndRemoveEmptyMatrixElements()
        
        if (len(self.tbms_T0) == 0) and (len(self.tbms_T1) == 0):
            raise Exception("no files given or files are empty, check: " + 
                            ", ".join(files.values()))
    
    def _getBlockQN_HamilType1(self, header):
        """ 
        :header <str>  with the sp states and JT values:
               0       1       1     103     103     205       1       2
               Tmin    Tmax    st1   st2     st3     st4       Jmin    Jmax
        """ 
        header = header.strip().split()
        spss = [int(sp) for sp in header[2:6]]
        return tuple(spss), int(header[-2]), int(header[-1])
    
    def _checkValenceSpace(self, valence_space_line):
        """ Verify if the matrix elements were obtained over the same valence 
        space """
        valence_space_line = valence_space_line.split()[2:]
        
        if len(self.valenceSpace) == 0:
            # if '1' in valence_space_line:
            #     valence_space_line[valence_space_line.index('1')] = '001'
            self.valenceSpace = [int(st) for st in valence_space_line]
        else:
            in_ = [(st, int(st) in self.valenceSpace) for st in valence_space_line]
            not_in = dict(filter(lambda x: not x[1], in_))
            
            for st in not_in.keys():
                # if st == '1' and (1 in self.valenceSpace):
                #     continue
                raise Exception('state/s {} not in previous valence space: {}'
                                .format(list(not_in.keys()), self.valenceSpace))
    
    def _getJTSchemeMatrixElements(self, force, filename, ignorelines=0):
        
        with open(filename, 'r') as f:
            data = f.readlines()
        self._checkValenceSpace(data[1])
        
        data = data[ignorelines:]
        
        JT_block = {0:{}, 1:{}}
        me_states = None
        index = 0
        j_min, j_max, T = 0, 0, 0
        for line in data:
            line = line.strip()
            if index == 0:
                me_states, j_min, j_max = self._getBlockQN_HamilType1(line)
                JT_block[0][me_states] = {}
                JT_block[1][me_states] = {}
            else:
                T = index - 1
                
                line = line.split() 
                for j in range(j_max - j_min +1):
                    JT_block[T][me_states][j_min + j] = float(line[j])
            
            index = index + 1 if index < 2 else 0
        
        self.tbms[force] = JT_block
    
    def _statesOrderedByEnergy(self):
        """ 
        Return the states in the order of shell_filling_order, give also j 
        """
        valenceSpaceOrdered = []
        for st, tpl, _ in shell_filling_order_ge_10:
            st = int(st)
            # if self.l_ge_10:
            #     st_le_10 = 10000*(st//1000) + (st%1000)
            # if st_le_10 in self.valenceSpace:
            if st in self.valenceSpace:
                valenceSpaceOrdered.append((int(st), tpl[2]))
                if len(self.valenceSpace) == len(valenceSpaceOrdered):
                    break
        
        self.valenceSpace = [st[0] for st in valenceSpaceOrdered]
        
        from itertools import combinations_with_replacement
        
        aux = [i for i in combinations_with_replacement(valenceSpaceOrdered, 2)]
        spss = []
        for i in range(len(aux)):
            bra = aux[i]
            for j in range(i, len(aux)):
                ket = aux[j]
                spss.append((*bra, *ket))
        
        j_lims = []
        for i in range(len(spss)):
            st_tpl = spss[i]
            s1, s2, s3, s4 = st_tpl
            jmin = max(abs(s1[1] - s2[1]), abs(s3[1] - s4[1])) // 2
            jmax = min(s1[1] + s2[1], s3[1] + s4[1]) // 2
            
            j_lims.append(tuple(range(jmin, jmax + 1)))
            spss[i] = (s1[0], s2[0], s3[0], s4[0])
        
        return zip(spss, j_lims)
    
    def _fillAndRemoveEmptyMatrixElements(self):
        """ 
        Null matrix elements might been removed form the file, assign 0 if 
        appear non-zero value in the other forces, remove an element if it's zero
        in all the files.
        """
        self.forces = self.tbms.keys()
            
        for spss, j_range in self._statesOrderedByEnergy():
            self._j_range = j_range
            all_null = True
            for force in self.forces:
                in_ = self._meInForce(force, spss)
                if in_:
                    all_null = False
            
            self._clearEmptyMatrixElement(spss, all_null, j_range)
    
    def _fillEmptyNewMatrixElement(self, force, new_spss):
        
        self.tbms[force][0][new_spss] = dict([(j, 0.0) for j in self._j_range])
        self.tbms[force][1][new_spss] = dict([(j, 0.0) for j in self._j_range])
        self.tbms_T0[force][new_spss] = dict([(j, 0.0) for j in self._j_range])
        self.tbms_T1[force][new_spss] = dict([(j, 0.0) for j in self._j_range])
    
    def _clearEmptyMatrixElement(self, spss, all_null, j_range): 
        """ 
        Remove the matrix element entry if all was null, if there are all J null
        values for fixed T, then remove the J entry. 
        """
        if all_null:
            for force in self.forces:
                del self.tbms[force][0][spss]
                del self.tbms[force][1][spss]
                del self.tbms_T0[force][spss]
                del self.tbms_T1[force][spss]
            return
        ## read j by j for the element and remove empty J
        for j in j_range:
            jT_all_null = [True, True]
            
            for force in self.forces:
                for T in (0, 1):
                    if abs(self.tbms[force][T][spss][j]) > 1e-10:
                        jT_all_null[T] = False
                # if not True in jT_all_null: break
            for force in self.forces:
                if jT_all_null[0]:
                    del self.tbms[force][0][spss][j]
                    del self.tbms_T0[force][spss][j]
                if jT_all_null[1]:
                    del self.tbms[force][1][spss][j]
                    del self.tbms_T1[force][spss][j]
    
    def _getSinglePartJ(self, twoBodyWF):
        _, _, j1 = readAntoine(twoBodyWF[0], l_ge_10=self.l_ge_10)
        _, _, j2 = readAntoine(twoBodyWF[1], l_ge_10=self.l_ge_10)
        return j1, j2
        
    def _meInForce(self, force, spss):
        """ 
        Search the matrix element if its states are exchanged, MOVE it to the
        standard spss for the program to read (remove the old entry in self.tbms)
        return if the new spss has been found.
        """
        bra, ket  =  spss[:2], spss[2:]
        b_exch, k_exch  =  (bra[1], bra[0]), (ket[1], ket[0])
        bra_diag, ket_diag = (bra == b_exch), (ket == k_exch)
        
        keys = [None]*7
        keys[0] = (*ket, *bra)                               # no   phase exch
        keys[1], keys[2] = (*b_exch, *ket), (*ket, *b_exch)  # bra  phase exch
        keys[3], keys[4] = (*k_exch, *bra), (*bra, *k_exch)  # ket_ phase exch
        keys[5], keys[6] = (*k_exch, *b_exch), (*b_exch, *k_exch) # both  exch
        
        #if spss == (1, 103, 103, 205):
        _ = 0
        # Don't create a matrix entry for the force if the element already exist
        if spss in self.tbms[force][0]: #if it is in T=0 is also in T=1
            self.tbms_T0[force][spss], self.tbms_T1[force][spss] = {}, {}
            for j in self._j_range:
                self.tbms_T0[force][spss][j] = self.tbms[force][0][spss][j]
                self.tbms_T1[force][spss][j] = self.tbms[force][1][spss][j]
            return True
        # Create empty matrix element for the force
        self._fillEmptyNewMatrixElement(force, spss)
        in_ = False
        if (bra == ket) and (bra_diag and ket_diag):
            return False
        for i in range(7):
            # Search in possibilities
            if keys[i] in self.tbms[force][0]: #if it is in T=0 is also in T=1
                if (i < 1):
                    self._mapNewMatrixElement(force, spss, keys[i], 0, False) 
                elif (i < 3):
                    if bra_diag: continue
                    p = 1 + (sum(self._getSinglePartJ(b_exch)) // 2)
                    self._mapNewMatrixElement(force, spss, keys[i], p, True)
                elif (i < 5):
                    if ket_diag: continue
                    p = 1 + (sum(self._getSinglePartJ(k_exch)) // 2)
                    self._mapNewMatrixElement(force, spss, keys[i], p, True)
                else:
                    if (bra_diag and ket_diag): continue
                    p = (sum([*self._getSinglePartJ(b_exch),
                              *self._getSinglePartJ(k_exch)])) // 2
                    self._mapNewMatrixElement(force, spss, keys[i], p, False)
                in_ = True
            if in_:
                return in_
        return in_
    
    def _mapNewMatrixElement(self, force, spss, old_spss, phs_exp, single_perm):
        """ 
        Apply the permutation in the new spss and remove the old_sp state 
        dictionary entry.
        :sigle_perm = True, add the T+J to the exponent (even permutation dropp it)
        """
        for j in self._j_range:
            
            phs0, phs1 = (-1)**(phs_exp), (-1)**(phs_exp)
            if single_perm:
                phs0 *= (-1)**(j)
                phs1 *= (-1)**(j + 1)
            
            self.tbms[force][0][spss][j] = phs0 * self.tbms[force][0][old_spss][j]
            self.tbms[force][1][spss][j] = phs1 * self.tbms[force][1][old_spss][j]
            self.tbms_T0[force][spss][j] = phs0 * self.tbms[force][0][old_spss][j]
            self.tbms_T1[force][spss][j] = phs1 * self.tbms[force][1][old_spss][j]
        
        # if old_spss != spss:
        # diagonal matrix elements <aa |V| aa> would be removed otherwise
        del self.tbms[force][0][old_spss]
        del self.tbms[force][1][old_spss]
        # self.tbms_T0[force][old_spss] do not exist
        
            
if __name__ == "__main__":
    
    test = MatrixElementFilesComparator(
        # '../results/dd_a1_b15_A16.2b', 
        '../results/Yukawa_analitic_3.sho',
        '../results/Yukawa_approx13g.sho', verbose=False)
        #'../results/kin2_bench.com', 
        #'../results/kin2.com', 
        # verbose=False)
    
    test.compareDictionaries()
    print(" === TEST RESULTS:    =================================\n")
    prettyPrintDictionary(test.getResults())
    # prettyPrintDictionary(test.getFailedME())
    test.plotFailedDifferences()
    prettyPrintDictionary(test.getMissingME())
    
    print("Missing in test file (but parity conserv)")
    miss = test.getMissingME()
    
    def conservesParity(spss):
        spss2 = [readAntoine(i, l_ge_10=True) for i in spss]
        ll = [sp[1] for sp in spss2]
        if (ll[0] + ll[1]) % 2 == (ll[2] + ll[3]) % 2:
            return True, ll
        return False, ll
    
    for spss in miss['in_test']:
        parityOk, ll = conservesParity(spss)
        if parityOk:
            Lb_min, Lb_max = abs(ll[0] - ll[1]), ll[0] + ll[1]
            Lk_min, Lk_max = abs(ll[2] - ll[3]), ll[2] + ll[3]
            print(spss, 'L_bra', list(range(Lb_min, Lb_max+1)), 
                        'L_ket', list(range(Lk_min, Lk_max+1)))
    
    print("\nMissing in bench file (non zero in our TBME)", len(miss['in_bench']))
    for i, spss in enumerate(miss['in_bench']):
        
        parityOk, ll = conservesParity(spss)
        if parityOk:
            print(i, spss)
        else:
            print(i, "WRONG PARITY", spss)
    
    # plot_ = MatrixElements_PlotComparator(
    #             shortRange_LS='../results/ls_short_SPSD.sho',
    #             bb='../results/bb_SPSD.sho',
    #             dd_fermi='../results/density_SPSD.sho')
    