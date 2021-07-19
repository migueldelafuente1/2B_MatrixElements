'''
Created on Jul 2, 2021

@author: Miguel

This script read a hamilType 1 matrix elements file from bench to compare with
one generated from the program.

(14/7/2021) "compareDictionaries_version1" is deprecated but still more general, 
    it is now fixed for an specific prepared block of T common matrix elements:
        {single particle state: {J: value, ...}, ...} (Asummed T=1 for LS m.e.)
    to be used for more general matrix elements require a fix in the reading and
    another loop for the "compareDictionaries" method (or just group the 
    dictionaries in T=0 block and T=1) 
'''
from helpers.Helpers import prettyPrintDictionary
from helpers.io_manager import readAntoine


def _getBlockQN_HamilType1(header):
    """ 
    :header <str>  with the sp states and JT values:
           0       1       1     103     103     205       1       2
           Tmin    Tmax    st1   st2     st3     st4       Jmin    Jmax
    """ 
    header = header.strip().split()
    spss = [int(sp) for sp in header[2:6]]
    return tuple(spss), int(header[-2]), int(header[-1])

def _isDiagonalMe(spss):
    bra, ket = sorted(spss[:2]), sorted(spss[2:])
    if tuple(bra) == tuple(ket):
        return True
    return False

def _getJTSchemeMatrixElements(filename, ignorelines=0):
    
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
            me_states, j_min, j_max = _getBlockQN_HamilType1(line)
            if _isDiagonalMe(me_states):
                # JT_block_diag[me_states] = {0: {}, 1: {}}
                JT_block_diag[me_states] = {}
                diagonal = True
            else:
                # JT_block_off_diag[me_states] = {0: {}, 1: {}}
                JT_block_off_diag[me_states] = {}
                diagonal = False
        else:
            T = index - 1
            if T == 1:
                line = line.split() 
                for j in range(j_max - j_min +1):
                    if diagonal: 
                        JT_block_diag[me_states][j_min + j] = float(line[j])
                    else:
                        JT_block_off_diag[me_states][j_min + j] = float(line[j])
        
        
        index = index + 1 if index < 2 else 0
    
    return JT_block_diag, JT_block_off_diag

aux_indent = {1: 'J', 2: ''}

_results = {
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

__TESTING_DIAG = False

def _countStatusFail(fail, value=None, missing=False):
    global _results, __TESTING_DIAG
        
    _results['TOTAL'] += 1
    valueIsNonZero = value and abs(value) > 1.e-6
    
    if __TESTING_DIAG:
        if valueIsNonZero:
            _results['DIAGONAL']['NON_ZERO']['TOTAL'] += 1
            if missing:
                _results['DIAGONAL']['NON_ZERO']['MISS'] += 1
        else:
            _results['DIAGONAL']['ZERO']['TOTAL'] += 1
    else:
        if valueIsNonZero:
            _results['OFF_DIAG']['NON_ZERO']['TOTAL'] += 1
            if missing:
                _results['OFF_DIAG']['NON_ZERO']['MISS'] += 1
        else:
            _results['OFF_DIAG']['ZERO']['TOTAL'] += 1
    
    
    if fail:
        _results['FAIL'] += 1
    else:
        _results['PASS'] += 1
        if __TESTING_DIAG:
            if valueIsNonZero:
                _results['DIAGONAL']['NON_ZERO']['OK'] += 1
            else:
                _results['DIAGONAL']['ZERO']['OK'] += 1
        else:
            if valueIsNonZero:
                _results['OFF_DIAG']['NON_ZERO']['OK'] += 1
            else:
                _results['OFF_DIAG']['ZERO']['OK'] += 1


def _meIndexInDict_version1(spss, dict_):
    """ Return:
    :in_dict <bool>
    :keys[i] Is the coret tuple in the new dictionary
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
            return True, keys[i]
    return False, spss

def _meIndexInDict(spss, dict_):
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
        
def compareDictionaries_version1(dict1, dict2, level=0):
    """ 
    This version only go for T=1, 
    dict1 : dictionary of bench
    dict2 : dictionary to test
    """
    indent = ''.join(['.']*level)
    aux = '\n' if level < 1 else aux_indent[level]
    if type(dict1) != type(dict1):
        print(indent, "Error, ",  type(dict1), type(dict2))
        
    if isinstance(dict1, dict):
        for k, val in dict1.items():
            if isinstance(k,tuple):
                in_dict, k_correct = _meIndexInDict(k, dict2)
                if in_dict:
                    print(indent, aux, k_correct)
                    compareDictionaries(val, dict2[k_correct], level+1)
                    continue
            elif isinstance(k, int):
                if k in dict2:
                    print(indent, aux, k)
                    compareDictionaries(val, dict2[k], level+1)
                    continue
            
            print("\n[Missing]", k, "in m.e. to test:")
            prettyPrintDictionary(val)
            _countStatusFail(False)
    else:
        
        if abs(dict1 - dict2) > 0.00001:
            print(indent, "Not equal:", dict2, "must be", dict1)
            if dict1 != 0:
                print(indent, "[{:+5.4f}={:+5.3f}%]".format(dict2/dict1, 
                                                            100*((dict2/dict1)-1)))
                    
            _countStatusFail(True, dict1)
        else:
            #print(indent, "Equal [OK]", dict1)
            _countStatusFail(False, dict1)
        

def compareDictionaries(d_bench, d_test):
    
    for spss_0, j_block in d_bench.items():
        
        in_dict, spss, phs_pow = _meIndexInDict(spss_0, d_test)
        
        head = spss_0 if spss_0==spss else str(spss_0)+" ->> "+str(spss)
        print("\n", head)
        
        if not in_dict:
            print("\n[Missing]", spss, "in m.e. to test:")
            _countStatusFail(True, 1, missing=True)
            continue
            
        for J, val_bench in j_block.items():
            print(". J=", J)
            double_perm = 1 if phs_pow < 0 else 0
            phs = (-1)**((double_perm + 1)*(1 + J) + phs_pow) if phs_pow else 0
            val = d_test[spss][J]
            if phs != 0:
                val *= phs
            if abs(val_bench - val) > 0.00001:
                print(".. Not equal:", val, "must be", val_bench)
                if val != 0:
                    print(".. [{:+5.4f}={:+5.3f}%]".format(val_bench/val, 
                                                           100*(val_bench/val)))
                _countStatusFail(True, val)
            else:
                print(".. Equal [OK]",  val, "=", val_bench, "(bench)")
                _countStatusFail(False, val)
            
            

if __name__ == '__main__':
    
    
    b_bench_diag, b_bench_off = _getJTSchemeMatrixElements('LS_me_BLC_test_JT.2b', ignorelines=2)
    b_2test_diag, b_2test_off = _getJTSchemeMatrixElements('../results/ls_SPSD.sho', ignorelines=4)
    
    __TESTING_DIAG = False
    compareDictionaries(b_bench_off, b_2test_off)
    __TESTING_DIAG = True
    compareDictionaries(b_bench_diag, b_2test_diag)
    
    print(" === TEST RESULTS:    =================================\n")
    prettyPrintDictionary(_results)
    # print("\nFail [{f}/{tot}] OK[{ok}/{tot}]".format(tot=TOTAL, ok=PASS, f=FAIL))
    # print("Diagonal matrix elements: [{ok}/{tot}]".format(ok=_DIAGONAL_OK, tot=_DIAGONAL))
    # print("Non Diag matrix elements: [{ok}/{tot}]".format(ok=_OFF_DIAG_OK, tot=_OFF_DIAG))