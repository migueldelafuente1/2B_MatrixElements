'''
Created on 10 abr 2025

@author: delafuente
'''
from helpers import MATPLOTLIB_INSTALLED
from helpers.Helpers import readAntoine
if MATPLOTLIB_INSTALLED: 
    import matplotlib.pyplot as plt

if __name__ == '__main__':
    FLD   = '../results/'
    file_ = FLD + 'D1S_LS_MZ7.com'
    FLD   = '../savedHamilsBeq1/'
    file_ = FLD + 'COM_MZ10.com'
    with open(file_, 'r') as f:
        data = f.readlines()[1:]
    
    qqnn = {}
    me_pppp = {}
    me_pnpn = {}
    me_pnnp = {}
    me_nnnn = {}
    
    sets_me = [[], [], [], []]
    for i, line in enumerate(data):
        line = line.strip()
        if line.startswith('0 5'):
            _, _, a, b, c, d, jmin, jmax = line.split()
            jmin, jmax = int(jmin), int(jmax)
            tpl_  = (a, b, c, d)
            tpl_  = tuple([int(x) for x in tpl_])
            
            tpl_N = [readAntoine(x, l_ge_10=True) for x in tpl_]
            tpl_N = [2*x[0] + x[1]  for x in tpl_N]
            assert not tpl_ in qqnn.keys(), f"tpl={tpl_} already read" 
            qqnn[ tpl_ ] = (jmin, jmax)
            J = jmin
            me_pppp[tpl_] = []
            me_pnpn[tpl_] = []
            me_pnnp[tpl_] = []
            me_nnnn[tpl_] = []
            continue
        pppp, pnpn, pnnp, _, _, nnnn = line.split()
        if 'e' in pppp or 'd' in pppp: 
            if 9 in tpl_N or 10 in tpl_N: continue
            str_ = "" if list(filter(lambda x: x > 6, tpl_N)) != [] else "**(N<7)"
            print(f" J[{J:2}]"+ "  ({:5} {:5} {:5} {:5})".format(*tpl_)
                  +" [{:2},{:2},{:2},{:2}]  {}".format(*tpl_N, str_) )
        pppp, pnpn, pnnp, nnnn = float(pppp), float(pnpn), float(pnnp), float(nnnn)
        me_pppp[tpl_].append(pppp)
        me_pnpn[tpl_].append(pnpn)
        me_pnnp[tpl_].append(pnnp)
        me_nnnn[tpl_].append(nnnn)
        
        sets_me[0].append(pppp)
        sets_me[1].append(pnpn)
        sets_me[2].append(pnnp)
        sets_me[3].append(nnnn)
        
        assert J <= jmax, f"J[{J}] exceed Jmax={jmax}"
        J += 1
    
    print(" >> Importing done:", len(qqnn), len(sets_me[0]))
    for i in range(4):
        print(f"  >> min, max values [{i}]", min(sets_me[i]), max(sets_me[i]))
    
    fig, ax = plt.subplots(2, 2)
    x = [i for i in range(len(sets_me[0]))]
    ax[0,0].scatter(x, sets_me[0])
    ax[0,1].scatter(x, sets_me[3])
    ax[1,0].scatter(x, sets_me[1])
    ax[1,1].scatter(x, sets_me[2])
    
    # ax[0,0].hist(sets_me[0], bins=1000)
    # ax[0,1].hist(sets_me[3], bins=1000)
    # ax[1,0].hist(sets_me[1], bins=1000)
    # ax[1,1].hist(sets_me[2], bins=1000)
    # k = 0
    # for tpl_, jrange in qqnn.items():
    #     i = 0
    #     sets_ = [[], [], [], []]
    #     for J in range(jrange[0], jrange[1]+1):
    #
    #         pppp = me_pppp[tpl_][i]
    #         pnpn = me_pnpn[tpl_][i]
    #         pnnp = me_pnnp[tpl_][i]
    #         nnnn = me_nnnn[tpl_][i]
    #
    #         sets_[0].append(pppp)
    #         sets_[1].append(pnpn)
    #         sets_[2].append(pnnp)
    #         sets_[3].append(nnnn)
    #
    #         i += 1
    #
    #     x = [k,]*i
    #     ax[0,0].scatter(x, sets_[0])
    #     ax[0,1].scatter(x, sets_[3])
    #     ax[1,0].scatter(x, sets_[1])
    #     ax[1,1].scatter(x, sets_[2])
    #
    #     k += 1
    #     if int(100 * len(qqnn) / k) % 10:
    #         print(f"  * done scatter [{k}/{len(qqnn)}]")
        
        
    ax[0,0].set_title('<pp-pp>')
    ax[0,1].set_title('<nn-nn>')
    ax[1,0].set_title('<pn-pn>')
    ax[1,1].set_title('<pn-np>')
    plt.show()