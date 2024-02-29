'''
Created on 21 nov 2023

@author: delafuente

Import two sets of matrix elments J scheme, read all the matrix elements, find 
coincident 
'''
from copy import deepcopy
from debugpy.launcher import channel
def get_qqnn_antoine(qqnn):
    
    n = qqnn // 10000
    l = (qqnn % 10000) // 100
    j = (qqnn % 10000) % 100  
    return n,l,j

def matrixReaderAndSorter(file_, states_order=None):
    """
    states_order = list [ <int> <int>, ...] in the order to export bra-kets
    """
    
    mmee = {}
    j_tops = [1000, 0]
    
    with open(file_, 'r') as f:
        data = f.readlines()[1:]
        
        bra_, ket_ = None, None
        for line in data:
            line = line.strip()
            if line.startswith('0 5'):
                _,_,st1,st2,st3,st4,jmin,jmax = line.split()
                st1,st2 = int(st1), int(st2)
                st3,st4 = int(st3), int(st4)
                jmin,jmax = int(jmin), int(jmax)
                j_curr = jmin
                
                j_tops = [min(j_tops[0], jmin), max(j_tops[1], jmax)]
                
                phs_ = 1
                if states_order:
                    if states_order.index(st1) > states_order.index(st2):
                        aux = st1
                        st1 = st2 
                        st2 = aux
                        
                        _,_,j1 = get_qqnn_antoine(st1)
                        _,_,j2 = get_qqnn_antoine(st2)
                        phs_ *= (-1)**((j1+j2)/2 + j_curr + 1)
                    if states_order.index(st3) > states_order.index(st4):
                        aux = st3 
                        st3 = st4
                        st4 = aux
                        
                        _,_,j1 = get_qqnn_antoine(st3)
                        _,_,j2 = get_qqnn_antoine(st4)
                        phs_ *= (-1)**((j1+j2)/2 + j_curr + 1)
                        
                bra_, ket_ = (st1, st2), (st3, st4)
                mmee[bra_, ket_] = dict([(j,[]) for j in range(jmin, jmax+1)])
                continue
            
            mmee[(bra_, ket_)][j_curr] = [float(x)*phs_ for x 
                                                      in line.strip().split()]
            j_curr += 1
    
    return mmee, j_tops


if __name__ == '__main__':
    
    #===========================================================================
    ## SET UP
    B_LEN = 1.81
    _b_str = int(10 * B_LEN)
    FLD_ = '../results/'
    # FILE_1 = f'LSSR_b{_b_str}_MZ3.2b'
    # FILE_2 = f'LSFR_b{_b_str}_MZ3.2b'
    
    FILE_1 = "onlyDD_D1S_scalar.2b"
    FILE_2 = "onlyDD_GDD_scalar.2b"
    
    ##
    #===========================================================================
    
    mmee_1, tops_j1 = matrixReaderAndSorter(FLD_ + FILE_1)
    mmee_2, tops_j2 = matrixReaderAndSorter(FLD_ + FILE_2)
    
    j_tops = min(tops_j1[0], tops_j2[0]), max(tops_j1[1], tops_j2[1])
    
    ## read, find and identify
    items = []
    items_1notin2 = set()
    for k,b in mmee_1.keys():
        if (k, b) not in mmee_2.keys():
            items_1notin2.add( (k,b) )
        else:
            items.append( (k,b))
    items_2notin1 = set()
    for k,b in mmee_2.keys():
        if (k, b) not in mmee_1.keys(): 
            items_2notin1.add( (k,b) )
        else:
            items.append( (k,b))
    items = set(items)
    
    print("Elements in FR not in ZR:", items_2notin1.__len__())
    for i in items_2notin1: print(i)
    print("Elements in ZR not in FR:", items_1notin2.__len__())
    for i in items_1notin2: print(i)
    print()
    
    data_pppp = []
    data_pnpn, data_pnnp = [], []
    data_nnnn = []
    
    phase_changing_me = [set(), set(), set(), set()]
    
    dim_j = j_tops[1] - j_tops[0] + 1
    for b, k in items:
        _base_list = [[-1,-1] for _ in range(dim_j)]
        data_pppp.append(deepcopy(_base_list))
        data_pnpn.append(deepcopy(_base_list))
        data_pnnp.append(deepcopy(_base_list))
        data_nnnn.append(deepcopy(_base_list))
        
        for i, mmee_ in enumerate((mmee_1, mmee_2)):
            for j, vals_ in mmee_[(b, k)].items():
                data_pppp[-1][j][i] = vals_[0]
                data_pnpn[-1][j][i] = vals_[1]
                data_pnnp[-1][j][i] = vals_[2]
                data_nnnn[-1][j][i] = vals_[5]
        
        for j in range(dim_j):
            if data_pppp[-1][j][0] * data_pppp[-1][j][1] < 0: 
                phase_changing_me[0].add((b, k))
            if data_pnpn[-1][j][0] * data_pnpn[-1][j][1] < 0: 
                phase_changing_me[1].add((b, k))
            if data_pnnp[-1][j][0] * data_pnnp[-1][j][1] < 0: 
                phase_changing_me[2].add((b, k))
            if data_nnnn[-1][j][0] * data_nnnn[-1][j][1] < 0: 
                phase_changing_me[3].add((b, k))
    
    print(phase_changing_me[0])
    ## TODO: PLOT log-log axis
    
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    # import pandas as pd 
    # import joypy
    from copy import deepcopy
    import numpy as np
    
    data = {"pppp": data_pppp, "pnpn": data_pnpn, 
            "pnnp": data_pnnp, "nnnn": data_nnnn,}
    for channel, data_ch in data.items():
        if not channel in ("pnpn", ): continue
        fig, ax_ = plt.subplots(1,2)
        
        max_ = 0
        list_for_histo = [[] for j in range(dim_j)]
        labels_histo   = [f"J={j}" for j in range(dim_j)]
        for j in range(dim_j-1, -1, -1):
            data_j = []
            for k in range(len(data_ch)):
                data_j.append(data_ch[k][j])
        
            x, y = (zip(*data_j))
            ## prepare to evaluate the histogram
            for k in range(len(data_ch)):
                if (abs(x[k]) > 1e-5) and (x[k]!=-1 and y[k]!=-1):
                    list_for_histo[j].append( np.log10(abs(y[k]/x[k])) )
            max_ = max(max_, max(x), max(y))
            ax_[0].scatter(x, y, s=10, label='J={}'.format(j)) 
        
        max_ *= (2)**-.5
        xxyy = [(max_*(2*k/10 - 1), max_*(2*k/10 - 1)) for k in range(11)]
        ax_[0].plot(*(zip(*xxyy)), '--k', lw=1)
        ax_[0].axhline(y=0.0, color='k', linestyle='--', linewidth=1)
        ax_[0].axvline(x=0.0, color='k', linestyle='--', linewidth=1)
        ax_[0].set_ylim(-10, 10)
        
        ax_[0].set_aspect('equal', 'box')
        ax_[0].set_title ("M.E. "+channel)
        # ax_[0].set_xlabel("Zero Range LS [MeV]")
        # ax_[0].set_ylabel("Finite Range LS [MeV]")
        ax_[0].set_xlabel("D1S-DD only [MeV]")
        ax_[0].set_ylabel("GDD only [MeV]")
        ax_[0].legend()
        
        #----------------------------------------------------------------------
        ## Prepare the histogram        
        ax_[1].hist(list_for_histo[1:5], label=labels_histo[1:5], bins = 20)
        ax_[1].legend()
        ax_[1].set_aspect('auto', 'box')
        ax_[1].set_title ("M.E. histogram distribution: "+channel)
        # ax_[1].set_xlabel("Zero Range LS [MeV]")
        # ax_[1].set_ylabel("Finite Range LS [MeV]")
        ax_[1].set_ylabel("Number")
        ax_[1].set_xlabel(r"$log_{10}|\frac{ GDD }{DD_{D1S}}| $")
        fig.tight_layout()
        
        
        
        fig2, ax2 = plt.subplots(1,1)
        _2plot = dict()
        val_min = min([min(x) for x in list_for_histo])
        val_max = max([max(x) for x in list_for_histo])
        dx = 0.1
        dim_boxes = int((val_max - val_min) / dx) + 1
        x = np.arange(val_min, val_max, dx)
        for j in range(0,7):
            aux = [0, ] * dim_boxes
            for val in list_for_histo[j]:
                i = int((val - val_min) / dx)
                aux[i] += 1
            _2plot[labels_histo[j]] = deepcopy(aux)
            ax2.plot(x, aux, marker=".*v^x+dD"[j], 
                     label=labels_histo[j]+
                            " (total:{})".format(len(list_for_histo[j])))
        
        ax2.set_ylabel("Number")
        ax2.set_xlabel(r"ratio $log_{10}|\frac{ GDD }{DD_{D1S}}| $")
        # df = pd.DataFrame(_2plot)
        # fig2, ax2 = joypy.joyplot(df, 
        #                           title = "Distribution of the m.e. (log) ratio "+channel)
        ax2.legend()
        plt.show()