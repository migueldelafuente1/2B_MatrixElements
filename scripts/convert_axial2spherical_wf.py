'''
Created on 9 abr 2025

@author: delafuente

Conversion for harmonic oscillator functions, from axial to spherical, 

convert also the nlm_l scheme of the transformation to jj coupling.

Nucl. Phys. A 90 (1967) 401-406.
Transformation scheme for harmonic-oscillator wave functions:
R.R. Chasman, S. Wahlborn
https://doi.org/10.1016/0375-9474(67)90242-4

'''
from helpers.Helpers import fact, gamma_half_int, valenceSpacesDict_l_ge10_byM,\
    readAntoine, safe_clebsch_gordan
import numpy as np

WF_TXT_AXIAL = """
     iqp     jz     mz     np     nz               Up                          Vp                         Un                         Un
     """

def _alpha_range(n, n_perp, l, m):
    
    a_min = max((n_perp - n) // 2, 0)
    a_max = min((l - abs(m))//2, l)
    
    #assert a_min >= 0 and a_max >= 0, f"Error, alpha range negative Min,Max:{a_min}, {a_max}"
    return a_min, a_max 
    
def conversion_coefficient_axial2Spherical(n_perp, m, n_z, n, l):
    """
    <n_perp, m, n_z|  n, l, m>
    """
    sum_ = 0.0
    if (2*n + l) != (2*n_perp + abs(m) + n_z):
        return 0
    # a_min, a_max = _alpha_range(n, n_perp, l, m)
    a_min, a_max = 0, l//2
    if a_min > a_max: return 0
    for k in range(a_min, a_max +1):
        phs  = (-1)**k
        aux = fact(2*(l-k)) + fact(n+k) - (
              fact(k) + fact(l-k) + fact(l - 2*k - abs(m)) + fact(n - n_perp + k)) 
        sum_ += phs * np.exp(aux)
    
    # aux = fact(l-m) + fact(n_z) + fact(n_perp + abs(m)) - (
    #     fact(n) + gamma_half_int(2*n+2*l+3) + fact(l+m) + fact(n_perp)
    #     + np.log(2)*(2*l + n_z))
    aux = fact(l-abs(m)) + fact(n_z) + fact(n_perp + abs(m)) - (
        fact(n) + gamma_half_int(2*n+2*l+3) + fact(l+abs(m)) + fact(n_perp)
        + np.log(2)*(2*l + n_z))
    
    aux = np.exp(aux / 2) * (np.pi**(1/2) * (l + 0.5))**0.5
    sum_ *= aux * ((-1)**(n + m + n_perp))
    
    if m < 0: sum_ *= (-1)**m   ## Legendre polynomial phase for m < 0
    return sum_ 

def convert_axial2spherical_wf(n_perp, m, n_z):
    N = 2*n_perp + abs(m) + n_z
    set_ = {}
    
    for n in range(0, N//2 +1):
        l = N - 2*n
        
        coeff = conversion_coefficient_axial2Spherical(n_perp, m, n_z, n, l)
        
        if abs(coeff) < 1e-7: continue
        set_[(n, l, m)] = coeff
        
    return set_

def convert_spherical2axial_wf(n, l, m):
    N = 2*n + l
    set_ = {}
    
    for n_perp in range(0, (N - abs(m))//2 +1):
        n_z = N - 2*n_perp - abs(m)
        
        coeff = conversion_coefficient_axial2Spherical(n_perp, m, n_z, n, l)
        
        if abs(coeff) < 1e-7: continue
        set_[(n_perp, m, n_z)] = coeff
    
    return set_

def test_norm_on_spherical_states(MZMAX=5):
    """
    verify norm of decomposed states as 1
    """
    
    for N in range(MZMAX +1):
        total, ko, ok = 0, 0, 0
        for l in range(0, N +1):
            n = (N - l)//2
            if 2*n + l != N: continue
            
            for m in range(-l, l +1):
                total += 1
                set_ = convert_spherical2axial_wf(n, l, m)
                norm = [x**2 for x in set_.values()]
                norm = sum(norm)
                
                if abs(norm - 1) > 1.e-6:
                    print(f"not OK for N= {N}: {n},{l}.{m} : {norm:15.12f}", set_)
                    ko += 1
                else:
                    ok += 1
        print(f" TEST DONE FOR N={N}, ok={ok: >4} fail={ko: >4} / {total: > 5}")

def convertAxialWFtoSpherical4Taurus(filename=None):
    """
    Format for axial wf:
    * iqp jz mz nperp nz U_prot V_prot U_neut V_neut
    
    The axial wf has a block structure where the quasi-particles conserve the jz
    """
    global FLD_MAIN
    filename = FLD_MAIN + filename
    if filename: 
        with open(filename, 'r') as f:
            data = f.readlines()[1:]
    else:
        data = WF_TXT_AXIAL.split('\n')
    
    ax_UVpn = [[], [], [], []]
    ax_UVpn_by_qn = {}
    ax_qqnn = {}
    ax_qqnn_ms = {}
    ax_qqnn_by_N = {}
    ax_qp_set = set()
    NDUV, MZmax = 0, 0
    Upn_axi = np.zeros( (480, 480) )
    Vpn_axi = np.zeros( (480, 480) )
    ax_base_sort = []
    for k, line in enumerate(data):
        if not line.startswith('*') : continue
        if len(line) == 0 : continue
        
        args = line.split()
        qp_idx = int(args[1])
        qn   = [int(x) for x in args[2:6]]
        UVs  = [float(x) for x in args[6:]]
        qn[0] = 2*qn[0] - 1
        tpl_qn = tuple(qn)
        if not tpl_qn in ax_base_sort: ax_base_sort.append(tpl_qn)
        ax_qp_set.add(qp_idx)
        
        MZmax = max(MZmax, qn[0])
        if not tpl_qn in ax_qqnn:
            ax_qqnn   [ tpl_qn ] = []
            ax_qqnn_ms[ tpl_qn ] = []
        ax_qqnn   [ tpl_qn ].append(qp_idx)
        ax_qqnn_ms[ tpl_qn ].append(qn[0] - 2*qn[1])
        assert ax_qqnn_ms[ tuple(qn) ][-1] in (1, -1), "Invalid MS!"
        
        N = 2*qn[2] + qn[3] + abs(qn[1])
        if not N in ax_qqnn_by_N:  ax_qqnn_by_N[N] = set()
        ax_qqnn_by_N[N].add( tpl_qn )
        
        for i in range(4): ax_UVpn[i].append(UVs[i])
        ax_UVpn_by_qn[ (qp_idx, *tpl_qn) ] = k
        
        Upn_axi[ax_base_sort.index(tpl_qn), qp_idx-1]         = UVs[0]
        Vpn_axi[ax_base_sort.index(tpl_qn), qp_idx-1]         = UVs[1]
        Upn_axi[ax_base_sort.index(tpl_qn)+240, qp_idx-1+240] = UVs[2]
        Vpn_axi[ax_base_sort.index(tpl_qn)+240, qp_idx-1+240] = UVs[3]
        
        phsU = (-1)**(tpl_qn[1])
        phsV = phsU
        Upn_axi[ax_base_sort.index(tpl_qn)+120, qp_idx-1+120] = UVs[0] * phsU
        Vpn_axi[ax_base_sort.index(tpl_qn)+120, qp_idx-1+120] = UVs[1] * phsV
        Upn_axi[ax_base_sort.index(tpl_qn)+360, qp_idx-1+360] = UVs[2] * phsU
        Vpn_axi[ax_base_sort.index(tpl_qn)+360, qp_idx-1+360] = UVs[3] * phsV
        NDUV += 1
    
    ## get the shell and quantum states for the wave function
    MZmax = MZmax // 2
    dim_ax_base = sum([len(x) for x in ax_qqnn.values()])
    print(" MZMax read =", MZmax, "dim UV (NDUV)=", NDUV, 
          " # ax_qn / 2 (only +mz stored)=",len(ax_qqnn), )
    
    sho_base = []
    for M in range(MZmax +1):
        sho_base = sho_base + list(valenceSpacesDict_l_ge10_byM[M])
    #for x in sho_base: print(x)
    sho_sp_base = []
    sho_sp_base_tr = []
    i = 0
    for x in sho_base:
        n, l, j = readAntoine(x, l_ge_10=True)
        for jz in range(j, -j-1, -2):
            sho_sp_base.append( (n, l, j, jz) )
            
            sho_sp_base_tr.append( i + (j+jz)//2 )
        i += j + 1
    
    n_dim = len(sho_sp_base)
    _N2 = n_dim 
    Upn_sph = np.zeros( (2*n_dim, 2*n_dim) )
    Vpn_sph = np.zeros( (2*n_dim, 2*n_dim) )
    __text_block = []
    def printt(*str_):
        line = list(str_)
        line = ' '.join(line)
        __text_block.append(line)
        print(line)
    
    for i in range(n_dim):
        ## Transform the base states for each quasi-particle
        n, l, j, jz = sho_sp_base[i]
        
        N = 2*n + l
        for ms  in (-1, 1):
            m = (jz - ms) // 2
            if abs(m) > l: continue
            
            printt(f"sph=[{i: >3}] ({n: >2}, {l: >2}, {j: >2}/2, {jz: >+3}/2) ms={ms: >+2}")
            
            # cgc = safe_clebsch_gordan(1/2, l, j/2, ms/2, m, jz/2)
            cgc = safe_clebsch_gordan(l, 1/2, j/2, m, ms/2, jz/2)
            if abs(cgc) < 1.0e-7: continue 
            
            for qn_ax in ax_qqnn_by_N[N]:
                ## Read all compatible axial states for the spherical state.
                jz_ax, mz, nperp, nz = qn_ax
                jz_ax2, mz2 = jz_ax, mz
                
                if jz_ax != abs(jz):
                    #print(" Why you reach this qn, jz does not match: skip!")
                    continue
                elif jz_ax != jz:
                    ## qqnn with oposite m, ms sign:
                    jz_ax2 = -jz_ax
                    mz2    = -mz
                # cgc *= (-1)**((2*l + 1 -j) // 2)
                ms_ax = jz_ax2 - 2*mz2
                if ms_ax != ms: continue
                
                coeff = conversion_coefficient_axial2Spherical(nperp, mz2, nz, n, l)
                aux = coeff * cgc
                
                printt(f"  >>> ax ({nperp: >2}, {nz: >2}, {mz2: >+3}, {jz_ax2: >+3}/2) -> CG, coeff ={aux: 4.3f}, {coeff: 4.3f}")
                for iqp in ax_qqnn[qn_ax]:
                    
                    k = ax_UVpn_by_qn[ (iqp, jz_ax, mz, nperp, nz) ]
                    
                    _str = '  '.join([f"{ax_UVpn[iii][k]: >+6.5e}" for iii in range(4)])
                    
                    if jz > 0:
                        phsU,phsV = 1, -1
                        # phsU = (-1)**((j - 1)//2)
                        # phsV = (-1)**((j - 1)//2)
                        
                        Upn_sph[i, iqp-1]                += aux * ax_UVpn[0][k] * phsU
                        Upn_sph[i+ n_dim, iqp-1 + n_dim] += aux * ax_UVpn[2][k] * phsU
                        _str2 = [f"{aux * ax_UVpn[ii][k] * phsU: >+6.5e}" for ii in (0, 2)]
                        
                        iqp += n_dim//2
                        Vpn_sph[i, iqp-1]                += aux * ax_UVpn[1][k] * phsV
                        Vpn_sph[i+ n_dim, iqp-1 + n_dim] += aux * ax_UVpn[3][k] * phsV
                        
                        _str2 += [f"{aux * ax_UVpn[ii][k] * phsV: >+6.5e}" for ii in (1, 3)]
                        printt(f"     > i={i: >4}, iqp={iqp: >4} * aux /UV pn: {_str} = {_str2}")
                    else:
                        phsU = (-1)**(mz)
                        phsV = (-1)**(mz)
                        # phsU = (-1)**((j - 1 - 2*l)//2)
                        # phsV = (-1)**((j - 1 - 2*l)//2)
                        
                        itr = i #sho_sp_base_tr[i] # i ya apunta al indice base TR
                        Vpn_sph[itr, iqp-1]                += aux * ax_UVpn[1][k] * phsV
                        Vpn_sph[itr+ n_dim, iqp-1 + n_dim] += aux * ax_UVpn[3][k] * phsV
                        _str2 = [f"{aux * ax_UVpn[ii][k] * phsV: >+6.5e}" for ii in (0, 2)]
                        
                        iqp += n_dim//2
                        Upn_sph[itr, iqp-1]                += aux * ax_UVpn[0][k] * phsU
                        Upn_sph[itr+ n_dim, iqp-1 + n_dim] += aux * ax_UVpn[2][k] * phsU
                        _str2 += [f"{aux * ax_UVpn[ii][k] * phsU: >+6.5e}" for ii in (1, 3)]
                        printt(f"     > i={i: >4}, iqp={iqp: >4} * aux /UV pn: {_str} = {_str2}")
                    
                    # for ii in range(4):
                    #     aux[ii] += coeff * cgc * ax_UVpn[ii][iqn]
                    # print(f"k={k}, {sho_sp_base[k]} += {qn_ax} // {aux}")
            _ = 0
    ## reshape the order of the axial quasiparticle
    REORDER = False
    if REORDER:
        auxU = np.zeros( (2*n_dim, 2*n_dim) )
        auxV = np.zeros( (2*n_dim, 2*n_dim) )
        axial_qp_by_sph_indx = {}
        k1, k2 = 0, 0
        for k in range(n_dim):
            m = sho_sp_base[k][3]
            if m > 0:
                axial_qp_by_sph_indx[k] = k1
                k1 += 1
            else:
                axial_qp_by_sph_indx[k] = k2 + n_dim // 2
                k2 += 1
        print("  Example of the first 30-qp states associated with the basis state ")
        for j in range(30):
            k = axial_qp_by_sph_indx[j]
            print(f" j={j:3} -> k={k:3} {sho_sp_base[j]}")
        
        for j in range(n_dim):
            
            k = axial_qp_by_sph_indx[j]
            for i in range(2*n_dim):
                auxU[i, j] = Upn_sph[i, k]
                auxU[i, j + n_dim] = Upn_sph[i, k + n_dim]
                
                auxV[i, j] = Vpn_sph[i, k]
                auxV[i, j + n_dim] = Vpn_sph[i, k + n_dim]
            
        Upn_sph = auxU
        Vpn_sph = auxV
        _ = 0
    
    with open(FLD_MAIN+'_kk_printLine.txt', 'w+') as f:
        txt = '\n'.join(__text_block)
        f.write(txt)
    
    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].imshow(Upn_axi, cmap='rainbow')
    # ax[0].set_title("U-axial")
    # ax[1].imshow(Vpn_axi, cmap='rainbow')
    # ax[1].set_title("V-axial")
    # plt.tight_layout()
    
    C, x = 1, np.matmul(np.transpose(Upn_sph), Upn_sph) + np.matmul(np.transpose(Vpn_sph), Vpn_sph)
    C, x = 0, np.matmul(np.transpose(Upn_sph), Vpn_sph) + np.matmul(np.transpose(Vpn_sph), Upn_sph)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                if abs(x[i, j]) > 1.0e-9: 
                    raise Exception(f" Fail at {i}, {j} {x[i, j]}") 
            else:
                if abs(x[i, j]) - C > 1.0e-9: 
                    raise Exception(f" Fail at Diagonal {i}, {j} ={x[i, j]}")
    print("[TEST] Orthogonality and Unitary UV relations: PASS")
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(Upn_sph , cmap='rainbow')
    ax[0].set_title("U-sph")
    ax[1].imshow(Vpn_sph, cmap='rainbow')
    ax[1].set_title("V-sph")
    plt.tight_layout()
    
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].imshow(Upn_sph, cmap='rainbow')
    # ax[0].set_title("U-sph")
    # ax[1].imshow(Vpn_sph, cmap='rainbow')
    # ax[1].set_title("V-sph")
    # plt.tight_layout()
    
    rho = np.matmul(Vpn_sph, np.transpose(Vpn_sph))
    kap = np.matmul(Vpn_sph, np.transpose(Upn_sph))
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(rho, cmap='rainbow')
    ax[0].set_title(f"rho-sph \nTr(Z,N)={np.trace(rho[:_N2, :_N2]):6.6f}, {np.trace(rho[_N2:, _N2:]):6.6f}")
    ax[1].imshow(kap, cmap='rainbow')
    ax[1].set_title(f"kappa-sph \nTr(0)={np.trace(kap[:_N2, :_N2]):6.6f}, {np.trace(kap[_N2:, _N2:]):6.6f}")
    plt.tight_layout()
    
    # rhoA = np.matmul(Vpn_axi, np.transpose(Vpn_axi))
    # kapA = np.matmul(Vpn_axi, np.transpose(Upn_axi))
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].imshow(rhoA, cmap='rainbow')
    # ax[0].set_title("rho-axial")
    # ax[1].imshow(kapA, cmap='rainbow')
    # ax[1].set_title("kappa-axial")
    # plt.tight_layout()
    plt.show()
    
    lines = [f"{len(sho_base): >12}" ,]
    for x in sho_base: lines.append(f"{x: >12}")
    lines.append('   949388824938899328')
    for i in range(2*n_dim):
        for j in range(2*n_dim):
            t = f"{Upn_sph[j][i]: >22.16E}"
            t = t.replace('E+00','') if t.endswith('0.0000000000000000E+00') else t
            t = "  "+t if t[0]=='-' else "   "+t
            lines.append(t)
    for i in range(2*n_dim):
        for j in range(2*n_dim):
            t = f"{Vpn_sph[j][i]: >22.16E}"
            t = t.replace('E+00','') if t.endswith('0.0000000000000000E+00') else t
            t = "  "+t if t[0]=='-' else "   "+t
            lines.append(t)
    with open('initial_wf.txt', 'w+') as f:
        lines = '\n'.join(lines)
        f.write(lines)
    
    print(" DONE: Function exported [initial_wf.txt]")

def readTaurusWaveFunction(filename):
    
    global FLD_MAIN
    with open(FLD_MAIN + filename, 'r') as f:
        data = f.readlines()
    
    UV = []
    uv_section = False
    for line in data:
        if not uv_section:
            if int(line) > 10000000: uv_section = True
            continue
        else:
            UV.append(float(line.strip()))  # .replace('E', 'e')
    
    n_dim = int(np.sqrt(len(UV)/2))
    _N2 = n_dim // 2
    U = np.zeros( (n_dim, n_dim) )
    V = np.zeros( (n_dim, n_dim) )
    k = 0
    for i in range(n_dim):
        for j in range(n_dim):
            U[j, i] = UV[k]
            k += 1
    for i in range(n_dim):
        for j in range(n_dim):
            V[j, i] = UV[k]
            k += 1
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(U, cmap='rainbow')
    ax[0].set_title("U-sph")
    ax[1].imshow(V, cmap='rainbow')
    ax[1].set_title("V-sph")
    plt.tight_layout()
    
    rho = np.matmul(V, np.transpose(V))
    kap = np.matmul(V, np.transpose(U))
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(rho, cmap='rainbow')
    ax[0].set_title(f"rho-sph \nTr(Z,N)={np.trace(rho[:_N2, :_N2]):6.6f}, {np.trace(rho[_N2:, _N2:]):6.6f}")
    ax[1].imshow(kap, cmap='rainbow')
    ax[1].set_title(f"kappa-sph \nTr(0)={np.trace(kap[:_N2, :_N2]):6.6f}, {np.trace(kap[_N2:, _N2:]):6.6f}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    FLD_MAIN = '../results/TEST_AXIALSPH_WF/'
    
    ## Tests conversion on HOWF (Axial to Spherical)
    wf_1 = (1, -2, 1) # n_perp, m, n_z
    set_sph = convert_axial2spherical_wf(*wf_1)
    print("np,m,nz=", wf_1, sum([x**2 for x in set_sph.values()]), set_sph)
    
    ## Test reversed conversion (Spherical to Axial)
    wf_1 = (10, 2, -2) # n, l, m
    set_axi = convert_spherical2axial_wf(*wf_1)
    print("n, l, m=",wf_1, sum([x**2 for x in set_axi.values()]), set_axi)
    ## Test the transformation is unitary and orthogonal <wf_1 | wf_2>
    
    test_norm_on_spherical_states(25)
    
    ## Call function to convert a axial wave function
    _ = 0
    # convertAxialWFtoSpherical4Taurus(filename='axial_wf16O_qp1.txt')
    convertAxialWFtoSpherical4Taurus(filename='axial_wf16C_qp3.txt')
    
    ## Read Taurus wavefunction and extract the Vs and Us
    # readTaurusWaveFunction('final_wf.txt')
    
    