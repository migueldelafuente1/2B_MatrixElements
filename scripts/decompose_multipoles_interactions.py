'''
Created on 20 feb 2024

@author: delafuente

This script decompose gaussian - gogny interactions and print every multipolar 
component,
'''
from helpers import MATPLOTLIB_INSTALLED
from helpers.Helpers import fact, double_factorial
if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os


def multipole_integral(L, r1, r2, mu_lengths):
    """
    return the function for each lambda
    """
    N_max = 10
    Z = [0, 0]
    for i, mu_ in enumerate(mu_lengths):

        Z[i] = 0 * r1
        for n in range(N_max):
            ## Note: require condition to have same l than lambda while n loop
            if ((n % 2) != (L % 2)) or (L > n): 
                continue
            
            ## Note: (n - Lambda) is even.
            aux = np.power(2*r1*r2 /(mu_**2) , n)
            cst = 2*(2*L + 1) / (np.exp(fact((n-L)//2) + 
                                        double_factorial(L+n+1) +
                                        (0.5*np.log(2)*((n-L)/2))) 
                                * (2*n + 1))
            Z[i] = Z[i] + (aux * cst)
            
        Z[i] = Z[i] * 2*np.pi/ np.exp((np.power(r1,2) + np.power(r2,2))/(mu_**2))
    return Z

def factor_ST(S,T, kargs):
    vals_ = [
        kargs['W'],
        kargs['B'] * ((-1)**(S + 1)),
        kargs['H'] * ((-1)**(T)),
        kargs['M'] * ((-1)**(S + T + 1)),
    ]
    return sum(vals_)

if __name__ == '__main__':
    
    INTERACTIONS = dict(
        B1 = {0: {"mu": 0.7, "W": 595.55, "B": 0, "H": 0, "M":-206.05}, 
              1: {"mu": 1.4, "W": -72.21, "B": 0, "H": 0, "M": -68.39}, },
        D1S= {0: {"mu": 0.7, "W":-1720.3, "B":  1300,   "H":-1813.53, "M": 1397.6}, 
              1: {"mu": 1.2, "W":103.639, "B":-163.483, "H": 162.812, "M":-223.934}, },
        D1 = {0: {"mu": 0.7, "W":-402.4, "B":-100,   "H":-496.2, "M": -23.56}, 
              1: {"mu": 1.2, "W": -21.3, "B":-11.77, "H": 37.27, "M":-68.81}, },
        )
    
    INTER_TITLE = 'B1'
    LAMBDA_MAX  = 10
    
    INTERACTION = INTERACTIONS[INTER_TITLE]
    r1 = np.linspace(0, 4, 100)
    r2 = np.linspace(0, 4, 100)
    
    X1, X2 = np.meshgrid(r1, r2)
    
    pdfs_ = []
    # Create a 3D plot  
    Z_TOTAL = [[0*X1, 0*X1], [0*X1, 0*X1]]
    Z_TOTAL = np.array(Z_TOTAL)
    for lambda_ in range(LAMBDA_MAX +1):
        mu_lengths = INTERACTION[0]['mu'], INTERACTION[1]['mu']
        Z = multipole_integral(lambda_, X1, X2, mu_lengths)
        
        fig, ax = plt.subplots(2, 2, figsize=(7, 5))  
        for S in range(2):
            for T in range(2):
                indx = (S, T)
                Z_ST = 0 * X1
                for i, kwargs in INTERACTION.items():
                    Z_ST = Z_ST + factor_ST(S, T, kwargs) * Z[i]
                Z_TOTAL[indx] += Z_ST
                # Plot the surface
                contour_ = ax[indx].contourf(X1, X2, Z_ST, cmap='viridis')
    
                # Set labels and title
                if indx[0] == 1: 
                    ax[indx].set_xlabel('r1')
                if indx[1] == 0:
                    ax[indx].set_ylabel('r2')
                # ax.set_zlabel('Z-axis')
                ax[indx].set_title(f'S={S} T={T}')
                cbar = fig.colorbar(contour_, ax=ax[indx])
                # cbar.set_label("MeV")
        
        plt.tight_layout()
        fig.suptitle(f"Multipole ($\lambda=${lambda_}) for {INTER_TITLE}", 
                     fontsize= 10, fontweight= 'bold')
        pdf_title = f"channels_{INTER_TITLE}_lam{lambda_}.pdf"
        fig.savefig(pdf_title)
        pdfs_.append(pdf_title)
    
    # ------------------------------------------------------------------------ #
    fig, ax = plt.subplots(2, 2, figsize=(7, 5))  
    for S in range(2):
        for T in range(2):
            indx = (S, T)
            # Plot the surface
            contour_ = ax[indx].contourf(X1, X2, Z_TOTAL[indx], cmap='viridis')

            # Set labels and title
            if indx[0] == 1:
                ax[indx].set_xlabel('r1')
            if indx[1] == 0:
                ax[indx].set_ylabel('r2')
            # ax.set_zlabel('Z-axis')
            ax[indx].set_title(f'S={S} T={T}')
            cbar = fig.colorbar(contour_, ax=ax[indx])
            # cbar.set_label("MeV")
    
    plt.tight_layout()
    fig.suptitle(f"Multipole Combined for {INTER_TITLE}", 
                 fontsize= 10, fontweight= 'bold')
    pdf_title = f"channels_{INTER_TITLE}.pdf"
    fig.savefig(pdf_title)
    pdfs_.append(pdf_title)
    
    plt.show()
    
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    for pdf in pdfs_:
        merger.append(pdf)
    merger.write(f"contributions_v12_{INTER_TITLE}.pdf")
    merger.close()
    for pdf in pdfs_:
        os.remove(pdf)
    
    
    
    