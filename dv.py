"""
compute error in velocity approximation
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import brentq

exp = np.exp

def v_full(v,pdict):
    """
    find velocity
    """
    A=pdict['A'];B=pdict['B'];u=pdict['u'];d=pdict['d'];be=pdict['be']
    ze=pdict['ze']
    
    return -v + A*(u-d)/(d*(1-exp(be*(A-B)/v))/be + u/be + ze)

def v_num(pdict):
    """
    pdict: dict, of parameters

    return numerical velocity given pdict
    """
    #A=pdict['A'];u=pdict['u'];d=pdict['d'];be=pdict['be']
    #ze=pdict['ze']

    # run brentq to get velocity
    return brentq(v_full,.1,1000,args=pdict)

def v_ana(pdict,o=1):
    """
    pdict: dict, of parameters

    return analytical velocity (approx) given pdict
    """
    A=pdict['A'];B=pdict['B'];u=pdict['u'];d=pdict['d'];be=pdict['be']
    ze=pdict['ze'];k=1

    if o == 1:

        if d > u:
            return (-(B*be*d*k) + 2*A*be*k*j - B*be*k*j + A*be**2*ze - B*be**2*ze - np.sqrt(-4*(-(A**2*be**2*d*k) + A*B*be**2*d*k + A**2*be**2*k*j - A*B*be**2*k*j)*(k*j + be*ze) + (B*be*d*k - 2*A*be*k*j + B*be*k*j - A*be**2*ze + B*be**2*ze)**2))/(2.*(k*j + be*ze))
        else:
            return (-(B*be*d*k) + 2*A*be*k*u - B*be*k*u + A*be**2*ze - B*be**2*ze + np.sqrt(-4*(-(A**2*be**2*d*k) + A*B*be**2*d*k + A**2*be**2*k*u - A*B*be**2*k*u)*(k*u + be*ze) + (B*be*d*k - 2*A*be*k*u + B*be*k*u - A*be**2*ze + B*be**2*ze)**2))/(2.*(k*u + be*ze))

    elif o == 2:

        return -0.3333333333333333*(B*be*d*k - 2*A*be*k*u + B*be*k*u - A*be**2*ze + B*be**2*ze)/(k*u + be*ze) - (2**0.3333333333333333*(-(B*be*d*k - 2*A*be*k*u + B*be*k*u - A*be**2*ze + B*be**2*ze)**2 + 3*(k*u + be*ze)*(-(A*B*be**2*d*k) + B**2*be**2*d*k + 2*A**2*be**2*k*u - 3*A*B*be**2*k*u + B**2*be**2*k*u + A**2*be**3*ze - 2*A*B*be**3*ze + B**2*be**3*ze)))/(3.*(k*u + be*ze)*(-2*B**3*be**3*d**3*k**3 + 3*A*B**2*be**3*d**2*k**3*u + 3*B**3*be**3*d**2*k**3*u - 27*A**3*be**3*d*k**3*u**2 + 66*A**2*B*be**3*d*k**3*u**2 - 57*A*B**2*be**3*d*k**3*u**2 + 12*B**3*be**3*d*k**3*u**2 + 7*A**3*be**3*k**3*u**3 - 6*A**2*B*be**3*k**3*u**3 - 6*A*B**2*be**3*k**3*u**3 + 7*B**3*be**3*k**3*u**3 - 3*A*B**2*be**4*d**2*k**2*ze + 3*B**3*be**4*d**2*k**2*ze - 54*A**3*be**4*d*k**2*u*ze + 138*A**2*B*be**4*d*k**2*u*ze - 108*A*B**2*be**4*d*k**2*u*ze + 24*B**3*be**4*d*k**2*u*ze + 6*A**3*be**4*k**2*u**2*ze + 6*A**2*B*be**4*k**2*u**2*ze - 33*A*B**2*be**4*k**2*u**2*ze + 21*B**3*be**4*k**2*u**2*ze - 27*A**3*be**5*d*k*ze**2 + 66*A**2*B*be**5*d*k*ze**2 - 51*A*B**2*be**5*d*k*ze**2 + 12*B**3*be**5*d*k*ze**2 - 6*A**3*be**5*k*u*ze**2 + 33*A**2*B*be**5*k*u*ze**2 - 48*A*B**2*be**5*k*u*ze**2 + 21*B**3*be**5*k*u*ze**2 - 7*A**3*be**6*ze**3 + 21*A**2*B*be**6*ze**3 - 21*A*B**2*be**6*ze**3 + 7*B**3*be**6*ze**3 + np.sqrt((-2*B**3*be**3*d**3*k**3 + 3*A*B**2*be**3*d**2*k**3*u + 3*B**3*be**3*d**2*k**3*u - 27*A**3*be**3*d*k**3*u**2 + 66*A**2*B*be**3*d*k**3*u**2 - 57*A*B**2*be**3*d*k**3*u**2 + 12*B**3*be**3*d*k**3*u**2 + 7*A**3*be**3*k**3*u**3 - 6*A**2*B*be**3*k**3*u**3 - 6*A*B**2*be**3*k**3*u**3 + 7*B**3*be**3*k**3*u**3 - 3*A*B**2*be**4*d**2*k**2*ze + 3*B**3*be**4*d**2*k**2*ze - 54*A**3*be**4*d*k**2*u*ze + 138*A**2*B*be**4*d*k**2*u*ze - 108*A*B**2*be**4*d*k**2*u*ze + 24*B**3*be**4*d*k**2*u*ze + 6*A**3*be**4*k**2*u**2*ze + 6*A**2*B*be**4*k**2*u**2*ze - 33*A*B**2*be**4*k**2*u**2*ze + 21*B**3*be**4*k**2*u**2*ze - 27*A**3*be**5*d*k*ze**2 + 66*A**2*B*be**5*d*k*ze**2 - 51*A*B**2*be**5*d*k*ze**2 + 12*B**3*be**5*d*k*ze**2 - 6*A**3*be**5*k*u*ze**2 + 33*A**2*B*be**5*k*u*ze**2 - 48*A*B**2*be**5*k*u*ze**2 + 21*B**3*be**5*k*u*ze**2 - 7*A**3*be**6*ze**3 + 21*A**2*B*be**6*ze**3 - 21*A*B**2*be**6*ze**3 + 7*B**3*be**6*ze**3)**2 + 4*(-(B*be*d*k - 2*A*be*k*u + B*be*k*u - A*be**2*ze + B*be**2*ze)**2 + 3*(k*u + be*ze)*(-(A*B*be**2*d*k) + B**2*be**2*d*k + 2*A**2*be**2*k*u - 3*A*B*be**2*k*u + B**2*be**2*k*u + A**2*be**3*ze - 2*A*B*be**3*ze + B**2*be**3*ze))**3))**0.3333333333333333) + (-2*B**3*be**3*d**3*k**3 + 3*A*B**2*be**3*d**2*k**3*u + 3*B**3*be**3*d**2*k**3*u - 27*A**3*be**3*d*k**3*u**2 + 66*A**2*B*be**3*d*k**3*u**2 - 57*A*B**2*be**3*d*k**3*u**2 + 12*B**3*be**3*d*k**3*u**2 + 7*A**3*be**3*k**3*u**3 - 6*A**2*B*be**3*k**3*u**3 - 6*A*B**2*be**3*k**3*u**3 + 7*B**3*be**3*k**3*u**3 - 3*A*B**2*be**4*d**2*k**2*ze + 3*B**3*be**4*d**2*k**2*ze - 54*A**3*be**4*d*k**2*u*ze + 138*A**2*B*be**4*d*k**2*u*ze - 108*A*B**2*be**4*d*k**2*u*ze + 24*B**3*be**4*d*k**2*u*ze + 6*A**3*be**4*k**2*u**2*ze + 6*A**2*B*be**4*k**2*u**2*ze - 33*A*B**2*be**4*k**2*u**2*ze + 21*B**3*be**4*k**2*u**2*ze - 27*A**3*be**5*d*k*ze**2 + 66*A**2*B*be**5*d*k*ze**2 - 51*A*B**2*be**5*d*k*ze**2 + 12*B**3*be**5*d*k*ze**2 - 6*A**3*be**5*k*u*ze**2 + 33*A**2*B*be**5*k*u*ze**2 - 48*A*B**2*be**5*k*u*ze**2 + 21*B**3*be**5*k*u*ze**2 - 7*A**3*be**6*ze**3 + 21*A**2*B*be**6*ze**3 - 21*A*B**2*be**6*ze**3 + 7*B**3*be**6*ze**3 + np.sqrt((-2*B**3*be**3*d**3*k**3 + 3*A*B**2*be**3*d**2*k**3*u + 3*B**3*be**3*d**2*k**3*u - 27*A**3*be**3*d*k**3*u**2 + 66*A**2*B*be**3*d*k**3*u**2 - 57*A*B**2*be**3*d*k**3*u**2 + 12*B**3*be**3*d*k**3*u**2 + 7*A**3*be**3*k**3*u**3 - 6*A**2*B*be**3*k**3*u**3 - 6*A*B**2*be**3*k**3*u**3 + 7*B**3*be**3*k**3*u**3 - 3*A*B**2*be**4*d**2*k**2*ze + 3*B**3*be**4*d**2*k**2*ze - 54*A**3*be**4*d*k**2*u*ze + 138*A**2*B*be**4*d*k**2*u*ze - 108*A*B**2*be**4*d*k**2*u*ze + 24*B**3*be**4*d*k**2*u*ze + 6*A**3*be**4*k**2*u**2*ze + 6*A**2*B*be**4*k**2*u**2*ze - 33*A*B**2*be**4*k**2*u**2*ze + 21*B**3*be**4*k**2*u**2*ze - 27*A**3*be**5*d*k*ze**2 + 66*A**2*B*be**5*d*k*ze**2 - 51*A*B**2*be**5*d*k*ze**2 + 12*B**3*be**5*d*k*ze**2 - 6*A**3*be**5*k*u*ze**2 + 33*A**2*B*be**5*k*u*ze**2 - 48*A*B**2*be**5*k*u*ze**2 + 21*B**3*be**5*k*u*ze**2 - 7*A**3*be**6*ze**3 + 21*A**2*B*be**6*ze**3 - 21*A*B**2*be**6*ze**3 + 7*B**3*be**6*ze**3)**2 + 4*(-(B*be*d*k - 2*A*be*k*u + B*be*k*u - A*be**2*ze + B*be**2*ze)**2 + 3*(k*u + be*ze)*(-(A*B*be**2*d*k) + B**2*be**2*d*k + 2*A**2*be**2*k*u - 3*A*B*be**2*k*u + B**2*be**2*k*u + A**2*be**3*ze - 2*A*B*be**3*ze + B**2*be**3*ze))**3))**0.3333333333333333/(3.*2**0.3333333333333333*(k*u + be*ze))
     
    #return A*(u-d)/(d/be+u/be+ze)


def main():

    pdict = {'A':5,'B':5.1,'u':15,'d':4,'al':14,'be':126,'ze':.2}

    ulist = np.arange(0,100)
    dlist = np.arange(0,100)

    matsol_ana = np.zeros((len(ulist),len(dlist)))
    matsol_num = np.zeros((len(ulist),len(dlist)))
    matsol_err = np.zeros((len(ulist),len(dlist)))

    for i in ulist:
        for j in dlist:
            if i > j:
                pdict['u'] = ulist[i]; pdict['d'] = dlist[j]

                v1 = v_num(pdict)
                v2 = v_ana(pdict)

                matsol_num[i,j] = v1
                matsol_ana[i,j] = v2
                
                matsol_err[i,j] = np.abs((v1-v2)/v1)

    # make matrix symmetric
    for i in ulist:
        for j in dlist:
            if i < j:
                matsol_err[i,j] = matsol_err[j,i]

    fig, axs = plt.subplots(1,2,figsize=(8,4))

    im = axs[0].imshow(matsol_err,origin='lower')

    #im2 = axs[1].imshow(matsol_num,origin='lower')
    #im3 = axs[2].imshow(matsol_ana,origin='lower')

    ratio = pdict['al']/(pdict['al']+pdict['be'])
    rect = patches.Rectangle((0, 0), ratio*100, ratio*100,
                             linewidth=2, edgecolor='tab:red',
                             facecolor='none',zorder=4)

    max_idx = int(ratio*100)
    im2 = axs[1].imshow(matsol_err[:max_idx,:max_idx],origin='lower')

    # Add the patch to the Axes
    axs[0].add_patch(rect)


    axs[0].set_xlabel(r'$D$')
    axs[1].set_xlabel(r'$D$')
    
    axs[0].set_ylabel(r'$U$')
    axs[1].set_ylabel(r'$U$')

    # colorbar nonsense
    divider0 = make_axes_locatable(axs[0])
    divider1 = make_axes_locatable(axs[1])
    
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    
    fig.colorbar(im,cax=cax0)
    fig.colorbar(im2,cax=cax1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
