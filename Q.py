"""
Generator matrix code

find local maxima above threshold of 1e-3

select 2 local maxima that exist above and below diagonal i=j.
select 1 local max that exists along diaonal i=j.

if the 2 local max > 1 local max then compute mfpt.
if the 2 local max < 1 local max then take note but still compute
mfpt of off-diagonal mfpt

need to make code robust to having only 1 max.
"""


import copy
import argparse

import numpy as np
#import mpmath as mp
#import matplotlib.pyplot as plt

from scipy.ndimage import maximum_filter
from numpy.linalg import lstsq
from scipy.signal import argrelmax

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_array, lil_array
from scipy.ndimage import maximum_filter

import sys, os

class Parameters(object):

    def __init__(self,A=5,B=5.5,be=150,ze=1,k=0.322*4,
                 nx=100,ny=100,al=14):

        """
        nx, ny are total number of binding sites
        so attached motors range from 0 to nx or ny.
        """
        self.A = A
        self.B = B
        self.be = be
        self.ze = ze
        self.k = k
        self.al = al
        self.nx = nx
        self.ny = ny

        self.pars_list = [self.A,self.B,self.be,
                          self.ze,self.k,self.al,
                          self.nx,self.ny]
        
        self.pars_dict = {'A':self.A,'B':self.B,'be':self.be,
                          'ze':self.ze,'k':self.k,'al':self.al,
                          'nx':self.nx,'ny':self.ny}

        self.pars_name = ['A','B','be','ze','k','al','nx','ny']


def Q_fname(name,p,legacy=True,base='dat/',ext='.txt'):

    if legacy:
        pref = 'master'
        fname = 'dat/switch_time_'+pref+'_A={}_B={}_al={}_be={}_ze={}'
        fname2 = 'dat/'+pref+'_peaks_A={}_B={}_al={}_be={}_ze={}.txt'
        fname3 = 'dat/'+pref+'_peak_idxs_A={}_B={}_al={}_be={}_ze={}.txt'
        return fname, fname2, fname3

    else:
        par_str = 'A={}_B={}_be={}_ze={}_k={}_al={}_nx={}_ny={}'
        fname = base+name+par_str.format(*p.pars_list)

        return fname

def v(i,j,p):

    A=p.A;B=p.B;be=p.be
    ze=p.ze;k=1

    if i > j:
        vel = (-2*A*be*i*k + B*be*i*k + B*be*k*j - A*be**2*ze + B*be**2*ze - np.sqrt(-4*(A**2*be**2*i*k - A*B*be**2*i*k - A**2*be**2*k*j + A*B*be**2*k*j)*(i*k + be*ze) + (2*A*be*i*k - B*be*i*k - B*be*k*j + A*be**2*ze - B*be**2*ze)**2))/(2.*(i*k + be*ze))

    elif j > i:
        vel = (-(B*be*i*k) + 2*A*be*k*j - B*be*k*j + A*be**2*ze - B*be**2*ze + np.sqrt(-4*(-(A**2*be**2*i*k) + A*B*be**2*i*k + A**2*be**2*k*j - A*B*be**2*k*j)*(k*j + be*ze) + (B*be*i*k - 2*A*be*k*j + B*be*k*j - A*be**2*ze + B*be**2*ze)**2))/(2.*(k*j + be*ze))
    else:
        vel = 1e-10
    
    return vel

def varr(i,j,p):
    """
    same as v but allows arrays.
    """
    A=p.A;B=p.B;be=p.be
    ze=p.ze;k=1

    i = np.asarray(i);j = np.asarray(j)

    v1 = (-2*A*be*i*k + B*be*i*k + B*be*k*j - A*be**2*ze + B*be**2*ze - np.sqrt(-4*(A**2*be**2*i*k - A*B*be**2*i*k - A**2*be**2*k*j + A*B*be**2*k*j)*(i*k + be*ze) + (2*A*be*i*k - B*be*i*k - B*be*k*j + A*be**2*ze - B*be**2*ze)**2))/(2.*(i*k + be*ze))
    v1 *= i > j

    v2 = (-(B*be*i*k) + 2*A*be*k*j - B*be*k*j + A*be**2*ze - B*be**2*ze + np.sqrt(-4*(-(A**2*be**2*i*k) + A*B*be**2*i*k + A**2*be**2*k*j - A*B*be**2*k*j)*(k*j + be*ze) + (B*be*i*k - 2*A*be*k*j + B*be*k*j - A*be**2*ze + B*be**2*ze)**2))/(2.*(k*j + be*ze))

    v2 *= j > i

    v2[j==i] *= 1e-10

    return v1 + v2

def dx(i,j,p):
    """
    detachment rate for x (down)
    """

    if i <= 0:
        return 0
    
    if i > j:
        return p.be*i

    else:
        vv = v(i,j,p)
        return p.be*i/(1-np.exp(p.be*(p.A-p.B)/np.abs(vv)))
        #return p.be*i

def dy(i,j,p):
    """
    detachment rate for y (up)
    """
    if j <= 0:
        return 0
    if i > j:
        vv = v(i,j,p)
        return p.be*j/(1-np.exp(p.be*(p.A-p.B)/np.abs(vv)))
        #return p.be*j

    else:
        return p.be*j

def a(i,p):
    """
    attachment rate
    """
    return p.al*(p.nx-i)

def dhat(i,j,p):

    return dx(i,j,p) + dy(i,j,p) + a(i,p) + a(j,p)

def vec_to_mat(vec,p):
    nx = p.nx+1
    ny = p.ny+1
    
    mat = np.zeros((nx,ny))
    #print(np.shape(mat))

    m = 0
    for i in range(nx*ny):
        n = i%nx

        if i%ny == 0 and i > 0:
            m += 1

        mat[n,m] = vec[i]

    return mat

def load_local_idxs(pars,recompute=True):
    """
    get indices of peaks in steady-state distributions.
    """
    
    fname = Q_fname(name='local_idxs',p=pars,legacy=False)

    if recompute or not(os.path.isfile(fname)):
        Qt = get_transition_mat(pars)
        qss = get_ss(pars,Qt)
        mat = vec_to_mat(qss,pars)
        #x_idxs,y_idxs = get_local_idxs(mat,pars) # 1 2 or 3 pairs


        out = get_local_idxs_v2(mat,pars) # 1 2 or 3 pairs
        x_idxs = out[:,0]
        y_idxs = out[:,1]

        

        dat = np.zeros([2,len(x_idxs)])
        dat[0] = x_idxs
        dat[1] = y_idxs
        np.savetxt(fname,dat)
        
        return x_idxs, y_idxs
    
    else:
        x_idxs, y_idxs = np.loadtxt(fname,dtype=np.int32)
        return x_idxs, y_idxs

            
    

def get_local_idxs(mat,p):
    """
    returns 1, 2, or 3 pairs.
    p: object
    """

    # get local max idxs
    # find local max idx in each col and row
    n = len(mat[0,:]) # all available motors
    n_eq = int(len(mat[0,:])*(p.al/(p.al+p.be)))+5 # mean motor attachment upper bound
    col_max = {}
    row_max = {}

    for i in range(n_eq):
        col_max[i] = (np.argmax(mat[:,i]),np.amax(mat[:,i]))
        row_max[i] = (np.argmax(mat[i,:]),np.amax(mat[i,:]))

    x_idxs = []
    y_idxs = []
    
    # assume symmetry. includes i == j
    for i in range(n_eq):
        for j in range(n_eq):
            is_row_max = False
            is_col_max = False
            
            if np.allclose(row_max[i][1],col_max[j][1]) and\
               col_max[j][1] > 1e-3 and\
               row_max[i][0] == j:

                # make sure given row/col is local max in
                # respective axis.

                if i > 0 and i < n_eq-1:
                    if row_max[i][1] > row_max[i-1][1] and\
                       row_max[i][1] > row_max[i+1][1]:
                        is_row_max = True
                if i == 0:
                    if row_max[i][1] > row_max[i+1][1]:
                        is_row_max = True
                if i == n_eq-1:
                    if row_max[i][1] > row_max[i-1][1]:
                        is_row_max = True

                if j > 0 and j < n_eq-1:
                    if col_max[j][1] > col_max[j-1][1] and\
                       col_max[j][1] > col_max[j+1][1]:
                        is_col_max = True
                if j == 0:
                    if col_max[j][1] > col_max[j+1][1]:
                        is_col_max = True
                        
                if j == n_eq-1:
                    if col_max[j][1] > col_max[j-1][1]:

                        is_col_max = True

                # in case the max is across 2 cells
                if i > 1 and i < n_eq-2:
                    if row_max[i][1] > row_max[i-2][1] and\
                       row_max[i][1] > row_max[i+2][1]:
                        is_row_max = True
                        
                if j > 1 and j < n_eq-2:
                    if col_max[j][1] > col_max[j-2][1] and\
                       col_max[j][1] > col_max[j+2][1]:
                        is_col_max = True
                
                if is_row_max and is_col_max:
                    x_idxs.append(i);y_idxs.append(j)


    # if there are more than 3 local max, restrict to 3.
    x_idxs = np.array(x_idxs);y_idxs = np.array(y_idxs)

    if len(x_idxs) > 3:
        # check if there are two peaks with i==j.
        # if so, remove one.

        # get indices of peaks with equal numbers of indices. 
        equal_idxs = np.where((x_idxs-y_idxs)==0)[0]

        bad_idxs = x_idxs[equal_idxs]
        
        # just make sure mask works if there are no peaks at i==j
        mask = np.ones(len(x_idxs),dtype=bool)
        mask[equal_idxs] = False
            
        print('mask,eq idxs,bad',mask,equal_idxs,bad_idxs)

        values = []
        for i in range(len(x_idxs)):
            if x_idxs[i] not in bad_idxs:
                values.append(col_max[x_idxs[i]][1])

        max_idxs_sorted = np.argsort(values)[-2:]

        # preserve order of max
        max_idxs = np.sort(max_idxs_sorted)

        xs = x_idxs[mask][max_idxs]; ys = y_idxs[mask][max_idxs]
        x_idxs = np.append(xs,bad_idxs)[:3]
        y_idxs = np.append(ys,bad_idxs)[:3]
        print(x_idxs,y_idxs)

        """
        # just make sure mask works if there are no peaks at i==j
        idxs = np.arange(len(x_idxs))
        if len(equal_idxs) == 0:
            mask = np.ones(len(x_idxs),dtype=bool)
        else:
            mask = idxs != equal_idxs
            
        print('mask,eq idxs,bad',mask,equal_idxs,bad_idxs)

        values = []
        for i in range(len(x_idxs)):
            if x_idxs[i] not in bad_idxs:
                values.append(col_max[x_idxs[i]][1])

        max_idxs_sorted = np.argsort(values)[-2:]

        # preserve order of max
        max_idxs = np.sort(max_idxs_sorted)

        xs = x_idxs[mask][max_idxs]; ys = y_idxs[mask][max_idxs]
        x_idxs = np.append(xs,bad_idxs)
        y_idxs = np.append(ys,bad_idxs)
        """

    return x_idxs,y_idxs


def get_local_idxs_v2(mat,p,size=3,threshold=1e-5):
    """
    returns 1, 2, or 3 pairs.
    p: object (is this even used?)
    """

    local_max = maximum_filter(mat,size=size)

    peaks = (mat == local_max) & (mat > threshold)

    peak_coordinates = np.argwhere(peaks)

    return peak_coordinates



def get_qj(Qt,I,J):

    #print('Computing hitting time')
    r,c = np.shape(Qt)
    #QJ = np.zeros((r-1,c-1))
    QJ = lil_array((r-1,c-1))

    Q = Qt.T
    
    
    
    QJ[:J-1,:J-1] = Q[:J-1,:J-1]
    QJ[:J-1,J-1:] = Q[:J-1,J:]
    
    QJ[J-1:,:J-1] = Q[J:,:J-1]

    QJ = QJ.tolil() # halves item assignment time
    QJ[J-1:,J-1:] = Q[J:,J:]

    return QJ
    

def get_full_mfpt(x_idxs,y_idxs,Qt,p):
    #if len(x_idxs) != 2 or len(y_idxs) != 2:
    #    return np.empty(Qt.shape[0])+np.nan
    
    assert(len(x_idxs) == len(y_idxs))

    n = p.nx+1

    I = x_idxs[0]+n*y_idxs[0]
    J = x_idxs[-1]+n*y_idxs[-1]

    QJ = get_qj(Qt,I,J)

    b =  np.ones((QJ.shape[0]))
    tauJ = spsolve(-QJ,b)

    return tauJ


def sep(x_idxs,y_idxs):
    """separate into 1 center and 2 symmetric"""
    assert(len(x_idxs) == 3)

    x_idxs2 = []; y_idxs2 = []
    x_idxs1 = []; y_idxs1 = []
    for i in range(len(x_idxs)):
        if x_idxs[i] != y_idxs[i]:
            
            # some peaks might have slightly asymmetric coordinates.
            if x_idxs[i]+1 == y_idxs[i] or x_idxs[i]-1 == y_idxs[i]:
                x_idxs1.append(x_idxs[i])
                y_idxs1.append(y_idxs[i])
            else:
                x_idxs2.append(x_idxs[i])
                y_idxs2.append(y_idxs[i])
                
        else:
            x_idxs1.append(x_idxs[i])
            y_idxs1.append(y_idxs[i])
    
    return x_idxs1,y_idxs1,x_idxs2,y_idxs2

def get_peaks(mat,x_idxs,y_idxs,pref='master'):

    if len(x_idxs) == 3: # separate
        x_idxs1,y_idxs1,x_idxs2,y_idxs2 = sep(x_idxs,y_idxs)
    
    
    peaks = np.empty(3);peaks[:] = np.nan
    peak_idxs = np.empty((3,2));peak_idxs[:,:] = np.nan

    if len(x_idxs) == 1:
        peaks[1] = mat[x_idxs[0],y_idxs[0]]
        peak_idxs[1,0] = x_idxs[0];peak_idxs[1,1] = y_idxs[0]
    if len(x_idxs) == 2:
        peaks[0] = mat[x_idxs[0],y_idxs[0]]
        peaks[2] = mat[x_idxs[1],y_idxs[1]]

        peak_idxs[0,0] = x_idxs[0];peak_idxs[0,1] = y_idxs[0]
        peak_idxs[2,0] = x_idxs[1];peak_idxs[2,1] = y_idxs[1]
        
    if len(x_idxs) == 3:
        peaks[0] = mat[x_idxs2[0],y_idxs2[0]]
        peaks[1] = mat[x_idxs1[0],y_idxs1[0]]
        peaks[2] = mat[x_idxs2[1],y_idxs2[1]]

        peak_idxs[0,0] = x_idxs2[0];peak_idxs[0,1] = y_idxs2[0]
        peak_idxs[1,0] = x_idxs1[0];peak_idxs[1,1] = y_idxs1[0]
        peak_idxs[2,0] = x_idxs2[1];peak_idxs[2,1] = y_idxs2[1]

    return peaks,peak_idxs

    
def get_transition_mat(p):
    
    #Qt = csr_array(((p.nx+1)*(p.ny+1),(p.nx+1)*(p.ny+1)),dtype=float)
    Qt = lil_array(((p.nx+1)*(p.ny+1),(p.nx+1)*(p.ny+1)),dtype=float)
    
    #Qt = np.zeros(((p.nx+1)*(p.ny+1),(p.nx+1)*(p.ny+1)))
    #print(np.shape(Qt))

    # set diagonal blocks
    for j in range(p.ny+1):
        #M = np.zeros((p.nx+1,p.nx+1))
        M = lil_array((p.nx+1,p.nx+1))
        for i in range(p.nx+1):
            M[i,i] = -dhat(i,j,p)

            if i < p.nx:
                M[i+1,i] = a(i,p)

            if i < p.nx:
                M[i,i+1] = dx(i+1,j,p)

        Qt[j*(p.ny+1):j*(p.ny+1)+p.nx+1,j*(p.ny+1):j*(p.ny+1)+p.nx+1] = M
        #Qt[j*(p.ny+1):j*(p.ny+1)+(i+1)+1,j*(p.ny+1):j*(p.ny+1)+(i+1)+1] = -dhat(i,j,p)

    #sys.exit()
    
    # set off-diagonal blocks    
    for j in range(p.ny):

        M1 = lil_array((p.nx+1,p.nx+1))#np.zeros((p.nx+1,p.nx+1))
        M2 = lil_array((p.nx+1,p.nx+1))#np.zeros((p.nx+1,p.nx+1))
        
        for i in range(p.nx+1):
            
            M1[i,i] = a(j,p)
            M2[i,i] = dy(i,j+1,p)
            #print(i,j+1,dy(i,j+1,p))
            
        Qt[j*(p.ny+1)+p.nx+1:j*(p.ny+1)+2*(p.nx+1),j*(p.ny+1):j*(p.ny+1)+p.nx+1] = M1
        Qt[j*(p.ny+1):j*(p.ny+1)+p.nx+1,j*(p.ny+1)+p.nx+1:j*(p.ny+1)+2*(p.nx+1)] = M2
        
    return Qt


def get_ss(p,Qt):
    """
    get steady-state probability distribution
    """
        
    #### steady-state
    # append normalization condition p_0 + p_1 + ... + p_M = 1
    ones = np.array([np.ones((p.nx+1)*(p.ny+1))])

    # solve by augmenting coefficient matrix with row of 1s
    #Qt_aug = np.append(Qt,ones,axis=0)
    #b_aug = np.zeros(len(Qt[:,0]))
    #b_aug[-1] = 1
    #sol = lstsq(Qt_aug,b_aug,rcond=None,lapack_driver='gelsy')[0]
    #sol = lstsq(Qt_aug.T.dot(Qt_aug),Qt_aug.T.dot(b_aug),rcond=None)[0]

    ### solve for steady-state by replacing a row of Qt with 1s
    Qt_new = copy.deepcopy(Qt);Qt_new[[-1],:] = ones
    b = np.zeros(Qt_new[:,[0]].shape[0]);b[-1] = 1
    #sol = np.linalg.solve(Qt_new,b)
    sol = spsolve(Qt_new.tocsr(),b)
        
    return sol

def get_mfpt(x_idxs,y_idxs,p,Qt):

    n = p.nx+1
    ###### MFPT
    # get peaks for MFPT
    if len(x_idxs) == 1:
        mfpt = 0
        I = -1; J = -1
        
    elif len(x_idxs) == 2:
        tauJ = get_full_mfpt(x_idxs,y_idxs,Qt,p)
        #print('x and y for I',x_idxs[0],y_idxs[0],n)
        #print('x and y for J',x_idxs[1],y_idxs[1],n)
        
        I = x_idxs[0]+n*y_idxs[0]; J = x_idxs[1]+n*y_idxs[1]
        mfpt = tauJ[I]
        #print(x_idxs[0],y_idxs[0],'and',x_idxs[1],y_idxs[1])
        
    elif len(x_idxs) == 3:
        
        x_idxs1,y_idxs1,x_idxs2,y_idxs2 = sep(x_idxs,y_idxs)
        #print('after sep',x_idxs1,y_idxs1,x_idxs2,y_idxs2)
        tauJ = get_full_mfpt(x_idxs2,y_idxs2,Qt,p)
        #print('x and y for I',x_idxs2[0],y_idxs2[0],n)
        #print('x and y for J',x_idxs2[1],y_idxs2[1],n)
        I = x_idxs2[0]+n*y_idxs2[0]; J = x_idxs2[1]+n*y_idxs2[1]
        mfpt = tauJ[I]
        #print(x_idxs2[0],y_idxs2[0],'and',x_idxs2[1],y_idxs2[1])

    elif len(x_idxs) == 4:
        # assume symmetric indices, e.g., array([1, 3, 4, 5]), array([5, 4, 3, 1])
        # (1,5) has same peak as (5,1), (3,4) has same peak as (4,3)

        x_idxs1 = np.array([x_idxs[0],x_idxs[-1]])
        y_idxs1 = np.array([y_idxs[0],y_idxs[-1]])
        
        x_idxs2 = np.array([x_idxs[1],x_idxs[-2]])
        y_idxs2 = np.array([y_idxs[1],y_idxs[-2]])

        tauJ1 = get_full_mfpt(x_idxs1,y_idxs1,Qt,p)
        tauJ2 = get_full_mfpt(x_idxs2,y_idxs2,Qt,p)

        I1 = x_idxs1[0]+n*y_idxs1[0]; J1 = x_idxs1[1]+n*y_idxs1[1]
        mfpt1 = tauJ1[I1]
        
        I2 = x_idxs2[0]+n*y_idxs2[0]; J2 = x_idxs2[1]+n*y_idxs2[1]
        mfpt2 = tauJ2[I2]
        print('mfpt1,2',np.round(mfpt1,4),np.round(mfpt2,4),'x1',x_idxs1,'x2',x_idxs2)

        # keep the larger mfpt for simplicity

        if mfpt1 > mfpt2:
            mfpt = mfpt1
            I = I1; J = J1
        else:
            mfpt = mfpt2
            I = I2; J = J2


    else:
        raise ValueError('Invalid length in x_idxs',x_idxs,y_idxs)

    return mfpt,I,J

def main():
    
    parser = argparse.ArgumentParser(description='generate first passage time matrix',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-z','--zeta',default=.4,type=np.float64,
                        help='Set viscous drag')
    parser.add_argument('-a',default=14,type=np.float64,
                        help='Set attach rate')
    parser.add_argument('-b',default=126,type=np.float64,
                        help='Set deattach rate')
    parser.add_argument('-A',default=5,type=np.float64,
                        help='Set deattach rate')
    parser.add_argument('-B',default=5.05,type=np.float64,
                        help='Set deattach rate')
    parser.add_argument('--show-plot',default=False,action='store_true',
                        help='Show plot (and dont save)')
        
    args = parser.parse_args()
    print('args',args)
    #print(args)

    p = Parameters(nx=100,ny=100,A=args.A,B=args.B,
                   al=args.a,be=args.b,ze=args.zeta)

    Qt = get_transition_mat(p)
    
    print('sanity check. sum each column of Qt:',np.sum(np.sum(Qt,axis=0)))

    x_idxs0,y_idxs0 = load_local_idxs(p)
    
    qss = get_ss(p,Qt)
    mat = vec_to_mat(qss,p)
    x_idxs,y_idxs = get_local_idxs(mat,p) # 1 2 or 3 pairs

    print(x_idxs0,x_idxs,'\t',y_idxs0,y_idxs)

    mfpt,I,J = get_mfpt(x_idxs,y_idxs,p,Qt)
    vel = v(x_idxs[0],y_idxs[0],p)
    peaks,peak_idxs = get_peaks(mat,x_idxs,y_idxs,pref='master')

    print('mfpt switch={}, v={},'.format(mfpt,vel))


    # look just... don't ask...
    fname, fname2, fname3 = Q_fname('',p,legacy=True)
        
    np.savetxt(fname.format(args.A,args.B,args.a,args.b,args.zeta),[mfpt])
    np.savetxt(fname2.format(args.A,args.B,args.a,args.b,args.zeta),peaks)
    np.savetxt(fname3.format(args.A,args.B,args.a,args.b,args.zeta),peak_idxs)
    
    #fig, axs = plt.subplots(nrows=1,ncols=1)

    #fsize = 15
    #im = axs.imshow(mat)
    #axs.scatter(x_idxs,y_idxs,c='red',s=50)

    #axs.set_xlim(0,15)
    #axs.set_ylim(0,15)

    #axs.set_xlabel('Up Motors',size=fsize)
    #axs.set_ylabel('Down Motors',size=fsize)
    #axs.scatter(y_idxs,x_idxs,color='tab:red',s=20)
    #axs.set_title('Master Steady-State',size=fsize)

    #axs.tick_params(axis='both',labelsize=fsize)

    #cbar = plt.colorbar(im)
    #cbar.ax.tick_params(labelsize=15)

    #plt.tight_layout()
    #pars = (p.ze,p.al,p.be,p.A,p.B)

    #if args.show_plot:
    #    plt.show()
    #else:
    #    fname = 'q_figs/ze={}_al={}_be={}_A={}_B={}.png'.format(*pars)
    #    print('saving fig to')
    #    print(fname)
        #plt.savefig(fname)
    


if __name__ == "__main__":
    main()
