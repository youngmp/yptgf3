"""
File for telegraph process

in this file, run telegraph process until escape through either boundary.

record escape side, time to escape, and seed in CSV

file name should include initial velocity, lambda, initial position, side length

make serial for now, parallelize later if needed.

run as euler.  gillespie later if needed.

"""

import Q

import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import argparse

#import matplotlib.pyplot as plt

def e0p(L,U,lam):
    """
    probability of escape through right given different domain lengths L
    and starting position 0.
    """
    return U/(U+L*lam)

def t0p(L,U,lam):
    return (3*L*U**3 + 3*L**2*U**2*lam + L**3*U*lam**2)/(3.*U**3*(U + L*lam))

def pp(x0,U,lam,L):
    """
    pp means (p)robability to escape right with (p)ositive initial vel.
    """
    return (U + x0*lam)/(U + L*lam)

def tp(x0,U,lam=2,L=3):
    """
    tp means (t)ime to escape right with (p)ositive initial vel.
    """
    return ((2*(L - x0))/U + ((L - x0)*(L + x0)*lam)/U**2
            + L/(U + L*lam) - x0/(U + x0*lam))/3.

def run_sim(dt=.000001,seed=0,X0=0,V0=-1,
            return_data = False):
    """
    d: dict of parameters
    return_data: return simulation trajectory.
    """
    
    
    in_domain = True
    
    i = 0
    x_prev = X0
    V = V0
    sol = []

    np.random.seed(seed)
    
    while in_domain:

        # update position
        x_next = x_prev + dt*V
        sol.append(x_next)

        # change direction?
        r = np.random.rand()
        if r < dt*lam:
            V *= -1

        # check if boundary hit
        if x_next <= A:
            #print('hit A')
            in_domain = False
            hit = 'L'
            break

        if x_next >= B:
            #print('hit B')
            in_domain = False
            hit = 'R'
            break
        
        x_prev = x_next
        i += 1

    total_time = i*dt    

    if return_data: 
        return hit, total_time, seed, sol
    else:
        return hit, total_time, seed
    

def batch():

    # for x position, loop over 100 seeds.
    x_list = np.arange(A,B,(B-A)/4)
    seeds = np.arange(10)

    time_left = []
    time_right = []
    l_list = []
    r_list = []

    for i,x0 in enumerate(x_list):

        time_left_temp = []
        time_right_temp = []

        l_counter = 0
        r_counter = 0

        for j in seeds:

            hit, pt, seed, sol = run_sim(
                X0=x0,seed=j,V0=121,return_data = True)
            
            if hit == 'R' and x0 == B and False:
                print(hit,pt,seed,sol)
                print(time_right_temp)
            
            if hit == 'L':
                time_left_temp.append(pt)
                l_counter += 1
            else:
                time_right_temp.append(pt)
                r_counter += 1

        print(l_counter,r_counter)
        l_list.append(l_counter)
        r_list.append(r_counter)
        
        time_left.append(np.mean(time_left_temp))
        time_right.append(np.mean(time_right_temp))

    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(x_list,np.array(r_list)/(np.array(l_list)+np.array(r_list)),
             label='simulation')
    ax1.plot(x_list,pp(x_list,121,lam=lam,L=B-A),label='theory')
    
    #ax.plot(x_list,time_left)
    ax2.plot(x_list,time_right,label='simulation')
    ax2.plot(x_list,tp(x_list,121,lam=lam,L=B-A),label='theory')

    
    ax2.legend()
    ax2.set_title('MFPT to exit right given initial positive velocity')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('MFPT')
    
    plt.show()

def calculate_mfpt(pardict,recompute=False,data_dir1='dat_v2/',data_dir2='switch_time_master/'):
    """
    pardict: specify parameter names and values
    """

    #if not(os.path.isdir(data_dir1)):
    #    os.mkdir(data_dir1)
        
    #if not(os.path.isdir(data_dir1+data_dir2)):
    #    os.mkdir(data_dir1+data_dir2)
    
    ft = data_dir1+data_dir2+'A={}_B={}_al={}_be={}_ze={}.txt'

    B = pardict['B']
    al = pardict['al']
    be = pardict['be']
    ze = pardict['ze']

    pars = Q.Parameters(nx=100,ny=100,A=5.,B=B,al=al,be=be,ze=ze)

    fname = ft.format(round(pars.A,5),round(pars.B,5),round(pars.al,5),
                      round(pars.be,5),round(pars.ze,5))

    if os.path.isfile(fname) and not(recompute):
        mfpt = np.loadtxt(fname)
    else:

        Qt = Q.get_transition_mat(pars)

        x_idxs,y_idxs = Q.load_local_idxs(pars) # 1 2 or 3 pairs

        mfpt,I,J = Q.get_mfpt(x_idxs,y_idxs,pars,Qt)
        np.savetxt(fname,[mfpt])

    return mfpt



def calculate_mfpt_only(i,j,p1,p2,pname1,pname2,**kwargs):
    """
    pardict: specify parameter names and values
    """
    
    pardict = {'al':14,'ze':1,'be':126,'B':5.05}
    pardict[pname1] = p1
    pardict[pname2] = p2
    
    B = pardict['B']
    al = pardict['al']
    be = pardict['be']
    ze = pardict['ze']

    pars = Q.Parameters(nx=100,ny=100,A=5.,B=B,al=al,be=be,ze=ze)
    #Qt = Q.get_transition_mat(pars)

    # avoid loading. np.loadtxt and savtext takes too long.
    #x_idxs,y_idxs = Q.load_local_idxs(pars) # 1 2 or 3 pairs


    Qt = Q.get_transition_mat(pars)
    qss = Q.get_ss(pars,Qt)
    mat = Q.vec_to_mat(qss,pars)
    

    out = Q.get_local_idxs_v2(mat,pars,**kwargs) # 1 2 or 3 pairs
    x_idxs = out[:,0]
    y_idxs = out[:,1]
    
    print('out i,j',i,j,pname1,np.round(p1,3),pname2,np.round(p2,3),out,mat[x_idxs,y_idxs])
    
    if False and ((i>=157) and (j<=19)):
        fig,axs = plt.subplots()
        axs.imshow(mat)
        axs.scatter(x_idxs,y_idxs)
        plt.show()



    mfpt,I,J = Q.get_mfpt(x_idxs,y_idxs,pars,Qt)

    #print('i,j',i,j,'mfpt,I,J',mfpt,I,J,'x,y',x_idxs,y_idxs,pname1,pname2,p1,p2)

    return i,j,mfpt

    

def calculate_all_mfpt(pars_and_ranges,recompute=False,linspace=True,**kwargs):

    
    # all pars: ze,al,be,B,A

    # pairs:
    # (ze,al),(ze,be),(ze,B)
    # (al,be),(be,B)
    # (al,B)

    # try generating a surface...
    #par1 = r'$\zeta$'
    #p1 = np.linspace(0.1,1.5,11,endpoint=False) # ze

    par1 = pars_and_ranges['par1']#r'$\alpha$'
    if linspace:
        p1 = np.linspace(*pars_and_ranges['par1_linspace'],endpoint=False)
    else:
        p1 = np.arange(*pars_and_ranges['par1_arange'],endpoint=False)
    #p1 = np.linspace(10,18,12,endpoint=False) # al

    #par2 = r'$\alpha$'
    #p2 = np.linspace(10,18,12,endpoint=False) # al

    #par2 = r'$\beta$'
    #p2 = np.linspace(106,146,12,endpoint=False) # be

    par2 = pars_and_ranges['par2']#r'$B$'
    if linspace:
        p2 = np.linspace(*pars_and_ranges['par2_linspace'],endpoint=False)
    else:
        p2 = np.arange(*pars_and_ranges['par2_arange'],endpoint=False)
        
    #p2 = np.linspace(5.02,5.2,12,endpoint=False) # B
    

    pdict_vals = {r'$\alpha$':14,r'$\zeta$':1,r'$\beta$':126,r'$B$':5.05}

    X,Y = np.meshgrid(p1,p2,indexing='ij')

    mfpt_mat = np.zeros(np.shape(X))
    
    for i in range(len(p1)):
        
        print(round((i+1)/len(p1),3),end='\r')
        for j in range(len(p2)):

            pdict_vals[par1] = X[i,j]
            pdict_vals[par2] = Y[i,j]

            B = pdict_vals[r'$B$']
            al = pdict_vals[r'$\alpha$']
            be = pdict_vals[r'$\beta$']
            ze = pdict_vals[r'$\zeta$']
            
            mfpt = calculate_mfpt(pdict_vals,**kwargs)

            mfpt_mat[i,j] = mfpt

    return X,Y,mfpt_mat
    
def main():
    parser = argparse.ArgumentParser(description='generate switch time matrix',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-z','--zeta',default=.4,type=np.float64,
                        help='Viscous drag')
    parser.add_argument('-a',default=14,type=np.float64,
                        help='Attach rate')
    parser.add_argument('-b',default=126,type=np.float64,
                        help='Deattach rate')
    parser.add_argument('-A',default=5,type=np.float64,
                        help='Attach position')
    parser.add_argument('-B',default=5.05,type=np.float64,
                        help='Deattach position')
    parser.add_argument('--show-plot',default=False,action='store_true',
                        help='Show plot (and dont save)')
    parser.add_argument('-r','--recompute',default=False,action='store_true',
                        help='Recompute data')

    args = parser.parse_args()
    print('args',args)

    pdict_vals = {r'$\alpha$':args.a,r'$\zeta$':args.zeta,r'$\beta$':args.b,r'$B$':args.B}


    
    print(calculate_mfpt_only(0,0,3.3,87.5,'ze','be'))


    # try generating a surface...
    #par1 = r'$\zeta$'
    #p1 = np.linspace(0.1,1.5,11,endpoint=False) # ze

    #par1 = pars_and_ranges['par1']#r'$\alpha$'
    #p1 = np.linspace(*pars_and_ranges['par1_linspace'],endpoint=False)
    #p1 = np.linspace(10,18,12,endpoint=False) # al

    #par2 = r'$\alpha$'
    #p2 = np.linspace(10,18,12,endpoint=False) # al

    #par2 = r'$\beta$'
    #p2 = np.linspace(106,146,12,endpoint=False) # be

    #par2 = pars_and_ranges['par2']#r'$B$'
    #p2 = np.linspace(*pars_and_ranges['par2_linspace'],endpoint=False)
    #p2 = np.linspace(5.02,5.2,12,endpoint=False) # B
    
    """
    pars_and_ranges = {'par1':r'$\alpha$','par1_linspace':(10,18,12),
                       'par2':r'$B$','par2_linspace':(5.02,5.2,12)}

    X,Y,mfpt_mat = calculate_all_mfpt(pars_and_ranges)

    
    fig,axs = plt.subplots()
    im = axs.contourf(X,Y,np.log(mfpt_mat))

    axs.set_xlabel(pars_and_ranges['par1'])
    axs.set_ylabel(pars_and_ranges['par2'])
    
    cb = plt.colorbar(im,ax=axs)

    cb.set_label('log(Time to Switch Velocity)')

    plt.tight_layout()
    plt.show()
    """
    
    #dl = 0
    #dr = 200
    
    #fig,axs = plt.subplots(1,2)
    #x_list = np.linspace(dl,dr,100)
    #axs[0].plot(x_list,pp(x_list,121,lam=1/mfpt,L=dr-dl),label='theory')
    #axs[1].plot(x_list,tp(x_list,121,lam=1/mfpt,L=dr-dl),label='theory')
    #plt.show()
    #batch()
    
if __name__ == "__main__":
    main()
