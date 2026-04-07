"""
figure file for paper. documentation
"""
import dv
import Q
import telegraph
from lubrication import lubrication

import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LightSource
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import brentq
import string
import copy

from scipy.sparse.linalg import spsolve
import multiprocessing
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from itertools import product

import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.legend_handler import HandlerBase
import matplotlib as mpl
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{siunitx}')
plt.rc('pgf.texsystem.pdflatex')

import time
import os

MY_DPI = 96

color_list = ['tab:blue','tab:orange','tab:green',
              'tab:red','tab:purple','tab:brown']

size = 12

exp = np.exp


labels = list(string.ascii_uppercase)

for i in range(len(labels)):
    labels[i] = r'\textbf{{{}}}'.format(labels[i])




class Arrow3D(FancyArrowPatch):
    """
    A class for drawing arrows in 3d plots.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        #xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def cylinder_motors():
    
    T1 = .1
    
    gs = gridspec.GridSpec(nrows=1,ncols=2,wspace=-.1,hspace=.5)
    fig = plt.figure(figsize=(6,3))
    ax11 = fig.add_subplot(gs[0],projection='3d')
    ax22 = fig.add_subplot(gs[1])
    #ax11 = axs[0]
    #ax22 = axs[1]
    ax11.set_aspect('equal')
    ax22.set_aspect('equal')
    
    
    a = lubrication(phi1=.57,Rp=0.96,Rc=1.22,base_radius=1.22,
                    pi3=1,pi4=4.7,pi5=0.1,pi6=10,
                    mu=1.2,T=T1,constriction='piecewise',U0=0.2,
                    dt=0.02,eps=1,
                    F0=50,method='euler')
    a.Z0 = -5/a.Rp
    
    z = np.linspace(-7,7,100)  # dimensional
    r = a.pi1(z)
    th = np.linspace(0,2*np.pi,100)
    
    
    radius_al = 0.25
    
    # draw arrow going into spine
    
    ar1 = Arrow3D([0,0],[0,0],[-5,-1],
                  mutation_scale=10, 
                  lw=2, arrowstyle="-|>", color="k")
    
    ax11.add_artist(ar1)

    # A
    # draw spine
    Z,TH = np.meshgrid(z,th)
    #Z,TH = np.mgrid[-7:7:.1, 0:2*np.pi:.1]
    X = np.zeros_like(Z)
    Y = np.zeros_like(Z)
    #print(np.shape(Z))
    
    for i in range(len(Z[:,0])):
        X[i,:] = a.pi1(Z[i,:])*np.cos(TH[i,:])
        Y[i,:] = a.pi1(Z[i,:])*np.sin(TH[i,:])
    
    ax11.plot_surface(X,Y,Z,alpha=.25)
    
    shifts = np.array([3,-3,0])
    names = ['X','Y','Z']
    size = 2
    
    
    for i in range(3):
        coords = np.zeros((3,2))
        
        coords[:,0] += shifts
        coords[:,1] += shifts
        
        coords[i][1] += size
        arx = Arrow3D(*list(coords),
                      mutation_scale=5, 
                      lw=2, arrowstyle="-|>", color="k")
    
        ax11.text(*list(coords[:,1]),names[i],horizontalalignment='center')
    
        ax11.add_artist(arx)
        
    

    # draw sphere for cap
    b = a.base_radius
    r = np.sqrt(b**2+7**2)
    th2 = np.linspace(0,np.arctan(b/7),100)
    phi = np.linspace(0,2*np.pi,100)
    
    TH2,PHI = np.meshgrid(th2,phi)
    X = r*np.sin(TH2)*np.cos(PHI)
    Y = r*np.sin(TH2)*np.sin(PHI)
    Z = r*np.cos(TH2)
    ax11.plot_surface(X,Y,Z,color='tab:blue',alpha=.5)

    
    # draw sphere vesicle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = np.cos(u)*np.sin(v)
    Y = np.sin(u)*np.sin(v)
    Z = np.cos(v)
    ax11.plot_surface(X,Y,Z,color='gray',alpha=.5)
    
    # label spine head and base
    ax11.text(2.7,-5.5,6,r'\setlength{\parindent}{0pt}Spine Head\\(Closed End)')
    ax11.text(2.5,-5.5,-7,r'\setlength{\parindent}{0pt}Spine Base\\(Open End)')    
    
    # C
    # draw molecular motors
    
    pad = .15
    ax22.fill(a.Rp*np.cos(th),a.Rp*np.sin(th),color='gray',alpha=.5)
    
    ax22.plot([a.Rc+pad,a.Rc+pad],[-2,2],color='tab:blue',alpha=.5)
    ax22.plot([-a.Rc-pad,-a.Rc-pad],[-2,2],color='tab:blue',alpha=.5)
    
    ax22.text(0,0,'Vesicle',ha='center',va='center')
    ax22.text(-a.Rc-3*pad,0,'Spine Wall',rotation=90)
    
    ax22.annotate(r'$Z$',xy=(a.Rp,0),xytext=(a.Rc+4*pad,0),
                  ha='center',va='center',rotation=-90,
                  arrowprops=dict(arrowstyle='-|>',
                                  color='k',lw=1),annotation_clip=False)
    
    # draw axes
    shift = .75
    x_center, y_center = (-1.5,-1.5)

    ax22.annotate(r'X,Y', xy=(x_center,y_center),
                  xytext=(x_center+shift,y_center),
                  va='center',
                  arrowprops=dict(mutation_scale=5, 
                                  arrowstyle='<|-', 
                                  color='k',lw=2),
                  annotation_clip=False)
    
    ax22.annotate(r'Z', xy=(x_center,y_center),
                  xytext=(x_center,y_center+shift),
                  ha='center', 
                  arrowprops=dict(mutation_scale=5,
                                  arrowstyle='<|-',
                                  color='k',lw=2),
                  annotation_clip=False)
    
    
    ax11.set_title(r'\textbf{A} Idealized Spine Geometry',loc='left')
    ax22.set_title(r'\textbf{B} Longitudinal Cross-section',loc='left')
    
    
    ax11.set_axis_off()
    ax22.set_axis_off()
    
    ax22.set_xticks([])
    ax22.set_yticks([])
    
    lo = -4.4
    hi = 4.4
    dx = -.5
    
    ax11.set_xlim(lo-dx,hi+dx)
    ax11.set_ylim(lo-dx,hi+dx)
    ax11.set_zlim(lo,hi)
    
    #ax22.set_xlim(-1.4,1.4)
    #ax22.set_ylim(-1.4-pad,1.4+pad)
    
    ax11.view_init(20,45)
    #fig.align_titles([ax11,ax22])
    #fig.align_titles()
    
    #plt.tight_layout()
    
    return fig


def dv_err(pdict):
    
    ulist = np.arange(0,100)
    dlist = np.arange(0,100)
    
    matsol_err = np.zeros((len(ulist),len(dlist)))
    
    for i in ulist:
        for j in dlist:
            if i > j:
                pdict['u'] = ulist[i]; pdict['d'] = dlist[j]

                v1 = dv.v_num(pdict)
                v2 = dv.v_ana(pdict,o=1)
                
                #print('v1,v2',v1,v2)
                #time.sleep(1)
                
                matsol_err[i,j] = np.abs((v1-v2)/v1)

    # make matrix symmetric
    for i in ulist:
        for j in dlist:
            if i < j:
                matsol_err[i,j] = matsol_err[j,i]

    return matsol_err


def dv_fig():
    """
    calculate error between v1 and v2
    """

    pdict = {'A':5,'B':5.1,'u':15,'d':4,'al':14,'be':126,'ze':.2}

    matsol_err = dv_err(pdict)

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

    axs[0].set_title(r'\textbf{A}',loc='left')
    axs[1].set_title(r'\textbf{B}',loc='left')

    # colorbar nonsense
    divider0 = make_axes_locatable(axs[0])
    divider1 = make_axes_locatable(axs[1])
    
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    
    fig.colorbar(im,cax=cax0)
    fig.colorbar(im2,cax=cax1)
    
    plt.tight_layout()

    return fig


def vel_mats(pdict):
    
    ulist = np.arange(0,100)
    dlist = np.arange(0,100)
    
    matsol_num = np.zeros((len(ulist),len(dlist)))
    matsol_ana = np.zeros((len(ulist),len(dlist)))
    
    for i in ulist:
        for j in dlist:
            if i > j:
                pdict['u'] = ulist[i]; pdict['d'] = dlist[j]

                v1 = dv.v_num(pdict)
                v2 = dv.v_ana(pdict,o=1)
                
                matsol_num[i,j] = v1
                matsol_ana[i,j] = v2

    # make matrix symmetric
    for i in ulist:
        for j in dlist:
            if i < j:
                matsol_num[i,j] = -matsol_num[j,i]
                matsol_ana[i,j] = -matsol_ana[j,i]

    return matsol_num, matsol_ana




def vel_fig():
    """
    calculate error between v1 and v2
    """

    pdict = {'A':5,'B':5.1,'u':15,'d':4,'al':14,'be':126,'ze':.2}

    matsol_num, matsol_ana = vel_mats(pdict)

    fig, axs = plt.subplots(1,2,figsize=(8,3.5))

    #im = axs[0].imshow(matsol_num,origin='lower')
    im = axs[0].contourf(matsol_num,origin='lower',levels=np.arange(-500,600,100))
    #im2 = axs[1].imshow(matsol_ana,origin='lower')
    im2 = axs[1].contourf(matsol_ana,origin='lower',levels=np.arange(-500,600,100))
    
    # Add the patch to the Axes
    axs[0].set_xlabel(r'$D$')
    axs[1].set_xlabel(r'$D$')
    
    axs[0].set_ylabel(r'$U$')
    axs[1].set_ylabel(r'$U$')

    axs[0].set_title(r'\textbf{A}',loc='left')
    axs[1].set_title(r'\textbf{B}',loc='left')
    
    t0 = axs[0].get_title()
    t0 += 'Numerical Velocity'
    axs[0].set_title(t0)
    
    t1 = axs[1].get_title()
    t1 += 'Analytical Velocity'
    axs[1].set_title(t1)
    
    # colorbar nonsense
    #divider0 = make_axes_locatable(axs[0])
    divider1 = make_axes_locatable(axs[1])
    
    #cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    
    #fig.colorbar(im,cax=cax0)
    cbar = fig.colorbar(im2,cax=cax1)
    
    cbar.ax.set_ylabel(r'\si{nm/s}')
    
    plt.tight_layout()

    return fig


def v_switch_mfpt():
    """
    get mfpt for agents and master...
    """
    fig, axs = plt.subplots(1,2,figsize=(8,3))

    # invisble plot with other legend
    ax2 = axs[0].twinx(); styles=['-','--']; labels_temp=['Master','Agents']
    for ss, sty in enumerate(styles):
        ax2.plot(np.NaN, np.NaN, ls=styles[ss],
                 label=labels_temp[ss], c='black')
        ax2.get_yaxis().set_visible(False)
    
    B_list = [5.02,5.05,5.1]#np.round(np.arange(5.05,5.1,.01),2)
    A = 5
    al = 14
    be = 126
    ze_max = 15

    p2 = np.loadtxt('data/diagram_b=5.1_2par.dat')
    
    #ze_scale = 1e-6*(6*np.pi*a.Rp*a.mu)
    ze_list_a = np.round(np.arange(0.1,ze_max,.1),1)
    ze_list_a2 = np.round(np.arange(0.1,ze_max,.1),1)
    ze_list_m = np.round(np.arange(0.1,ze_max,.1),1)

    ze_bif = np.zeros(len(B_list))
    
    for j,B in enumerate(B_list):
    
         
        if B == 5.02:
            ze_list_a = np.round(np.arange(0.1,ze_max,.2),1)
            ze_list_a2 = np.round(np.arange(0.1,ze_max,.2),1)
            ze_list_m = np.round(np.arange(0.1,ze_max,.2),1)
        elif B == 5.05:
            ze_list_a = np.round(np.arange(.2,7.2,.2),1)
            ze_list_a2 = np.round(np.arange(.2,7.2,.2),1)
            ze_list_m = np.round(np.arange(.1,7.2,.1),1)
        elif B == 5.05:
            ze_list_a = np.round(np.arange(0.,7.2,.2),1)
            ze_list_a2 = np.round(np.arange(0.,7.2,.2),1)
            ze_list_m = np.round(np.arange(0.,7.2,.2),1)


        # get z value given b
        idx = np.argmin(np.abs(p2[:,1]-B))
        ze_bif[j] = p2[idx,0]
    
        #B = 5.1#B_list[0]
        print(B)

        # agents
        mfpt_list_a = []

        # average mfpt over all seeds for each ze
        # data from cluster, agent-based model
        if B == 5.01:
            ze_list = ze_list_a2
        else:
            ze_list = ze_list_a
            
        for i in range(len(ze_list)):
            ze = ze_list[i]

            mfpt_temp = 0
            
            seed_count = 0
            for k in range(50):

                fname = ('dat/switch_time_agents_T=10.0_'
                         'A={}_B={}_al={}_be={}_ze={}_seed={}')
                    
                try:
                    mfpt_temp += np.loadtxt(fname.format(A,B,al,be,ze,k))
                    seed_count += 1
                except IOError:
                    print('file missing,', fname.format(A,B,al,be,ze,k))
                    pass
            
            mfpt = mfpt_temp/seed_count
            mfpt_list_a.append(mfpt)

        # master        
        mfpt_list_m = []
        mfpt_peaks = np.zeros((len(ze_list_m),3))
        mfpt_peak_idxs = np.zeros((len(ze_list_m),3))

        # mfpt estimated from master equation
        for i in range(len(ze_list_m)):
            ze = ze_list_m[i]
            fname = 'dat/switch_time_master_A={}_B={}_al={}_be={}_ze={}'
            fname2 = 'dat/master_peaks_A={}_B={}_al={}_be={}_ze={}.txt'
            fname3 = 'dat/master_peak_idxs_A={}_B={}_al={}_be={}_ze={}.txt'

            mfpt = np.loadtxt(fname.format(A,B,al,be,ze))
            peaks = np.loadtxt(fname2.format(A,B,al,be,ze))
            peak_idxs = np.loadtxt(fname3.format(A,B,al,be,ze))
            
            if mfpt == 0:
                mfpt = np.nan

            mfpt_list_m.append(mfpt)
            #mfpt_peaks.append(peaks)
            mfpt_peaks[i,:] = peaks
            mfpt_peak_idxs[i,:] = peak_idxs[:,0]

        axs[0].plot(ze_list_m,mfpt_list_m,label='B='+str(B),color=color_list[j])
        axs[0].plot(ze_list,mfpt_list_a,color=color_list[j],ls='--')

    # label biologically relevant region
    axs[0].axvline(1,lw=1,color='gray',zorder=-3)
    axs[0].axvspan(1,20,alpha=.1,lw=1,color='tab:blue',zorder=-3)
    axs[0].text(.52,.55,'Biologically relevant\n regime',
                transform=axs[0].transAxes,fontsize=12)

    # plot the 2 par diagram    
    axs[1].plot(p2[:,0],p2[:,1],color='k')
    axs[1].fill_between(p2[:,0],p2[:,1],color='tab:blue',alpha=.1)
    axs[1].text(.1,.15,'Bistability',transform=axs[1].transAxes,fontsize=12)
    #axs[1].text(.1,.1,'Bistability',transform=axs[1].transAxes)
    

    # label corresponding saddle nodes
    marks = ['^','s','o']
    for ll in range(len(B_list)):
        axs[0].scatter([ze_bif[ll]],[2.5e-2],s=50,marker=marks[ll],
                       zorder=5)
        axs[1].scatter([ze_bif[ll]],B_list[ll],s=50,marker=marks[ll],
                       zorder=5)
    
    axs[0].set_xscale('log');axs[1].set_xscale('log')
    axs[0].set_yscale('log')
    
    #axs[0][1].set_yscale('log')
    #axs[0][2].set_yscale('log')

    #axs[0].set_xlabel(r'$\zeta$ (\si{kg/s})',fontsize=size)
    axs[0].set_xlabel(r'$\zeta$ (\si{mg/s})',fontsize=size)
    axs[1].set_xlabel(r'$\zeta$ (\si{mg/s})',fontsize=size)
    #axs[0][1].set_xlabel(r'$\zeta$')
    #axs[0][2].set_xlabel(r'$\zeta$')

    axs[0].set_ylabel(r'MFPT Velocity Switch (\si{s})',fontsize=size)
    axs[1].set_ylabel(r'$B$',fontsize=size)
    #axs[0][1].set_ylabel(r'Abs. Error')
    #axs[0][2].set_ylabel(r'Rel. Error')

    
    axs[0].set_xlim(9e-2,ze_max)
    axs[1].set_xlim(9e-2,ze_max)

    axs[0].set_ylim(0.02,3)
    axs[1].set_ylim(4.95,6)

    
    for i in range(len(axs)):
        axs[i].set_title(labels[i],loc='left')

    
    leg = axs[0].legend(loc='upper center', bbox_to_anchor=(.5,1.2),
                        columnspacing=.5,fancybox=True,shadow=False,ncol=3)

    leg.set_zorder(10)
    #axs[0].set_zorder(1)

    ax2.legend()

    
    plt.tight_layout()

    # constriction label
    yshift_arrow = -.18
    yshift_text = -.16
    label = axs[0].text(.125, yshift_text, 'Looser', fontsize = 12, 
                     ha = 'center',transform=axs[0].transAxes)
    axs[0].annotate('',xy=(0,yshift_arrow),xycoords='axes fraction',
                 xytext=(.25,yshift_arrow),
                 arrowprops=dict(arrowstyle="-|>"),ha='center')

    label = axs[0].text(1-.125,yshift_text, 'Tighter', fontsize = 12, 
                     ha = 'center',transform=axs[0].transAxes)
    axs[0].annotate('',xy=(1,yshift_arrow),xycoords='axes fraction',
                 xytext=(1-.25,yshift_arrow),
                 arrowprops=dict(arrowstyle="-|>"),ha='center')


    #plt.subplots_adjust(wspace=.5)

    return fig

def fp_steady():
    fig, axs = plt.subplots(1,2,figsize=(8,4))

    p1_b501 = np.loadtxt('data/diagram_b=5.01_1par.dat')
    p1_b505 = np.loadtxt('data/diagram_b=5.05_1par.dat')
    p1_b51 = np.loadtxt('data/diagram_b=5.1_1par.dat')
    
    p2_b501 = np.loadtxt('data/diagram_b=5.01_2par.dat')
    p2_b505 = np.loadtxt('data/diagram_b=5.05_2par.dat')
    p2_b51 = np.loadtxt('data/diagram_b=5.1_2par.dat')

    axs[0].plot(p1_b501[:,0],p1_b501[:,1])
    axs[0].plot(p1_b505[:,0],p1_b505[:,1]*100)
    axs[0].plot(p1_b51[:,0],p1_b51[:,1])
    
    axs[1].scatter(p2_b501[:,0],p2_b501[:,1])
    axs[1].scatter(p2_b505[:,0],p2_b505[:,1])
    axs[1].scatter(p2_b51[:,0],p2_b51[:,1])
    
    #axs[0].set_xlim(0,5)
    axs[1].set_xlim(0,5)
    
    return fig

def get_agent_peak_idxs(ze_list,A=5,B=5.1,al=14,be=126):
        
    # get and plot agent peak idxs
    mfpt_peaks = np.zeros((len(ze_list),3))
    mfpt_peak_idxs = np.zeros((len(ze_list),3))

    for i in range(len(ze_list)):
        ze = ze_list[i]

        pref = 'agents'
        fname2 = 'dat/'+pref+'_peaks_A={}_B={}_al={}_be={}_ze={}.txt'
        fname3 = 'dat/'+pref+'_peak_idxs_A={}_B={}_al={}_be={}_ze={}.txt'

        peaks = np.loadtxt(fname2.format(A,B,al,be,ze))
        peak_idxs = np.loadtxt(fname3.format(A,B,al,be,ze))

        mfpt_peaks[i,:] = peaks
        mfpt_peak_idxs[i,:] = peak_idxs[:,0]

    return mfpt_peaks, mfpt_peak_idxs

def steady_state_examples():

    fig,axs = plt.subplots(1,2,figsize=(8,3.5))

    B = 5.04
    ze = 3.1
    N=100;al=14;be=126
    n_eq = N*al/(al+be)

    # needs to be dropbox path which might change based on the computer....
    # try different paths, default to local (since people are most likely to have downloaded this from the repo)
    homes = ['./agents_hist/',os.path.expanduser('~')+'Dropbox/data/agents_hist/','D:/Dropbox/data/agents_hist/',
             'E:/Dropbox/data/agents_hist/']
    Bs = str(B)

    
    file_not_found = True
    count = 0
    while file_not_found:
        
        home = homes[count]
        if os.path.isfile(fname):
            file_not_found = False
        count += 1
    
    fname = home+'master_hist_T=10.0_A=5_B='+Bs+'_al=14_be=126_ze={}_seed={}.txt'
    ## agents steady state
    # despite the name, this is the histogram for agents
    

    mattot = np.zeros((100,100))

    for seed in range(50):
        mat = np.loadtxt(fname.format(ze,seed))
        mattot += mat

    mattot /= 50

    ## master steady-state + peak
    p = Q.Parameters(B=B,al=al,be=be,ze=ze)
    Qt = Q.get_transition_mat(p)
    ones = np.array([np.ones((p.nx+1)*(p.ny+1))])
    
    Qt_new = copy.deepcopy(Qt);Qt_new[[-1],:] = ones
    b = np.zeros(Qt_new[:,[0]].shape[0]);b[-1] = 1
    sol = spsolve(Qt_new.tocsr(),b)
    mat_m = Q.vec_to_mat(sol,p)
    
    axs[0].imshow(mattot)
    axs[1].imshow(mat_m)

    # mark agents peak
    pref = 'agents'
    fname3 = 'dat/'+pref+'_peak_idxs_A=5_B={}_al={}_be={}_ze={}.txt'
    peak_idxs = np.loadtxt(fname3.format(B,al,be,ze))

    axs[0].scatter(peak_idxs[1,0],peak_idxs[1,1],
                   zorder=2,color='tab:red',label='Peaks',
                   marker='X',s=60)
    axs[0].scatter(peak_idxs[2,0],peak_idxs[2,1],
                   zorder=2,color='tab:red',label='Peaks',
                   marker='*',s=60)
    axs[0].scatter([peak_idxs[0,0]],[peak_idxs[0,1]],zorder=2,color='tab:red',
                   marker='D',s=60)

    #_, peak_idxs_m = get_master_peak_idxs([ze],A=5,B=B,al=14,be=126)

    #peak_idxs_m = [[[],[],[]]]

    #print(peak_idxs_m[0]) # currently showing nan due to updates in peak detection
    # insert points manually
    axs[1].scatter([8],[8],
                   zorder=3,color='tab:red',label='Peaks',
                   marker='X',s=60)
    axs[1].scatter([10],[4],
                   zorder=3,color='tab:red',label='Peaks',
                   marker='*',s=60)
    axs[1].scatter([4],[10],
                   zorder=3,color='tab:red',
                   marker='D',s=60)

    
    axs[0].set_xlim(0,2*n_eq)
    axs[1].set_xlim(0,2*n_eq)

    axs[0].set_ylim(0,2*n_eq)
    axs[1].set_ylim(0,2*n_eq)
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    lab2 = ['Agents Steady-State', 'Master Steady-State']
    for i in range(2):
        axs[i].set_xlabel(r'Up Motors ($U$)')
        axs[i].set_ylabel(r'Down Motors ($D$)')
        axs[i].set_title(labels[i],loc='left')

        t1 = axs[i].get_title()
        axs[i].set_title(t1+lab2[i])

        # fp bound
        axs[i].plot([0,n_eq],[n_eq,n_eq],color='white',ls='--')
        axs[i].plot([n_eq,n_eq],[0,n_eq],color='white',ls='--')
        
    #fig.colorbar(axs[0].images[0],ax=axs[0],fraction=0.046)
    fig.colorbar(axs[1].images[0],ax=axs[1],fraction=0.046)

    #axs[0].legend(framealpha=1)

    plt.tight_layout()

    return fig

def get_master_peak_idxs(ze_list,A=5,B=5.1,al=14,be=126):
        
    # get and plot agent peak idxs
    mfpt_peaks = np.zeros((len(ze_list),3))
    mfpt_peak_idxs = np.zeros((len(ze_list),3))

    for i in range(len(ze_list)):
        ze = ze_list[i]

        pref = 'master'
        fname2 = 'dat/'+pref+'_peaks_A={}_B={}_al={}_be={}_ze={}.txt'
        fname3 = 'dat/'+pref+'_peak_idxs_A={}_B={}_al={}_be={}_ze={}.txt'

        peaks = np.loadtxt(fname2.format(A,B,al,be,ze))
        peak_idxs = np.loadtxt(fname3.format(A,B,al,be,ze))

        mfpt_peaks[i,:] = peaks
        mfpt_peak_idxs[i,:] = peak_idxs[:,0]

    return mfpt_peaks, mfpt_peak_idxs

def steady_states():

    fig, axs = plt.subplots(2,3,figsize=(8,5))
    
    # plot fp peak idxs
    #p1_b501 = np.loadtxt('data/diagram_b=5.01_1par.dat')
    p1_b502 = np.loadtxt('data/diagram_b=5.02_1par.dat')
    p1_b505 = np.loadtxt('data/diagram_b=5.05_1par.dat')
    p1_b51 = np.loadtxt('data/diagram_b=5.1_1par.dat')

    # decorate fp with stable and unstable
    idx = np.argmax(p1_b502[:,0])
    p1, = axs[0,0].plot(p1_b502[:idx,0],p1_b502[:idx,1],
                        color='gray',label='F-P',lw=2)
    p2, = axs[0,0].plot(p1_b502[idx:,0],p1_b502[idx:,1],
                        color='gray',ls='--',lw=2)

    axs[1,0].plot(p1_b502[:idx,0],p1_b502[:idx,1],color='gray',label='F-P',lw=2)
    axs[1,0].plot(p1_b502[idx:,0],p1_b502[idx:,1],color='gray',ls='--',lw=2)

    #axs[1,0].plot(p1_b502[:idx,0],p1_b502[:idx,2],color='k',label='F-P',lw=2)
    #axs[1,0].plot(p1_b502[idx:,0],p1_b502[idx:,2],color='k',ls='--',lw=2)

    idx = np.argmax(p1_b505[:,0])
    axs[0,1].plot(p1_b505[:idx,0],p1_b505[:idx,1]*100,
                  color='gray',label='F-P',lw=2)
    axs[0,1].plot(p1_b505[idx:,0],p1_b505[idx:,1]*100,
                  color='gray',ls='--',lw=2)

    axs[1,1].plot(p1_b505[:idx,0],p1_b505[:idx,1]*100,
                  color='gray',label='F-P',lw=2)
    axs[1,1].plot(p1_b505[idx:,0],p1_b505[idx:,1]*100,
                  color='gray',ls='--',lw=2)

    idx = np.argmax(p1_b51[:,0])
    axs[0,2].plot(p1_b51[idx:,0],p1_b51[idx:,1],color='gray',label='F-P',lw=2)
    axs[0,2].plot(p1_b51[:idx,0],p1_b51[:idx,1],color='gray',ls='--',lw=2)

    axs[1,2].plot(p1_b51[idx:,0],p1_b51[idx:,1],color='gray',label='F-P',lw=2)
    axs[1,2].plot(p1_b51[:idx,0],p1_b51[:idx,1],color='gray',ls='--',lw=2)

    # manually add stable line
    N=100;al=14;be=126
    n_eq = N*al/(al+be)
    for i in range(3):
        axs[0,i].plot([0,20],[n_eq,n_eq],color='gray',lw=2)
        axs[1,i].plot([0,20],[n_eq,n_eq],color='gray',lw=2)

    # plot master peak idxs    
    ze_list_m = np.round(np.arange(0.1,20,.1),1)
    mfpt_peaks, mfpt_peak_idxs = get_master_peak_idxs(ze_list_m,B=5.02)
    p3, = axs[0,0].plot(ze_list_m,mfpt_peak_idxs[:,0],
                        marker='*',markevery=10,markersize=10,alpha=.75,
                        color='tab:blue',label='Master')
    p4, = axs[0,0].plot(ze_list_m,mfpt_peak_idxs[:,1],
                        marker='X',markevery=10,markersize=7,alpha=.75,
                        color='tab:blue')
    p5, = axs[0,0].plot(ze_list_m,mfpt_peak_idxs[:,2],
                        marker='D',markevery=10,markersize=6,alpha=.75,
                        color='tab:blue')

    #axs[0,0].legend([(p1,p2),(p3,p4,p5)],['F-P','Master'],
    #                handler_map={tuple:HandlerTuple(ndivide=None)})

    #ze_list_m = np.round(np.arange(0.1,5,.1),1)
    mfpt_peaks, mfpt_peak_idxs = get_master_peak_idxs(ze_list_m,B=5.05)
    axs[0,1].plot(ze_list_m,mfpt_peak_idxs[:,0],
                  marker='*',markevery=5,markersize=10,alpha=.75,
                  color='tab:blue',label='Master')
    axs[0,1].plot(ze_list_m,mfpt_peak_idxs[:,1],
                  marker='X',markevery=5,markersize=7,alpha=.75,
                  color='tab:blue')
    axs[0,1].plot(ze_list_m,mfpt_peak_idxs[:,2],
                  marker='D',markevery=5,markersize=6,alpha=.75,
                  color='tab:blue')

    #ze_list_m = np.round(np.arange(0.0,2,.1),1)
    mfpt_peaks, mfpt_peak_idxs = get_master_peak_idxs(ze_list_m,B=5.1)
    axs[0,2].plot(ze_list_m,mfpt_peak_idxs[:,0],
                  marker='*',markevery=2,markersize=10,alpha=.75,
                  color='tab:blue',label='Master')
    axs[0,2].plot(ze_list_m,mfpt_peak_idxs[:,1],
                  marker='X',markevery=2,markersize=7,alpha=.75,
                  color='tab:blue')
    axs[0,2].plot(ze_list_m,mfpt_peak_idxs[:,2],
                  marker='D',markevery=2,markersize=6,alpha=.75,
                  color='tab:blue')

    ### plot agents peak idxs
    # D
    ze_list = np.round(np.arange(0.1,11,.2),1)
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.02)
    axs[1,0].plot(ze_list,mfpt_peak_idxs[:,0],
                        marker='*',markevery=5,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,0].plot(ze_list,mfpt_peak_idxs[:,1],
                        marker='X',markevery=5,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,0].plot(ze_list,mfpt_peak_idxs[:,2],
                        marker='D',markevery=5,markersize=6,alpha=.75,
                        color='tab:orange')

    # E
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.05)
    

    axs[1,1].plot(ze_list,mfpt_peak_idxs[:,0],
                        marker='*',markevery=2,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,1].plot(ze_list,mfpt_peak_idxs[:,1],
                        marker='X',markevery=2,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,1].plot(ze_list,mfpt_peak_idxs[:,2],
                        marker='D',markevery=2,markersize=6,alpha=.75,
                        color='tab:orange')

    # F
    ze_list = np.round(np.arange(0.1,4,.2),1)
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.1)
    #print('ze,peakidxs,')
    #for ii,jj in zip(ze_list,mfpt_peak_idxs[:,0]):
    #    print(ii,jj)
    axs[1,2].plot(ze_list,mfpt_peak_idxs[:,0],
                        marker='*',markevery=1,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,2].plot(ze_list,mfpt_peak_idxs[:,1],
                        marker='X',markevery=1,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,2].plot(ze_list,mfpt_peak_idxs[:,2],
                        marker='D',markevery=1,markersize=6,alpha=.75,
                        color='tab:orange')

    xhi1 = np.amax(p1_b502[:,0])+np.amax(p1_b502[:,0])/10
    xhi2 = np.amax(p1_b505[:,0])+np.amax(p1_b505[:,0])/10
    xhi3 = np.amax(p1_b51[:,0])+np.amax(p1_b51[:,0])/10

    yhi = n_eq + n_eq/5.5

    for j in range(2):
        axs[j,0].set_xlim(-np.amax(p1_b502[:,0])/30,xhi1)
        axs[j,1].set_xlim(-np.amax(p1_b505[:,0])/30,xhi2)
        axs[j,2].set_xlim(-np.amax(p1_b51[:,0])/30,xhi3)

        axs[j,0].set_ylim(-n_eq/10,yhi)
        axs[j,1].set_ylim(-n_eq/10,yhi)
        axs[j,2].set_ylim(-n_eq/10,yhi)

        axs[j,0].legend()

        axs[j,0].set_ylabel('$D$')

        

    nr,nc = np.shape(axs)

    count = 0
    for i in range(nr):
        
        for j in range(nc):
            axs[i,j].set_title(labels[count],loc='left')
            axs[i,j].set_xlabel(r'$\zeta$')
            axs[i,j].yaxis.set_major_locator(MaxNLocator(integer=True))

            count += 1


    titles = [[r'$B=5.02$',r'$B=5.05$',r'$B=5.1$'],
              [r'$B=5.02$',r'$B=5.05$',r'$B=5.1$']]
    for i in range(nr):
        for j in range(nc):
            t1 = axs[i,j].get_title()
            lab = t1 + titles[i][j]
            axs[i,j].set_title(lab)


    plt.tight_layout()
    
    return fig
                                


def collect(x,y,zs=None,use_nonan=True,lwstart=2.,lwend=2.,zorder=1.,cmapmax=1.,cmapmin=0.,cmap='viridis'):
    """
    add desired line properties
    """
    x = np.real(x)
    y = np.real(y)
    
    x_nonan = x[(~np.isnan(x))*(~np.isnan(y))]
    y_nonan = y[(~np.isnan(x))*(~np.isnan(y))]
    
    if use_nonan:
        points = np.array([x_nonan, y_nonan]).T.reshape(-1, 1, 2)
    else:
        points = np.array([x, y]).T.reshape(-1, 1, 2)


    lwidths = np.linspace(lwstart,lwend,len(x_nonan))

    cmap = plt.get_cmap(cmap)
    #my_cmap = truncate_colormap(cmap,gshift/ga[-1],cmapmax)
    my_cmap = truncate_colormap(cmap,cmapmin,cmapmax)

    
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=lwidths,cmap=my_cmap, norm=plt.Normalize(zs.min(), zs.max()),zorder=zorder)
    
    #points = np.array([x, y]).T.reshape(-1, 1, 2)
    #segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    #lc = LineCollection(segments, cmap=plt.get_cmap('copper'),
    #                    linewidths=1+np.linspace(0,1,len(x)-1)
    #                    #norm=plt.Normalize(0, 1)
    #)
    
    #lc.set_array(np.sqrt(x**2+y**2))
    lc.set_array(zs)
    
    return lc

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def steady_states_complete():
    
    fig, axs = plt.subplots(2,3,figsize=(8,5))


    collect_kws = dict(lwstart=1,lwend=4,zorder=10)
    
    # plot fp peak idxs
    #p1_b501 = np.loadtxt('data/diagram_b=5.01_1par.dat')
    p1_b502 = np.loadtxt('data/diagram_b=5.02_1par.dat')
    p1_b505 = np.loadtxt('data/diagram_b=5.05_1par.dat')
    p1_b51 = np.loadtxt('data/diagram_b=5.1_1par.dat')

    # decorate fp with stable and unstable
    idx = np.argmax(p1_b502[:,0])
    #p1, = axs[0,0].plot(p1_b502[:idx,0],p1_b502[:idx,1],
    #                    color='gray',label='F-P',lw=2)
    #p2, = axs[0,0].plot(p1_b502[idx:,0],p1_b502[idx:,1],
    #                    color='gray',ls='--',lw=2)

    # manually define stable line
    N=100;al=14;be=126
    n_eq = N*al/(al+be)
    
    ########### 5.02 (A,D)
    zes = p1_b502[:,0]
    y = p1_b502[:idx,1]
    x = np.ones(len(y))*n_eq
    z = zes

    # off-diagonal curve
    line = axs[0,0].add_collection(collect(x,y,z,**collect_kws))
    line = axs[0,0].add_collection(collect(y,x,z,**collect_kws))

    # on-diagonal curve
    dat = np.loadtxt('ode/bif_b=5.02_fixed.dat')
    skip_first = 0
    x_mid = dat[skip_first:,1]*100; y_mid = x_mid; z_mid = dat[skip_first:,0]
    axs[0,0].add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[1,0].add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    #axs[0,0].plot(x_mid,y_mid,z_mid,zorder=10)

    fig.colorbar(line, ax=axs[0,0], label=r'$\zeta$ (\si{mg/s})')

    # plot master peak idxs (A)
    ze_list_m = np.round(np.arange(0.1,20,.1),1)
    mfpt_peaks, mfpt_peak_idxs = get_master_peak_idxs(ze_list_m,B=5.02)
    p3, = axs[0,0].plot(mfpt_peak_idxs[:,0],mfpt_peak_idxs[:,2],
                        marker='*',markevery=10,markersize=10,alpha=.75,
                        color='tab:blue',label='Master')
    p4, = axs[0,0].plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                        marker='X',markevery=10,markersize=7,alpha=.75,
                        color='tab:blue')
    p5, = axs[0,0].plot(mfpt_peak_idxs[:,2],mfpt_peak_idxs[:,0],
                        marker='D',markevery=10,markersize=6,alpha=.75,
                        color='tab:blue')
    
    # inset to show middle branch for FP
    x1, x2, y1, y2 = 7.5, 8.5, 7.5, 8.5  # subregion of the original image
    axins00 = axs[0,0].inset_axes(
        [0.02, 0.02, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[],xticks=[],yticks=[])
    axins00.add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[0,0].indicate_inset_zoom(axins00, edgecolor="black")

    axins00.plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                 marker='X',markevery=10,markersize=7,alpha=.75,
                 color='tab:blue')

    # plot agents peak idxs (D)
    ze_list = np.round(np.arange(0.1,11,.2),1)
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.02)
    axs[1,0].plot(mfpt_peak_idxs[:,0],mfpt_peak_idxs[:,2],
                        marker='*',markevery=4,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,0].plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                        marker='X',markevery=4,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,0].plot(mfpt_peak_idxs[:,2],mfpt_peak_idxs[:,0],
                        marker='D',markevery=4,markersize=6,alpha=.75,
                        color='tab:orange')

    
    # inset to show middle branch for FP
    x1, x2, y1, y2 = 7.5, 8.5, 7.5, 8.5  # subregion of the original image
    axins10 = axs[1,0].inset_axes(
        [0.02, 0.02, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[],xticks=[],yticks=[])
    axins10.add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[1,0].indicate_inset_zoom(axins10, edgecolor="black")

    axins10.plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                 marker='X',markevery=4,markersize=7,alpha=.75,
                 color='tab:orange')

    # off-diagonal
    line = axs[1,0].add_collection(collect(x,y,z,**collect_kws))
    line = axs[1,0].add_collection(collect(y,x,z,**collect_kws))

    fig.colorbar(line, ax=axs[1,0], label=r'$\zeta$ (\si{mg/s})')


    ############## 5.05 (B,E)
    idx = np.argmax(p1_b505[:,0])
    zes = p1_b505[:,0]
    y = p1_b505[:idx,1]*100
    x = np.ones(len(y))*n_eq
    z = zes

    # off-diagonal curve
    line = axs[0,1].add_collection(collect(x,y,z,**collect_kws))
    line = axs[0,1].add_collection(collect(y,x,z,**collect_kws))
    
    # on-diagonal curve
    dat = np.loadtxt('ode/bif_b=5.05_fixed.dat')
    skip_first = 0
    x_mid = dat[skip_first:,1]*100; y_mid = x_mid; z_mid = dat[skip_first:,0]
    axs[0,1].add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[1,1].add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    #axs[0,0].plot(x_mid,y_mid,z_mid,zorder=10)

    fig.colorbar(line, ax=axs[0,1], label=r'$\zeta$ (\si{mg/s})')
    

    line = axs[1,1].add_collection(collect(x,y,z,**collect_kws))
    line = axs[1,1].add_collection(collect(y,x,z,**collect_kws))

    fig.colorbar(line, ax=axs[1,1], label=r'$\zeta$ (\si{mg/s})')

    
    # plot master peak idxs (B)
    ze_list_m = np.round(np.arange(0.1,20,.1),1)
    mfpt_peaks, mfpt_peak_idxs = get_master_peak_idxs(ze_list_m,B=5.05)
    p3, = axs[0,1].plot(mfpt_peak_idxs[:,0],mfpt_peak_idxs[:,2],
                        marker='*',markevery=10,markersize=10,alpha=.75,
                        color='tab:blue',label='Master')
    p4, = axs[0,1].plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                        marker='X',markevery=10,markersize=7,alpha=.75,
                        color='tab:blue')
    p5, = axs[0,1].plot(mfpt_peak_idxs[:,2],mfpt_peak_idxs[:,0],
                        marker='D',markevery=10,markersize=6,alpha=.75,
                        color='tab:blue')
    
    # inset to show middle branch for FP
    x1, x2, y1, y2 = 7.5, 8.5, 7.5, 8.5  # subregion of the original image
    axins01 = axs[0,1].inset_axes(
        [0.02, 0.02, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[],xticks=[],yticks=[])
    axins01.add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[0,1].indicate_inset_zoom(axins01, edgecolor="black")

    axins01.plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                 marker='X',markevery=10,markersize=7,alpha=.75,
                 color='tab:blue')

    
    # plot agents peak idxs (E)
    ze_list = np.round(np.arange(0.1,11,.2),1)
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.05)
    axs[1,1].plot(mfpt_peak_idxs[:,0],mfpt_peak_idxs[:,2],
                        marker='*',markevery=4,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,1].plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                        marker='X',markevery=4,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,1].plot(mfpt_peak_idxs[:,2],mfpt_peak_idxs[:,0],
                        marker='D',markevery=4,markersize=6,alpha=.75,
                        color='tab:orange')

    # inset to show middle branch for FP
    x1, x2, y1, y2 = 7.5, 8.5, 7.5, 8.5  # subregion of the original image
    axins11 = axs[1,1].inset_axes(
        [0.02, 0.02, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[],xticks=[],yticks=[])
    axins11.add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[1,1].indicate_inset_zoom(axins11, edgecolor="black")

    axins11.plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                 marker='X',markevery=4,markersize=7,alpha=.75,
                 color='tab:orange')


    ############## 5.1 (C,F)
    idx = np.argmax(p1_b51[:,0])
    
    y = p1_b51[idx:,1][::-1]
    zes = p1_b51[idx:,0][::-1]
    x = np.ones(len(y))*n_eq
    z = zes

    # off-diagonal curve
    line = axs[0,2].add_collection(collect(x,y,z,**collect_kws))
    line = axs[0,2].add_collection(collect(y,x,z,**collect_kws))

    # on-diagonal curve
    dat = np.loadtxt('ode/bif_b=5.1_fixed.dat')
    skip_first = 0
    x_mid = dat[skip_first:,1]*100; y_mid = x_mid; z_mid = dat[skip_first:,0]
    axs[0,2].add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[1,2].add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    #axs[0,0].plot(x_mid,y_mid,z_mid,zorder=10)


    fig.colorbar(line, ax=axs[0,2], label=r'$\zeta$ (\si{mg/s})')

    line = axs[1,2].add_collection(collect(x,y,z,**collect_kws))
    line = axs[1,2].add_collection(collect(y,x,z,**collect_kws))

    fig.colorbar(line, ax=axs[1,2], label=r'$\zeta$ (\si{mg/s})')

    # plot master peak idxs (C)
    ze_list_m = np.round(np.arange(0.1,20,.1),1)
    mfpt_peaks, mfpt_peak_idxs = get_master_peak_idxs(ze_list_m,B=5.1)
    p3, = axs[0,2].plot(mfpt_peak_idxs[:,0],mfpt_peak_idxs[:,2],
                        marker='*',markevery=10,markersize=10,alpha=.75,
                        color='tab:blue',label='Master')
    p4, = axs[0,2].plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                        marker='X',markevery=10,markersize=7,alpha=.75,
                        color='tab:blue')
    p5, = axs[0,2].plot(mfpt_peak_idxs[:,2],mfpt_peak_idxs[:,0],
                        marker='D',markevery=10,markersize=6,alpha=.75,
                        color='tab:blue')

    
    # inset to show middle branch for FP
    x1, x2, y1, y2 = 7.9, 8.01, 7.9, 8.01  # subregion of the original image
    axins02 = axs[0,2].inset_axes(
        [0.02, 0.02, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[],xticks=[],yticks=[])
    axins02.add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[0,2].indicate_inset_zoom(axins02, edgecolor="black")

    axins02.plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                 marker='X',markevery=10,markersize=7,alpha=.75,
                 color='tab:blue')

    
    # plot agents peak idxs (F)
    ze_list = np.round(np.arange(0.1,11,.2),1)
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.1)
    axs[1,2].plot(mfpt_peak_idxs[:,0],mfpt_peak_idxs[:,2],
                        marker='*',markevery=4,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,2].plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                        marker='X',markevery=4,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,2].plot(mfpt_peak_idxs[:,2],mfpt_peak_idxs[:,0],
                        marker='D',markevery=4,markersize=6,alpha=.75,
                        color='tab:orange')
    


    # inset to show middle branch for FP
    x1, x2, y1, y2 = 7.9, 8.02, 7.9, 8.02  # subregion of the original image
    axins12 = axs[1,2].inset_axes(
        [0.02, 0.02, 0.4, 0.4],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[],xticks=[],yticks=[])
    axins12.add_collection(collect(x_mid,y_mid,z_mid,**collect_kws))
    axs[1,2].indicate_inset_zoom(axins12, edgecolor="black")

    axins12.plot(mfpt_peak_idxs[:,1],mfpt_peak_idxs[:,1],
                 marker='X',markevery=4,markersize=7,alpha=.75,
                 color='tab:orange')

    
    #axs[0,0].legend([(p1,p2),(p3,p4,p5)],['F-P','Master'],
    #                handler_map={tuple:HandlerTuple(ndivide=None)})

    
    
    #axs[0,0].plot(const,y,color='gray',label='F-P',lw=2)
    #axs[0,0].plot(y,const,color='gray',label='F-P',lw=2)
    
    #axs[1,0].plot(p1_b502[idx:,0],p1_b502[idx:,1],color='gray',ls='--',lw=2)

    """
    #axs[1,0].plot(p1_b502[:idx,0],p1_b502[:idx,2],color='k',label='F-P',lw=2)
    #axs[1,0].plot(p1_b502[idx:,0],p1_b502[idx:,2],color='k',ls='--',lw=2)

    idx = np.argmax(p1_b505[:,0])
    axs[0,1].plot(p1_b505[:idx,0],p1_b505[:idx,1]*100,
                  color='gray',label='F-P',lw=2)
    axs[0,1].plot(p1_b505[idx:,0],p1_b505[idx:,1]*100,
                  color='gray',ls='--',lw=2)

    axs[1,1].plot(p1_b505[:idx,0],p1_b505[:idx,1]*100,
                  color='gray',label='F-P',lw=2)
    axs[1,1].plot(p1_b505[idx:,0],p1_b505[idx:,1]*100,
                  color='gray',ls='--',lw=2)

    idx = np.argmax(p1_b51[:,0])
    axs[0,2].plot(p1_b51[idx:,0],p1_b51[idx:,1],color='gray',label='F-P',lw=2)
    axs[0,2].plot(p1_b51[:idx,0],p1_b51[:idx,1],color='gray',ls='--',lw=2)

    axs[1,2].plot(p1_b51[idx:,0],p1_b51[idx:,1],color='gray',label='F-P',lw=2)
    axs[1,2].plot(p1_b51[:idx,0],p1_b51[:idx,1],color='gray',ls='--',lw=2)

    #ze_list_m = np.round(np.arange(0.1,5,.1),1)
    mfpt_peaks, mfpt_peak_idxs = get_master_peak_idxs(ze_list_m,B=5.05)
    axs[0,1].plot(ze_list_m,mfpt_peak_idxs[:,0],
                  marker='*',markevery=5,markersize=10,alpha=.75,
                  color='tab:blue',label='Master')
    axs[0,1].plot(ze_list_m,mfpt_peak_idxs[:,1],
                  marker='X',markevery=5,markersize=7,alpha=.75,
                  color='tab:blue')
    axs[0,1].plot(ze_list_m,mfpt_peak_idxs[:,2],
                  marker='D',markevery=5,markersize=6,alpha=.75,
                  color='tab:blue')

    #ze_list_m = np.round(np.arange(0.0,2,.1),1)
    mfpt_peaks, mfpt_peak_idxs = get_master_peak_idxs(ze_list_m,B=5.1)
    axs[0,2].plot(ze_list_m,mfpt_peak_idxs[:,0],
                  marker='*',markevery=2,markersize=10,alpha=.75,
                  color='tab:blue',label='Master')
    axs[0,2].plot(ze_list_m,mfpt_peak_idxs[:,1],
                  marker='X',markevery=2,markersize=7,alpha=.75,
                  color='tab:blue')
    axs[0,2].plot(ze_list_m,mfpt_peak_idxs[:,2],
                  marker='D',markevery=2,markersize=6,alpha=.75,
                  color='tab:blue')

    ### plot agents peak idxs
    # D
    ze_list = np.round(np.arange(0.1,11,.2),1)
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.02)
    axs[1,0].plot(ze_list,mfpt_peak_idxs[:,0],
                        marker='*',markevery=5,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,0].plot(ze_list,mfpt_peak_idxs[:,1],
                        marker='X',markevery=5,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,0].plot(ze_list,mfpt_peak_idxs[:,2],
                        marker='D',markevery=5,markersize=6,alpha=.75,
                        color='tab:orange')

    # E
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.05)
    

    axs[1,1].plot(ze_list,mfpt_peak_idxs[:,0],
                        marker='*',markevery=2,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,1].plot(ze_list,mfpt_peak_idxs[:,1],
                        marker='X',markevery=2,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,1].plot(ze_list,mfpt_peak_idxs[:,2],
                        marker='D',markevery=2,markersize=6,alpha=.75,
                        color='tab:orange')

    # F
    ze_list = np.round(np.arange(0.1,4,.2),1)
    mfpt_peaks, mfpt_peak_idxs = get_agent_peak_idxs(ze_list,B=5.1)
    #print('ze,peakidxs,')
    #for ii,jj in zip(ze_list,mfpt_peak_idxs[:,0]):
    #    print(ii,jj)
    axs[1,2].plot(ze_list,mfpt_peak_idxs[:,0],
                        marker='*',markevery=1,markersize=10,alpha=.75,
                        label='Agents',color='tab:orange')
    axs[1,2].plot(ze_list,mfpt_peak_idxs[:,1],
                        marker='X',markevery=1,markersize=7,alpha=.75,
                        color='tab:orange')
    axs[1,2].plot(ze_list,mfpt_peak_idxs[:,2],
                        marker='D',markevery=1,markersize=6,alpha=.75,
                        color='tab:orange')

    

    """
    xhi1 = np.amax(p1_b502[:,0])+np.amax(p1_b502[:,0])/10
    xhi2 = np.amax(p1_b505[:,0])+np.amax(p1_b505[:,0])/10
    xhi3 = np.amax(p1_b51[:,0])+np.amax(p1_b51[:,0])/10
    

    yhi = n_eq + n_eq/5.5


    for j in range(2):
        axs[j,0].set_xlim(0,yhi)
        axs[j,1].set_xlim(0,yhi)
        axs[j,2].set_xlim(0,yhi)

        axs[j,0].set_ylim(0,yhi)
        axs[j,1].set_ylim(0,yhi)
        axs[j,2].set_ylim(0,yhi)

        axs[j,0].legend()
    
        axs[j,0].set_ylabel('$D$')

        

    nr,nc = np.shape(axs)

    count = 0
    for i in range(nr):
        
        for j in range(nc):
            axs[i,j].set_title(labels[count],loc='left')
            axs[i,j].set_xlabel('$U$')
            axs[i,j].yaxis.set_major_locator(MaxNLocator(integer=True))

            count += 1


    titles = [[r'$B=\SI{5.02}{nm}$',r'$B=\SI{5.05}{nm}$',r'$B=\SI{5.1}{nm}$'],
              [r'$B=\SI{5.02}{nm}$',r'$B=\SI{5.05}{nm}$',r'$B=\SI{5.1}{nm}$']]
    for i in range(nr):
        for j in range(nc):
            t1 = axs[i,j].get_title()
            lab = t1 + titles[i][j]
            axs[i,j].set_title(lab)


    plt.tight_layout()

    return fig


def load_v_switch_pars(pars_and_ranges,recompute=False):

    data_dir1='switch_time_master/'
    if not(os.path.isdir(data_dir1)):
        os.mkdir(data_dir1)

    
    pname1 = pars_and_ranges['par1']
    pname2 = pars_and_ranges['par2']

    r1 = pars_and_ranges['par1_arange']
    r2 = pars_and_ranges['par2_arange']
    p1 = np.arange(*r1,dtype=float)
    p2 = np.arange(*r2,dtype=float)

    ft = data_dir1+'p1name={}_p2name={}_p1r={}_{}_{}_p2r={}_{}_{}.txt'
    ftX = data_dir1+'X_p1name={}_p2name={}_p1r={}_{}_{}_p2r={}_{}_{}.txt'
    ftY = data_dir1+'Y_p1name={}_p2name={}_p1r={}_{}_{}_p2r={}_{}_{}.txt'
    
    fname = ft.format(pname1,pname2,*r1,*r2)
    fnameX = ftX.format(pname1,pname2,*r1,*r2)
    fnameY = ftY.format(pname1,pname2,*r1,*r2)

    dne =  not(os.path.isfile(fname))
    dne += not(os.path.isfile(fnameX))
    dne += not(os.path.isfile(fnameY))

    if dne or recompute:

        X,Y = np.meshgrid(p1,p2,indexing='ij')

        M,N = X.shape

        out = np.empty_like(X)
        worker = telegraph.calculate_mfpt_only

        def cb(res):
            i, j, v = res
            print('i,j',i,j, 'cb out v',v)
            out[i, j] = v


        if False:
            for i, j in product(range(M), range(N)):
                _,_,mfpt =  worker(i, j, X[i,j], Y[i,j],pname1,pname2)
                out[i,j] = mfpt
        else:

            with Pool(processes=8) as p:
                for i, j in product(range(M), range(N)):
                    p.apply_async(worker, args=(i, j, X[i,j], Y[i,j],pname1,pname2), callback=cb)
                p.close(); p.join()


        """
        # do a second pass
        bools1 = (mfpt_mat==0)
        bools2 = (np.roll(mfpt_mat,1,axis=0)-mfpt_mat!=0)
        bools3 = (np.roll(mfpt_mat,1,axis=1)-mfpt_mat!=0)
        bools = bools1 & bools2 & bools3

        
        # get indices of potentially erroneous zeros
        i_idxs,j_idxs = np.where(bools)

        for j,k in zip(i_idxs,j_idxs):
            p1 = X[j,k] # parameter 1
            p2 = Y[j,k] # parameter 2

            j,k,mfpt = telegraph.calculate_mfpt_only(j,k,p1,p2,pname1,pname2,threshold=1e-3)
            mfpt_mat[j,k] = mfpt
        """
        
        np.savetxt(fname,out)
        np.savetxt(fnameX,X)
        np.savetxt(fnameY,Y)

    else:
        out = np.loadtxt(fname)
        X = np.loadtxt(fnameX)
        Y = np.loadtxt(fnameY)

    return X,Y,out




def clean_mfpt_mat(pars_and_ranges,mfpt_mat):

    pname1 = pars_and_ranges['par1']
    pname2 = pars_and_ranges['par2']

    r1 = pars_and_ranges['par1_arange']
    r2 = pars_and_ranges['par2_arange']
    p1 = np.arange(*r1,dtype=float)
    p2 = np.arange(*r2,dtype=float)

    X,Y = np.meshgrid(p1,p2,indexing='ij')
    
    # there are some zeros where there shouldn't be any.
    # this portion of the code finds them and re-calculates.
    bools1 = (mfpt_mat==0)
    bools2 = (np.roll(mfpt_mat,1,axis=0)-mfpt_mat!=0)
    bools3 = (np.roll(mfpt_mat,1,axis=1)-mfpt_mat!=0)
    bools = bools1 & bools2 & bools3

    # get indices of potentially erroneous zeros
    i_idxs,j_idxs = np.where(bools)

    for j,k in zip(i_idxs,j_idxs):
        p1 = X[j,k] # parameter 1
        p2 = Y[j,k] # parameter 2

        j,k,mfpt = telegraph.calculate_mfpt_only(j,k,p1,p2,pname1,pname2,threshold=1e-3)
        mfpt_mat[j,k] = mfpt

    return mfpt_mat
    

def load_cleaned_mfpt_mat(pars_and_ranges,mfpt_mat,recompute=False):
    data_dir1='switch_time_master/'
    if not(os.path.isdir(data_dir1)):
        os.mkdir(data_dir1)

    ft = data_dir1+'mfpt_cleaned_p1name={}_p2name={}_p1r={}_{}_{}_p2r={}_{}_{}.txt'

    pname1 = pars_and_ranges['par1']
    pname2 = pars_and_ranges['par2']

    r1 = pars_and_ranges['par1_arange']
    r2 = pars_and_ranges['par2_arange']
    p1 = np.arange(*r1,dtype=float)
    p2 = np.arange(*r2,dtype=float)

    fname = ft.format(pname1,pname2,*r1,*r2)
    dne =  not(os.path.isfile(fname))

    if dne or recompute:
        mfpt_mat = clean_mfpt_mat(pars_and_ranges,mfpt_mat)
        mfpt_mat = clean_mfpt_mat(pars_and_ranges,mfpt_mat)

        np.savetxt(fname,mfpt_mat)
            
    else:

        mfpt_mat = np.loadtxt(fname)

    return mfpt_mat

    


def v_switch_pars():
    #from scipy.interpolate import LinearNDInterpolator
    axis_labels = {'al':r'$\alpha$','be':r'$\beta$','ze':r'$\zeta$','B':r'$B$'}

    fig,axs = plt.subplots(2,2)
    axs = axs.reshape(-1)

    pars_and_ranges1 = {'par1':'al','par1_arange':(5,25,.15),
                        'par2':'be','par2_arange':(50,200,1)}
    
    pars_and_ranges2 = {'par1':r'ze','par1_arange':(0.1,10,.05),
                        'par2':r'al','par2_arange':(5,25,.15)}
    
    pars_and_ranges3 = {'par1':r'ze','par1_arange':(0.1,10,.05),
                        'par2':r'be','par2_arange':(50,200,1)}
    
    pars_and_ranges4 = {'par1':r'ze','par1_arange':(0.1,10,.05),
                        'par2':r'B','par2_arange':(5.02,5.2,.001)}

    pars_and_ranges_all = [pars_and_ranges1,pars_and_ranges2,
                           pars_and_ranges3,pars_and_ranges4]

    for i,pars_and_ranges in enumerate(pars_and_ranges_all):
        print('pars',pars_and_ranges['par1'],pars_and_ranges['par2'])
        pname1 = pars_and_ranges['par1']
        pname2 = pars_and_ranges['par2']

        X, Y, mfpt_mat = load_v_switch_pars(pars_and_ranges,recompute=False)
        
        #mfpt_mat = load_cleaned_mfpt_mat(pars_and_ranges,mfpt_mat,recompute=False)
        
        im = axs[i].contourf(X,Y,np.log(mfpt_mat))
        #im = axs[i].imshow(np.log(mfpt_mat))
        #im = axs[i].contourf(X,Y,mfpt_mat)

        im2 = axs[i].contour(X,Y,np.log(mfpt_mat),levels=[0],colors='tab:red')

        #axs[i].scatter(X[bools],Y[bools],s=1,color='red')
        #im = axs[i].imshow(mfpt_mat)

        axs[i].set_xlabel(axis_labels[pname1])
        axs[i].set_ylabel(axis_labels[pname2])
    
        cb = plt.colorbar(im,ax=axs[i])
        axs[i].set_title(labels[i],loc='left')
        cb.set_label('log[(Time to Switch Velocity)/$T_0$]')
        
        if i > 0:
            # set all x-axis limits for zeta x-axes to [.1,7]
            # instead of [0,10].

            axs[i].set_xlim(.1,7)


    # show inset for last plot
    axins = inset_axes(axs[-1], width="50%", height="50%", loc="upper right")
    X, Y, mfpt_mat = load_v_switch_pars(pars_and_ranges_all[-1],recompute=False)
    mfpt_mat = load_cleaned_mfpt_mat(pars_and_ranges_all[-1],mfpt_mat,recompute=False)
    
    axins.contourf(X,Y,np.log(mfpt_mat))
    axins.contour(X,Y,np.log(mfpt_mat),levels=[0],colors='tab:red')
    axins.set_xlim(.1,.3)
    axins.set_ylim(5.02,5.05)

    axins.set_xlabel(r'$\zeta$')
    axins.set_ylabel(r'$B$')

    mark_inset(axs[-1],axins,2,4,ec="0.5")
    
    

    plt.tight_layout()
    #plt.show()

    return fig


def get_mfpt_pars():
    """
    no-thought copy/paste to separate from the plotting functions in mfpt().
    """

    pardict = {'al':14,'ze':1,'be':126,'B':5.05}

    pars_and_ranges1 = {'par1':'al','par1_arange':(5,25,.15),
                        'par2':'be','par2_arange':(50,200,1)}
    
    pars_and_ranges2 = {'par1':r'ze','par1_arange':(0.1,10,.05),
                        'par2':r'al','par2_arange':(5,25,.15)}
    
    pars_and_ranges3 = {'par1':r'ze','par1_arange':(0.1,10,.05),
                        'par2':r'be','par2_arange':(50,200,1)}
    
    pars_and_ranges4 = {'par1':r'ze','par1_arange':(0.1,10,.05),
                        'par2':r'B','par2_arange':(5.02,5.2,.001)}


    data_dir1='switch_time_master/'
    if not(os.path.isdir(data_dir1)):
        os.mkdir(data_dir1)
    ft = data_dir1+'A={}_B={}_al={}_be={}_ze={}.txt'
    
    pars_and_ranges_all = [pars_and_ranges1,pars_and_ranges2,
                           pars_and_ranges3,pars_and_ranges4]

    Xs_list = []
    Ys_list = []
    lls_list = [] # switching rates for parameters
    vs_list = [] # steady-state velocities

    # for each range of parameters get switch time, velocity.
    for i,pars_and_ranges in enumerate(pars_and_ranges_all):
        pname1 = pars_and_ranges['par1']
        pname2 = pars_and_ranges['par2']

        X, Y, mfpt_mat = load_v_switch_pars(pars_and_ranges,recompute=False)

        mfpt_mat = load_cleaned_mfpt_mat(pars_and_ranges,mfpt_mat)

        
        pname1 = pars_and_ranges['par1']
        pname2 = pars_and_ranges['par2']
        
        r1 = pars_and_ranges['par1_arange']
        r2 = pars_and_ranges['par2_arange']
        p1 = np.arange(*r1,dtype=float)
        p2 = np.arange(*r2,dtype=float)

        X,Y,vels = load_vels(pars_and_ranges)


        
        lls = 1/mfpt_mat
        lls_list.append(lls)
        vs_list.append(vels)
        
        Xs_list.append(X)
        Ys_list.append(Y)

    fig,axs = plt.subplots(1,4)

    for k,ax in enumerate(axs):
        ax.imshow(vs_list[k])

    axs[0].set_title('vs')
    #plt.show()

    return Xs_list, Ys_list, lls_list, vs_list, pars_and_ranges_all


def get_vel(i,j,p1,p2,pname1,pname2):
    # get steady-state velocity for each parameter value
    pardict = {'al':14,'ze':1,'be':126,'B':5.05}
    pardict[pname1] = p1;pardict[pname2] = p2

    B = pardict['B'];al = pardict['al']
    be = pardict['be'];ze = pardict['ze']

    pars = Q.Parameters(nx=100,ny=100,A=5.,B=B,al=al,be=be,ze=ze)
    Qt = Q.get_transition_mat(pars)
    qss = Q.get_ss(pars,Qt)
    mat = Q.vec_to_mat(qss,pars)

    out = Q.get_local_idxs_v2(mat,pars) # 1 2 or 3 pairs
    x_idxs = out[:,0]
    y_idxs = out[:,1]

    if len(x_idxs) == 1:
        vel = 0
    else:
        vel = Q.v(x_idxs[0],y_idxs[0],pars)
    return i,j,vel


def load_vels(pars_and_ranges,recompute=False):
    """
    compute velocities given parameters
    """

    data_dir1='switch_time_master/'
    if not(os.path.isdir(data_dir1)):
        os.mkdir(data_dir1)

    pname1 = pars_and_ranges['par1']
    pname2 = pars_and_ranges['par2']

    r1 = pars_and_ranges['par1_arange']
    r2 = pars_and_ranges['par2_arange']
    p1 = np.arange(*r1,dtype=float)
    p2 = np.arange(*r2,dtype=float)

    ft = data_dir1+'vel_p1name={}_p2name={}_p1r={}_{}_{}_p2r={}_{}_{}.txt'
    ftX = data_dir1+'vel_X_p1name={}_p2name={}_p1r={}_{}_{}_p2r={}_{}_{}.txt'
    ftY = data_dir1+'vel_Y_p1name={}_p2name={}_p1r={}_{}_{}_p2r={}_{}_{}.txt'
    
    fname = ft.format(pname1,pname2,*r1,*r2)
    fnameX = ftX.format(pname1,pname2,*r1,*r2)
    fnameY = ftY.format(pname1,pname2,*r1,*r2)

    dne =  not(os.path.isfile(fname))
    dne += not(os.path.isfile(fnameX))
    dne += not(os.path.isfile(fnameY))

    if dne or recompute:
        X,Y = np.meshgrid(p1,p2,indexing='ij')
        M,N = X.shape

        vels = np.empty_like(X)
        worker = get_vel

        def cb(res):
            i, j, v = res
            print('cb out v',v)
            vels[i, j] = v


        with Pool() as p:
            for i, j in product(range(M), range(N)):
                p.apply_async(worker, args=(i, j, X[i,j], Y[i,j],pname1,pname2), callback=cb)
            p.close(); p.join()


        np.savetxt(fname,vels)
        np.savetxt(fnameX,X)
        np.savetxt(fnameY,Y)

    else:

        vels = np.loadtxt(fname)
        X = np.loadtxt(fnameX)
        Y = np.loadtxt(fnameY)

    return X,Y,vels


def mfpt(L):
    axis_labels = {'al':r'$\alpha$','be':r'$\beta$','ze':r'$\zeta$','B':r'$B$'}
    """L cylinder length in nanometers"""

     # see above for details
    Xs_list, Ys_list, lls_list, vs_list, pars_and_ranges_all = get_mfpt_pars()

    fig,axs = plt.subplots(2,4,figsize=(8,3))

    #for each ll, vel, get probability of escale and MFPT.
    im0 = [];im1 = []
    for i, (lam,v) in enumerate(zip(lls_list,vs_list)):
        

        pars_and_ranges = pars_and_ranges_all[i]

        pname1 = pars_and_ranges['par1']
        pname2 = pars_and_ranges['par2']

        X2, Y2, mfpt_v_mat = load_v_switch_pars(pars_and_ranges,recompute=False)
        
        mfpt_v_mat = load_cleaned_mfpt_mat(pars_and_ranges,mfpt_v_mat,recompute=False)

        # plot probability
        prob = telegraph.e0p(L,v,lam)

        # just remove extra low probs.
        prob[np.log(prob)<-7] = np.nan
        im0.append(axs[0,i].contourf(Xs_list[i],Ys_list[i],np.log(prob)))
        #axs[0,i].contour(Xs_list[i],Ys_list[i],np.log(prob),levels=[-7],
        #colors='red',zorder=2)

        im2 = axs[0,i].contour(Xs_list[i],Ys_list[i],np.log(mfpt_v_mat),levels=[0],colors='tab:red')
        
        # plot MPT
        mfpt = telegraph.t0p(L,v,lam)
        # just remove ultra high MFPTs.
        mfpt[np.log(mfpt)>10] = np.nan

        im1.append(axs[1,i].contourf(Xs_list[i],Ys_list[i],np.log(mfpt)))

        im2 = axs[1,i].contour(Xs_list[i],Ys_list[i],np.log(mfpt_v_mat),levels=[0],colors='tab:red')

        # axis labels
        axs[0,i].set_xlabel(axis_labels[pname1],labelpad=-4)
        axs[0,i].set_ylabel(axis_labels[pname2])

        axs[1,i].set_xlabel(axis_labels[pname1],labelpad=-4)
        axs[1,i].set_ylabel(axis_labels[pname2])

        axs[0,i].set_title(labels[i],loc='left')
        axs[1,i].set_title(labels[i+4],loc='left')

        if i > 0:
            
            # set all x-axis limits for zeta x-axes to [0,7.5]
            # instead of [0,10].

            axs[0,i].set_xlim(0,7)
            axs[1,i].set_xlim(0,7)

        if i == 3:
            

            # show inset for last column
            axins1 = inset_axes(axs[0,i], width="40%", height="40%", loc="upper right")
            axins2 = inset_axes(axs[1,i], width="40%", height="40%", loc="upper right")

            X, Y, mfpt_v_mat = load_v_switch_pars(pars_and_ranges_all[i],recompute=False)
            mfpt_v_mat = load_cleaned_mfpt_mat(pars_and_ranges_all[i],mfpt_v_mat,recompute=False)

            # prob.
            prob = telegraph.e0p(L,v,lam)
            prob[np.log(prob)<-7] = np.nan
            axins1.contourf(Xs_list[i],Ys_list[i],np.log(prob))
            axins1.contour(Xs_list[i],Ys_list[i],np.log(mfpt_v_mat),levels=[0],colors='tab:red')

            # mfpt
            mfpt = telegraph.t0p(L,v,lam)
            mfpt[np.log(mfpt)>10] = np.nan
            axins2.contourf(Xs_list[i],Ys_list[i],np.log(mfpt))
            axins2.contour(Xs_list[i],Ys_list[i],np.log(mfpt_v_mat),levels=[0],colors='tab:red')

            axins1.set_xlim(.1,.3)
            axins2.set_xlim(.1,.3)
            
            axins1.set_ylim(5.02,5.05)
            axins2.set_ylim(5.02,5.05)

            axins1.set_xlabel(r'$\zeta$',fontsize=8,labelpad=0)
            axins2.set_xlabel(r'$\zeta$',fontsize=8,labelpad=0)
            
            axins1.set_ylabel(r'$B$',fontsize=8,labelpad=0)
            axins2.set_ylabel(r'$B$',fontsize=8,labelpad=0)

            axins1.tick_params(axis='both',which='major',labelsize=5)
            axins1.tick_params(axis='both',which='minor',labelsize=5)

            axins2.tick_params(axis='both',which='major',labelsize=5)
            axins2.tick_params(axis='both',which='minor',labelsize=5)

            mark_inset(axs[0,-1],axins1,2,4,ec="0.5")
            mark_inset(axs[1,-1],axins2,2,4,ec="0.5")    
    

    plt.subplots_adjust(wspace=.55,hspace=.6,left=.07,right=1.05,
                        bottom=.15,top=.9)

    cb1 = plt.colorbar(im0[0],ax=axs[0,:])
    cb2 = plt.colorbar(im1[0],ax=axs[1,:])
    cb1.set_label('log(Probability)')
    cb2.set_label('log(MFPT/T_0)')

    
    
    return fig



class Arrow3D(FancyArrowPatch):
    """
    A class for drawing arrows in 3d plots.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        #xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def domain():

    
    """
    sideways cylinder for poster
    """
    
    T1 = .1
    
    #gs = gridspec.GridSpec(nrows=2,ncols=3,wspace=-.1,hspace=.5)
    fig = plt.figure(figsize=(5,3))
    ax11 = fig.add_subplot(111,projection='3d')
    #ax12 = fig.add_subplot(gs[0,2])
    #ax22 = fig.add_subplot(gs[1,2])
    
    
    a = lubrication(phi1=.57,Rp=0.96,Rc=1.22,base_radius=1.22,
                    pi3=1,pi4=4.7,pi5=0.1,pi6=10,
                    mu=1.2,T=T1,constriction='piecewise',U0=0.2,
                    dt=0.02,eps=1,
                    F0=50,method='euler')
    a.Z0 = -5/a.Rp
    
    z = np.linspace(-7,7,100)  # dimensional
    r = a.pi1(z)
    th = np.linspace(0,2*np.pi,100)
    
    radius_al = 0.25
    
    # draw two arrows out of spine    
    ar1 = Arrow3D([1.5,4],[0,0],[0,0],mutation_scale=10, 
                  lw=2, arrowstyle="-|>", color="k")
    ar2 = Arrow3D([-1.5,-4],[0,0],[0,0],mutation_scale=10, 
                  lw=2, arrowstyle="-|>", color="k")
    
    ax11.add_artist(ar1)
    ax11.add_artist(ar2)

    ax11.text(-3,0,.25,'$-V^*$',horizontalalignment='center',size=15)
    ax11.text(3,0,.25,'$V^*$',horizontalalignment='center',size=15)

    # draw spine
    Z,TH = np.meshgrid(z,th)
    X = np.zeros_like(Z)
    Y = np.zeros_like(Z)
    for i in range(len(Z[:,0])):
        X[i,:] = a.pi1(Z[i,:])*np.cos(TH[i,:])
        Y[i,:] = a.pi1(Z[i,:])*np.sin(TH[i,:])

    ls = LightSource(azdeg=0,altdeg=180)
    illuminated_surface = ls.shade(X,cmap=cm.Blues,blend_mode='soft')
    ax11.plot_surface(Z,Y,X,facecolors=illuminated_surface,
                      alpha=.3,edgecolor='none')
    

    # draw coordinate axes
    center = np.array([-6.5,0,2.])
    names = ['z','y','x']
    size = 2
    
    for i in range(3):
        coords = np.zeros((3,2))
        
        coords[:,0] += center
        coords[:,1] += center
        
        coords[i][1] += size
        arx = Arrow3D(*list(coords),
                      mutation_scale=5, 
                      lw=2, arrowstyle="-|>", color="k",)
    
        ax11.text(*list(coords[:,1]),names[i],horizontalalignment='center',
                  size=15)
        ax11.add_artist(arx)
        
    
    # draw sphere for cap
    b = a.base_radius
    r = np.sqrt(b**2+7**2)
    th2 = np.linspace(0,np.arctan(b/7),100)
    phi = np.linspace(0,2*np.pi,100)
    
    TH2,PHI = np.meshgrid(th2,phi)
    X = r*np.sin(TH2)*np.cos(PHI)
    Y = r*np.sin(TH2)*np.sin(PHI)
    Z = r*np.cos(TH2)
    ax11.plot_surface(Z,Y,X,color='tab:blue',alpha=.5)

    
    # draw sphere vesicle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = np.cos(u)*np.sin(v)
    Y = np.sin(u)*np.sin(v)
    Z = np.cos(v)
    
    illuminated_surface2 = ls.shade(X,cmap=cm.Greys,blend_mode='soft')
    
    ax11.plot_surface(Z,Y,X,facecolors=illuminated_surface2,
                      edgecolor='none')

    #ax11.plot_surface(Z,Y,X,color='gray',alpha=.5)
        
    # label spine head and base
    ax11.text(-7,0,-2,r'\setlength{\parindent}{0pt}Spine Base\\(Open End)',
              size=15)
    ax11.text(4.3,0,-2,r'\setlength{\parindent}{0pt}Spine Head\\(Closed End)',
              size=15)


    # annotate switching rate lambda with arrows
    ax11.text(0,0,3,r"Switch Rate $\lambda$",'x',ha='center',
              color='lightsalmon',
              bbox=dict(facecolor='none', edgecolor='lightsalmon',
                        boxstyle='round',pad=0.2),
              size=15)
    ar1b = Arrow3D([0,-3],[0,-1],[3,1.1],mutation_scale=10, 
                   lw=1, arrowstyle="-|>", color="lightsalmon",zorder=10)
    ar2b = Arrow3D([0,1.3],[0,-2.5],[3,1.8],mutation_scale=10, 
                  lw=1, arrowstyle="-|>", color="lightsalmon",zorder=10)
    
    ax11.add_artist(ar1b)
    ax11.add_artist(ar2b)
    
    
    # set equal aspect ratios
    #ax11.set_aspect('auto') # only auto allowed??
    ax11.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    
    ax11.set_axis_off()
    
    lo = -4.4
    hi = 4.4
    
    dx = -.5
    
    ax11.set_xlim(lo-dx,hi+dx)
    ax11.set_ylim(lo-dx,hi+dx)
    ax11.set_zlim(lo,hi)
    
    ax11.view_init(20,65+180)

    plt.subplots_adjust(bottom=-.3,left=0,right=1,top=1.3)

    
    #fig.tight_layout(pad=0.0)
    return fig

        

def generate_figure(function, args, filenames, bbox=False, dpi=MY_DPI):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;

    fig = function(*args)

    if type(filenames) == list:
        for name in filenames:
            fig.savefig(name,dpi=dpi,bbox_inches=bbox*'tight')
    else:
        fig.savefig(filenames,dpi=dpi,bbox_inches=bbox*'tight')


def main():

    # create figs directory if it doesn't exist
    if not(os.path.isdir('figs')):
        os.mkdir('figs')
        
    # listed in order of Figures in paper (usually...)
    figures = [
        (cylinder_motors,[],['figs/f_cylinder_motors.pdf']),

        (vel_fig,[],['figs/f_vel.pdf']),
        (dv_fig,[],['figs/dv.pdf']),
        (v_switch_mfpt,[],['figs/f_v_switch_mfpt.pdf']),
        
        (steady_state_examples,[],['figs/f_ss_examples.pdf']),
        (steady_states,[],['figs/f_ss.png','figs/f_ss.pdf']),
        (v_switch_pars,[],['figs/f_v_switch_pars.pdf']),
        (steady_states_complete,[],['figs/f_ss.png','figs/f_ss_complete.pdf']),
        
        (domain,[],['figs/f_domain.pdf']),
        (mfpt,[200],['figs/f_mfpt_200.pdf']),
        (mfpt,[1000],['figs/f_mfpt_1000.pdf'])
    ]
    
    for fig in figures:
        generate_figure(*fig)
    

if __name__ == "__main__":
    main()

