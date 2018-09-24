# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 07:58:22 2018

@author: Darin
"""

from Solver import Solver
from os import walk, getcwd, listdir
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.collections import PatchCollection


if __name__ == "__main__":
#    solver = Solver('Asymmetric_RightLoad.inp', getcwd() + '\\DELETEME')
    try:
        get_ipython().__class__.__name__
        ipython = True
    except:
        ipython = False
    
#    Disp = []
#    Reaction = []
#    step = 0
    Results = "C:\\Users\\Darin\\Documents\\Phase_Field\\Asymmetric_Coarse2_Decoupled_2e-3"
    Mesh = np.load(Results + "\\Mesh.npy").item()
    
    patches = []
    for el in Mesh['Elements']:
        patches.append(el.patch)
    patch_collection = PatchCollection(patches, cmap=cmap.jet)
    fig = plt.figure("Display", figsize=(10,10))
    fig.clf()
    ax = plt.gca()
    ax.add_collection(patch_collection)
    patch_collection.set_array(np.zeros(len(Mesh['Elements'])))
    cbar = fig.colorbar(patch_collection, ax=ax)
    colors = np.zeros(len(Mesh['Elements']))
        
    i = 0
    for filename in listdir(Results):
        if ".npy" in filename and "Step" in filename:
            step = int(filename.split("_")[1].split(".")[0])
            print step
            
            data = np.load(Results + "\\" + filename).item()
            i = 0
            for el in Mesh['Elements']:
                colors[i] = np.mean(data['uphi'][1::3][el.nodes])
                i += 1
                
            fig = plt.figure("Display", figsize=(10,10))
            if ipython:
                patch_collection = PatchCollection(patches, cmap=cmap.jet)
                fig.clf()
                ax = fig.gca()
                ax.add_collection(patch_collection)
                patch_collection.set_array(colors)
                cbar = fig.colorbar(patch_collection, ax=ax)
            else:
                patch_collection.set_array(colors)
                cbar.set_clim(colors.min(), colors.max())
                cbar.draw_all()

            plt.axis('equal')
            plt.xlim(Mesh['Nodes'][:,0].min(), Mesh['Nodes'][:,0].max())
            plt.ylim(Mesh['Nodes'][:,1].min(), Mesh['Nodes'][:,1].max())
            plt.title('Vert_Disp_t_%1.4g'%(data['time']))
            plt.savefig("DELETEME\\Vert_Disp_%i.png"%step)
#            solver.uphi = data['uphi']
#            solver.RHS = data['RHS']
#            solver.t = data['time']
#            solver.uphi = np.load(Results + "\\" + filename)
#            solver.Global_Assembly('UU', Assemble=2)
#            solver.t = solver.uphi[1::3].max()/0.2
#            solver.step = step
##                solver.plot(1., ['stress_x', 'stress_y', 'stress_xy'])
#            solver.plot()
##                plt.figure('Display')
##                plt.savefig("%s\\Energy_el_step_%i.png"%(Results, step))
#            Disp.append(np.max(solver.uphi[1::3]))
#            Reaction.append( np.linalg.norm(solver.RHS[(solver.dim+1)*np.array(solver.NSet['TOP'])+1]) )
#        
#    Disp = np.array(Disp)
#    Reaction = np.array(Reaction)
#    ind = np.argsort(Disp)
#    plt.figure(figsize=(12,12))
#    plt.plot(Disp[ind], Reaction[ind], 'b-x')
#    plt.savefig(Results + '\\Force_Displacement.png')
        
    