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
#    solver = Solver('Double_Cantilever.inp', getcwd() + '\\DELETEME')
    try:
        get_ipython().__class__.__name__
        ipython = True
    except:
        ipython = False
    
    Results = "C:\\Users\\Darin\\Documents\\Phase_Field\\Double_Cantilever_Global_Local_LongPad_FractureEng_Limit2"
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
        
    times = []
    energies = []
    Disp = []
    Reaction = []
    for filename in listdir(Results):
        if ".npy" in filename and "Step" in filename:
            step = int(filename.split("_")[1].split(".")[0])
#            if "%s_step_%i.png"%('uphi', step) in listdir(Results):
#                continue
            print step
            
            data = np.load(Results + "\\" + filename).item()
            uphi = data['uphi']
            RHS = data['RHS']
            
#            energies = np.zeros((uphi.shape[0], 3))
#            for i, el in enumerate(Mesh['Elements']):
#                el_energies = el.Energy(uphi[el.dof], types=7, reduction=np.sum)
#                energies[el.dof[el.phidof]] += el_energies/4
#            
#            shape = np.zeros_like(Mesh['Nodes'])
#            shape[:,0] = data['uphi'][0::3]
#            shape[:,1] = data['uphi'][1::3]
#            if not (np.abs(shape).max() == 0).any():
#                dims = np.max(Mesh['Nodes'], axis=0) - np.min(Mesh['Nodes'], axis=0)
#                scale = 0.02*np.min(dims/np.max(np.abs(shape), axis=0))
#            else:
#                scale = 1.
#            shape *= scale
#            shape += Mesh['Nodes']
#            minim = shape.min(axis=0)
#            maxim = shape.max(axis=0)
#
#            eng_types = ['el_eng', 'el_ten_eng', 'inel_eng']
#            for j in range(3):
#                for i, el in enumerate(Mesh['Elements']):
#                    el.patch.set_xy(shape[el.nodes,:])
#                    colors[i] = np.mean(energies[2::3,j][el.nodes])
#                    
#                fig = plt.figure("Display", figsize=(10,10))
#                if ipython:
#                    patch_collection = PatchCollection(patches, cmap=cmap.jet)
#                    fig.clf()
#                    ax = fig.gca()
#                    ax.add_collection(patch_collection)
#                    patch_collection.set_array(colors)
#                    cbar = fig.colorbar(patch_collection, ax=ax)
#                else:
#                    patch_collection.set_array(colors)
#                    cbar.set_clim(colors.min(), colors.max())
#                    cbar.draw_all()
#                
#                plt.axis('equal')
#                plt.xlim(minim[0], maxim[0])
#                plt.ylim(minim[1], maxim[1])
#                plt.title('%s_%f\nscale=%1.2g'%('uphi', data['time'], scale))
#                if True:#save:
##                    plt.savefig("Final.png")
#                    plt.savefig("%s\\%s_step_%i.png"%(Results, eng_types[j], step))
#                plt.draw()
#                plt.pause(0.05)

#            energy = 0
#            for el in Mesh['Elements']:
#                engs = el.Total_Energy(uphi[el.dof])
#                energy += np.inner(el.J*el.weights, engs)
#
            print data['time']
            times.append(data['time'])
#            energies.append(energy)
            keynodes = Mesh['NSet']['TOP']
            Disp.append( np.max(uphi[3*np.array(keynodes)+1]) )
            Reaction.append( np.linalg.norm(RHS[3*np.array(keynodes)+1]) )
    
    times = np.array(times)
    order = np.argsort(times)
    energies = np.array(energies)
    plt.figure("Evolution", figsize=(12,12))
    plt.clf()
    plt.plot(np.array(Disp)[order], np.array(Reaction)[order])
    plt.savefig('Force_Displacement.png')
#    np.save('Asymmetric_Global_Local.npy',{'times':times, 'energies':energies})
    