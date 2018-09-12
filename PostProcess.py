# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 07:58:22 2018

@author: Darin
"""

from Solver2 import Solver
from os import walk, getcwd, listdir
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
#    solver = Solver('Asymmetric_RightLoad.inp', getcwd() + '\\DELETEME')
##    try:
##        get_ipython().__class__.__name__
##    except:
##        solver.Init_Parallel()
#    
#    Disp = []
#    Reaction = []
#    step = 0
    Results = "C:\\Users\\Darin\\Box Sync\\Research\\Phase_Field_Tests\\Python_code\\Asym_Coupled_SmallerStep"
    i = 0
    for filename in listdir(Results):
        if ".npy" in filename:
            step = int(filename.split("_")[1].split(".")[0])
            print step
            data = np.load(Results + "\\" + filename).item()
            print step, data['time']
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
        
    