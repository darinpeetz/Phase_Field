# -*- coding: utf-8 -*-
"""
Created on Tue May 08 11:26:00 2018

@author: Darin
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from os import getcwd, path, makedirs, unlink, listdir, walk
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from Initialization import Read_Mesh, Read_Matlab
from Element import Element

import multiprocessing as mp

class BFGS_Operator:
    """ A class to encapsulate the functionality of a BFGS-updated Jacobian """
    
    def __init__(self, Kuu, Kpp, Free_u):
        """ Construct the operator with the initial Jacobian
        
        Parameters
        ----------
        Kuu, Kpp : scipy sparse matrices
            The initial Jacobian for displacement dofs and damage dofs, respectively
        Free_u : array_like
            The free degrees of freedom in the displacement block of dofs
        """
        
        self.Kuu = Kuu
        self.Kpp = Kpp
        self.FreeDofs = Free_u
        self.s = []
        self.y = []
        self.ys = []
        
    def Apply(self, b):
        """ Construct the operator with the initial Jacobian
        
        Parameters
        ----------
        b : array_like
            The right hand side of the linear system
            
        Returns
        -------
        dx : array_like
            Change in displacements and damage
        """
        
        term2 = []
        for i in reversed(range(len(self.s))):
            term2.append(self.s[i] * np.dot(self.s[i].T, b) / self.ys[i])
            b = b - self.y[i] * np.dot(self.s[i].T, b) / self.ys[i]
            
        dx = np.zeros_like(b)
        dx[self.FreeDofs] = spla.spsolve(self.Kuu[self.FreeDofs[:,np.newaxis],
                                         self.FreeDofs], b[self.FreeDofs])
        dx[self.Kuu.shape[0]:] = spla.spsolve(self.Kpp, b[self.Kuu.shape[0]:])
        
        for i in range(len(self.s)):
            dx = dx - self.s[i] * np.dot(self.y[i].T, dx) / self.ys[i] + term2.pop()
            
        return dx
        
    def Update(self, dx, dgrad):
        """ Construct the operator with the initial Jacobian
        
        Parameters
        ----------
        dx : array_like
            The change in displacements and damage
        dgrad : array_like
            The change in function right-hand-side 
        """
        
#        return
#        if self.s:
#            self.s[-1] = dx
#            self.y[-1] = dgrad
#            self.ys[-1] = np.dot(self.s[-1].T, self.y[-1])
#        else:
        self.s.append(dx)
        self.y.append(dgrad)
        self.ys.append(np.dot(self.s[-1].T, self.y[-1]))
    
def Container_Func(args):
    """ Provides an interface to do local element routines in parallel """
    if args[0] == 'Local_Assembly':
        if args[-1] is None or args[-2] in args[-1]:
            return args[1].Local_Assembly(*args[2:-2])
        else:
            if args[3].upper() == 'PP':
                c = 1
            elif args[3].upper() == 'UU':
                c = 2
            elif args[3].upper() == 'ALL':
                c = 3
            else:
                raise ValueError("Unknown section type")
            return (np.zeros((c*args[1].nodes.size, c*args[1].nodes.size)),
                    np.zeros(c*args[1].nodes.size) )
    else:
        string = "Can't run %s function in parallel"%args[0]
        raise ValueError(string)
        
def Dist_Func(args):
    """ Determines what points are within a radius of each other 
        args = point, points, radius
    """
    Dist = np.sqrt(np.sum((args[0] - args[1])**2, axis=1))
    
    return np.where(Dist < args[2])[0]
    

# For plotting
class UpdatablePatchCollection(PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths
    
    def get_facecolor(self):
        for i, patch in enumerate((self.patches)):
            self._facecolors[i,:] = patch.get_facecolor()
        return self._facecolors
    
class Solver:
    """Provides functionality to solve the coupled phase-field problem
    """
            
    def __init__(self, filename, Folder=None, processes=None):
        """Constructor from input file. Calls Read_Mesh then Setup()
        
        Parameters
        ----------
        filename : string
            Input filename
        Folder : string
            Directory where to store outputs
        """
        
        # Parallelism and plotting have to work differently in iPython consoles
        try:
            get_ipython().__class__.__name__
            self.ipython = True
            self.Parallel = False
        except:
            self.Init_Parallel(processes)
            self.ipython = False
        
        if filename[-4:] == '.inp':
            Nodes, Elements, Steps, Amplitudes, NSet, ElSet = Read_Mesh(filename)
        elif filename[-4:] == '.mat':
            Nodes, Elements, Steps, Amplitudes, NSet, ElSet = Read_Matlab(filename)
        else:
            raise ValueError("Unrecognized input file type")
            
        self.Setup(Nodes, Elements, Steps, Amplitudes, NSet, ElSet, Folder)
        
    def Setup(self, Nodes, Elements, Steps, Amplitudes, NSet, ElSet, Folder=None):
        """Constructor
        
        Parameters
        ----------
        Nodes : array_like
            Coordinates of every node in the mesh
        Elements : list of Element objects
            All elements in the mesh
        Steps : list of boundary information at each step
            Provides all boundary condition information
        Amplitude : list of amplitude information
            Provides amplitudes to be implemented in the steps
        NSet : dict
            All node sets
        ElSet : dict
            All element sets
        Folder : string
            Directory where to store outputs
        """
        self.Setup_Directory(Folder)
                   
        # Mesh
        self.Nodes = Nodes
        self.dim = Nodes.shape[1]
        self.udof = np.arange(Nodes.shape[0]*(self.dim+1))
        self.phidof = self.udof[self.dim::self.dim+1]
        self.udof = np.delete(self.udof,self.phidof)
        self.Elements = Elements
        self.Steps = Steps
        self.NSet = NSet
        self.Amplitudes = Amplitudes
        self.ElSet = ElSet
        self.Construct_Filter(Elements[0].lc)
        
        # Plotting
        self.patches = []
        self.sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0., vmax=1.))
        self.sm._A = []
        colors = np.zeros((len(self.Elements),4))
        for i, el in enumerate(self.Elements):
            self.patches.append(el.patch)
            colors[i,:] = self.sm.to_rgba(0)
        self.patch_collection = UpdatablePatchCollection(self.patches, match_original=True)
        fig = plt.figure("Display", figsize=(10,10))
        fig.clf()
        ax = plt.gca()
        ax.add_collection(self.patch_collection)
        self.cbar = plt.colorbar(self.sm, ax=ax)
    
        # Step info
        dim = Nodes.shape[1]
        for step in self.Steps:
            step['FixDof'] = []
            step['Fixed_Inc'] = []
            step['AMP'] = 'DEFAULT'
            for bc in step['INFO']:
                for dof in bc['DOF']:
                    step['FixDof'] += [(dim+1)*x+dof for x in NSet[bc['Set']]]
                    step['Fixed_Inc'] += [bc['VAL'] for x in NSet[bc['Set']]]
                    if step['AMP'] == 'DEFAULT':
                        step['AMP'] = bc['AMP']
            step['FixDof'] = np.array(step['FixDof']) # All dof that are fixed
            step['Fixed_Inc'] = np.array(step['Fixed_Inc']) # How to increment dof at timesteps
            step['FreeDof'] = np.arange(Nodes.shape[0]*(self.dim+1))
            step['FreeDof'] = np.delete(step['FreeDof'], step['FixDof']) # All dof that are free
        
            # Fixed displacement dofs out of only displacement dofs
            step['Fix_u'] = step['FixDof']//(self.dim+1)*self.dim + (step['FixDof'] % (self.dim+1))
            # Free displacement dofs out of only displacement dofs
            step['Free_u'] = np.delete(np.arange(self.udof.shape[0]), step['Fix_u'])
            # Fixed displacment dofs out of all dofs (should be teh same as 'FixDof')
            step['Free_u_glob'] = self.udof[step['Free_u']]
        
        self.RHS = np.zeros(Nodes.shape[0]*(self.dim+1), dtype=float)
        self.residual = self.RHS.copy()
        self.uphi = self.RHS.copy()
        self.uphi_old = self.uphi.copy()
        self.stage_end = self.uphi.copy()
        
        # Solver characteristics
        self.step = 0
        self.iter_max = 1000
        self.sub_iter_max = 20
        self.t = 0.
        self.t_max = 1.
        self.dt = 1e-2
        self.dt_min = 1e-8
        self.dt_max = 1e-2
        self.time_increase = []
        self.ftol = 5e-3
        self.ctol = 1e-2
        self.eps = 1e-5
        self.flux = {'avg_UU':0, 'avg_PP':0}
        self.delta_dmg_limit = 0.5
        self.dmg_limit = 0.7
        
        self.wait2increase = []
        self.solves = 0
        self.iterations = []
        
        # Save the mesh information for ease of use later
        np.save(self.Folder + "\\Mesh.npy", {'Elements':self.Elements,
                                             'Nodes':self.Nodes,
                                             'Steps':self.Steps,
                                             'ElSet':self.ElSet,
                                             'NSet':self.NSet,
                                             'Filter':self.Filter})
    
    def Resume(self, filename, step=0):
        """ Picks up where a previous solution left off
        
        Parameters
        ----------
        filename : string
            file containing last step information from other solution
        
        Returns
        -------
        None
        """
        data = np.load(filename, allow_pickle=True).item()
        self.uphi = data['uphi']
        self.RHS = data['RHS']
        self.t = data['time']
        self.stage = data['stage']
        self.stage_end = data['stage_end']
        self.step = step
        
        i = 0
        for el in self.Elements:
            el.Hist = data['History'][i]
            i += 1
        
    def Init_Parallel(self, processes=None):
        if processes is None:
            processes = mp.cpu_count()
            
        if processes <= 1:
            self.Parallel = False
        else:
            print("Setting up %i processes to do parallel local assembly" % processes)
            self.pool = mp.Pool(processes)
            self.Parallel = True
        
    def Setup_Directory(self, Folder):
        """ Prepares output directory for storing results
        
        Parameters
        ----------
        Folder : string
            Directory where to store outputs
        
        Returns
        -------
        None
        """
        
        if Folder is None:
            self.Folder = getcwd() + '\\Steps'
        else:
            self.Folder = Folder
        if not path.isdir(self.Folder):
            makedirs(self.Folder)
        else:
            for filename in listdir(self.Folder):
                file_path = path.join(self.Folder, filename)
                if path.isfile(file_path):
                    unlink(file_path)
    
    def Construct_Filter(self, lc):
        """ Sets up the damage filter operator
        
        Parameters
        ----------
        lc : scalar
            Length scale parameter (assumed constant throughout domain)
        
        Returns
        -------
        None
        """
        
        cells_x = int(np.ceil((self.Nodes[:,0].max() - self.Nodes[:,0].min())/lc))
        cells_y = int(np.ceil((self.Nodes[:,1].max() - self.Nodes[:,1].min())/lc))
        cells = np.zeros((cells_x, cells_y), dtype=object)
        
        cellNumbers = np.floor((self.Nodes - self.Nodes.min(axis=0))/lc)
        cellNumbers = np.minimum(cellNumbers, np.array(cells.shape)-1)
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                cells[i,j] = np.where(np.logical_and(cellNumbers[:,0] == i,
                                                     cellNumbers[:,1] == j))[0]
        
        adj_cells = np.zeros((2,3,3), dtype=int)
        adj_cells[0,0,:] = -1
        adj_cells[0,2,:] = 1
        adj_cells[1,:,0] = -1
        adj_cells[1,:,2] = 1
        adj_cells = adj_cells.reshape(2,-1).T
        
        indi = []
        indj = []
        valk = []
        for index, value in np.ndenumerate(cells):
            centers = cells[index]
            targets = []
            core = np.array(index)
            for offset in adj_cells:
                target = core + offset
                if (target < 0).any() or (cells.shape-target < 1).any():
                    continue
                else:
                    targets.append(cells[tuple(target)])
            targets = np.concatenate(targets)
                
            ct_crd = self.Nodes[centers,:].T.reshape(2, 1, -1)
            tg_crd = self.Nodes[targets,:].T.reshape(2, -1, 1)
            dist = np.sqrt(np.sum((tg_crd - ct_crd)**2, axis=0))
#            dist = np.maximum(1 - dist/lc, 0.)
            dist = (dist < lc).astype(float)
            
            indi += np.tile(targets.reshape(-1,1), (1,centers.size)).reshape(-1).tolist()
            indj += np.tile(centers.reshape(1,-1), (targets.size,1)).reshape(-1).tolist()
            valk += dist.reshape(-1).tolist()
        
        self.Filter = sparse.csr_matrix((valk,(indi,indj)))
        rowsum = self.Filter.sum(axis=1)
        self.Filter = sparse.dia_matrix((1/rowsum.T,0),shape=self.Filter.shape) * self.Filter
        
    def Global_Assembly(self, section, uphi=None, Assemble=3, couple=False, elems=None):
        """Assembles the global tangent stiffness matrix and internal force vector
        
        Parameters
        ----------
        section : scalar
            Part of the matrix to assemble ('uu', 'pp', 'up', or 'pu' or 'all')
            'uu' and 'pp' also return the internal force vectors.
        uphi : array_like
            Current field variables
        Assemble : integer, optional
            If Assemble & 1 == 1, assemble K
            If Assemble & 2 == 2, assemble RHS
            If both true, assemble both
        couple : boolean, optional
            Whether to include coupling terms
        elems : array_like, optional
            Which elements to assemble (for global-local approach)
        
        Returns
        -------
        K : sparse matrix
            Tangent stiffness matrix
        """
        if uphi is None:
            uphi = self.uphi
        if Assemble > 7 or Assemble < 0:
            raise ValueError("Value of 'Assemble' must be between 0 and 7, but is %"%Assemble)
        flag = np.zeros(len(self.Elements), dtype=int)
        if Assemble & 4 == 4:
            change = np.zeros(len(self.Elements))
            for i, el in enumerate(self.Elements):
                if section == 'UU':
                    change[i] = el.phi_old - np.sum((1-uphi[el.dof][el.phidof])**2)
                elif section == 'PP':
                    newstrain = np.dot(uphi[el.dof][el.udof].T, np.dot(el.Kuu, uphi[el.dof][el.udof].T))
                    change[i] = newstrain - el.strain_energy
            flag = 4 * (change > 1e-1*max(change))

        init = False
        if section == 'ALL':
            if Assemble & 2 == 2:
                self.RHS.fill(0.)
                self.flux['UU'] = 0.
                self.flux['PP'] = 0.
            if not hasattr(self, 'Ki_ALL'):
                init = True
                self.Ki_ALL = []
                self.Kj_ALL = []
                self.Kk_ALL = []
            Ki = self.Ki_ALL
            Kj = self.Kj_ALL
            Kk = self.Kk_ALL
        elif section == 'UU':
            if Assemble & 2 == 2:
                self.RHS[self.udof] = 0.
                self.flux['UU'] = 0.
            if not hasattr(self, 'Ki_UU'):
                init = True
                self.Ki_UU = []
                self.Kj_UU = []
                self.Kk_UU = []
            Ki = self.Ki_UU
            Kj = self.Kj_UU
            Kk = self.Kk_UU
        elif section == 'PP':
            if Assemble & 2 == 2:
                self.RHS[self.phidof] = 0.
                self.flux['PP'] = 0.
            if not hasattr(self, 'Ki_PP'):
                init = True
                self.Ki_PP = []
                self.Kj_PP = []
                self.Kk_PP = []
            Ki = self.Ki_PP
            Kj = self.Kj_PP
            Kk = self.Kk_PP

        index = 0
        # Any of these cases indicate we can't do assembly in parallel
        if init or not self.Parallel or elems is not None:
            for i, el in enumerate(self.Elements):
                # Loop over every element
                if elems is None or i in elems:
                    # Need to do assembly for this element
                    K, F = el.Local_Assembly(uphi[el.dof], section, Assemble+flag[i], couple)
#                    if i == 0:
#                        print(uphi[el.dof[el.phidof]])
#                        print(uphi[el.dof[el.udof]])
#                        np.save('K%s_%i-%i.npy' % (section, self.step, self.iter), K)
#                        np.save('F%s_%i-%i.npy' % (section, self.step, self.iter), F)
                else:
                    if section == 'ALL':
                        ndof = el.dof.size
                    elif section == 'UU':
                        ndof = el.udof.size
                    elif section == 'PP':
                        ndof = el.phidof.size
                    K = np.zeros((ndof, ndof))
                    F = np.zeros(ndof)
                if section == 'ALL':
                    if Assemble & 2 == 2:
                        self.RHS[el.dof] += F
                        self.flux['UU'] += np.abs(F[el.udof]).sum()
                        self.flux['PP'] += np.abs(F[el.phidof]).sum()
                    if init:
                        rows = el.dof
                        cols = el.dof
                elif section == 'UU':
                    if Assemble & 2 == 2:
                        self.RHS[el.dof[el.udof]]   += F
                        self.flux['UU'] += np.abs(F).sum()
                    if init:
                        rows = el.dof[el.udof]//(self.dim+1)*self.dim + (el.dof[el.udof] % (self.dim+1))
                        cols = el.dof[el.udof]//(self.dim+1)*self.dim + (el.dof[el.udof] % (self.dim+1))
                elif section == 'PP':
                    if Assemble & 2 == 2:
                        self.RHS[el.dof[el.phidof]] += F
                        self.flux['PP'] += np.abs(F).sum()
                    if init:
                        rows = el.dof[el.phidof] // (self.dim+1)
                        cols = el.dof[el.phidof] // (self.dim+1)
                
                if init:
                    Ki += np.tile(rows,(cols.size,1)).T.reshape(-1).tolist()
                    Kj += np.tile(cols,(rows.size,1)).reshape(-1).tolist()
                    Kk += K.reshape(-1).tolist()
                else:
                    Kk[index:index+K.size] = K.reshape(-1)
                    index += K.size

            Ki = np.array(Ki)
            Kj = np.array(Kj)
            Kk = np.array(Kk)
        else:
            # Parallel assembly
            Klist, Flist = zip(*self.pool.map(Container_Func,
                [('Local_Assembly', el, uphi[el.dof], section, Assemble+flag[i], couple, i, elems)
                    for i, el in enumerate(self.Elements)]))
            Kk = np.concatenate([Ke.reshape(-1) for Ke in Klist])
            for el in range(len(self.Elements)):
                if section == 'ALL' and Assemble & 2 == 2:
                    self.flux['UU'] += np.abs(Flist[el][self.Elements[el].udof]).sum()
                    self.flux['PP'] += np.abs(Flist[el][self.Elements[el].phidof]).sum()
                    self.RHS[self.Elements[el].dof] += Flist[el]
                elif section == 'UU' and Assemble & 2 == 2:
                    self.flux['UU'] += np.abs(Flist[el]).sum()
                    self.RHS[self.Elements[el].dof[self.Elements[el].udof]] += Flist[el]
                elif section == 'PP' and Assemble & 2 == 2:
                    self.flux['PP'] += np.abs(Flist[el]).sum()
                    self.RHS[self.Elements[el].dof[self.Elements[el].phidof]] += Flist[el]
        
        K = sparse.csr_matrix((Kk, (Ki, Kj)))
        if section == 'ALL':
            self.residual = self.RHS.copy()
        elif section == 'UU':
            self.residual[self.udof] = self.RHS[self.udof].copy()
        elif section == 'PP':
            self.residual[self.phidof] = self.RHS[self.phidof].copy()
        self.RHS[solver.stage['FixDof']] = 0
#        print "Assembly Time for section %s: %1.4g"%(section, time.time() - t0)
        
        print("%i elements fully assembled for section %s" % (sum([el.updated for el in self.Elements]), section))
        return K

    def Save_Status(self):
        """Saves current status of the solver
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        History = []
        self.Surface_Eng = 0
        for el in self.Elements:
            el.Update_Energy(self.uphi[el.dof])
            self.Surface_Eng += el.Surface_Eng
            History.append(el.Hist)
        
        np.save(self.Folder + "\\Step_%i.npy"%self.step, {'uphi':self.uphi,
                                                          'History':History,
                                                          'RHS':self.RHS,
                                                          'time':self.t,
                                                          'stage':self.stage,
                                                          'stage_end':self.stage_end})
        
    def Increment(self):
        """Increments the solver one step forward
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
#        if self.Parallel:
#            self.pool.map(Container_Func, 
#                [('Update_Energy', el, self.uphi[el.dof]) for el in self.Elements])
#        else:
        
        self.Save_Status()
        self.step += 1
        self.uphi_old = self.uphi.copy()
        self.RHS_old = self.RHS.copy()
        self.t += self.dt
        amplitude = self.Amplitudes[self.stage['AMP']]
        self.uphi[self.stage['FixDof']] = (self.stage_end[self.stage['FixDof']] +
                  np.interp(self.t, amplitude[:,0], amplitude[:,1]) *
                  self.stage['Fixed_Inc'])
        self.flux['crnt_UU'] = False
        self.flux['crnt_PP'] = False
            
    def Reduce_Step(self, ratio=0.5):
        """Reduces the step size in the case of non-convergence
        
        Parameters
        ----------
        ratio : scalar, optional
            Ratio to reduce step size by
        
        Returns
        -------
        None
        """
        
        self.time_increase.append(self.t)
        if len(self.time_increase) > 1 and self.time_increase[-1] > self.time_increase[-2]:
            print("Time: %1.12g" % self.t)
            raise ValueError("Bad increment control")
        print("Reducing step size from %1.2g to %1.2g" % (self.dt, self.dt*ratio))
        self.t -= self.dt
        
        # Force elements to fully reassemble on next call
        [el.Reset() for el in self.Elements]
            
        self.dt *= ratio
        self.wait2increase.append(int(np.round(1/ratio))-1)
        if self.dt < self.dt_min:
            self.dt = self.dt_min
            self.time_increase.pop()
#            raise ValueError("Step size too small")
            
        self.t += self.dt
        
        self.uphi = self.uphi_old.copy()
        self.RHS = self.RHS_old.copy()
        amplitude = self.Amplitudes[self.stage['AMP']]
        self.uphi[self.stage['FixDof']] = (self.stage_end[self.stage['FixDof']] +
                  np.interp(self.t, amplitude[:,0], amplitude[:,1]) *
                  self.stage['Fixed_Inc'])
        
    def Solve_Staggered(self):
        """Solve each subsystem in staggered fashion
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        uphi_p = self.uphi_old.copy()
        Free_u = self.stage['Free_u']
        Free_u_glob = self.stage['Free_u_glob']
        
        while self.dt >= self.dt_min:
#            accelerate = None
            
#            iter_max = self.iter_max
#            if self.dt == self.dt_min:
#                iter_max = 100000
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                uphi_p[self.phidof] = self.uphi[self.phidof]
                Kpp = self.Global_Assembly('PP', uphi=uphi_p, Assemble=3, couple=False)
                dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                self.uphi[self.phidof] += dp
                
                if self.Convergence(dp, 'PP'):
                    Converged = True
                    break
#                    if self.iter < 5:
#                        accelerate = True
#                    else:
#                        accelerate = False
#                    break

#            if accelerate is None:
#                self.Reduce_Step()
#                continue
            
            if not Converged:
                continue
            
            for self.iter in range(self.iter_max):
                Kuu = self.Global_Assembly('UU', Assemble=3, couple=False)
                du = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                  -self.RHS[Free_u_glob])
                self.uphi[Free_u_glob] += du
                
                if self.Convergence(du, 'UU'):
                    break
    #                    if self.iter < 5 and accelerate:
    #                        self.dt = min(2*self.dt, self.dt_max)
    #                    return
            break

            if self.dt > self.dt_min and np.logical_and(self.uphi[self.phidof] < self.dmg_limit,
                        (self.uphi[self.phidof] - self.uphi_old[self.phidof]) > self.delta_dmg_limit).any():
                self.Reduce_Step(ratio=0.1)
                break
            else:
                if self.time_increase and self.t >= self.time_increase[-1] - self.dt/10:
                    self.time_increase.pop()
                    self.dt *= 10
                return
            
        
    def Solve_Decoupled(self):
        """Solve each subsystem in a decoupled fashion
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        uphi_u = self.uphi_old.copy()
        uphi_p = self.uphi_old.copy()
        Free_u = self.stage['Free_u']
        Free_u_glob = self.stage['Free_u_glob']
        
        p_conv = False
        u_conv = False
        
        while self.dt >= self.dt_min:
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                uphi_u[self.udof] = self.uphi[self.udof]
                Kuu = self.Global_Assembly('UU', uphi=uphi_u, Assemble=3, couple=False)
                t0 = time.time()
                du = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                  -self.RHS[Free_u_glob])
                print("Time to solve section UU: %1.4g" % (time.time()-t0))
                self.uphi[Free_u_glob] += du
            
                uphi_p[self.phidof] = self.uphi[self.phidof]
                Kpp = self.Global_Assembly('PP', uphi=uphi_p, Assemble=3, couple=False)
                t0 = time.time()
                dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                print("Time to solve section PP: %1.4g" % (time.time()-t0))
                self.uphi[self.phidof] += dp
                
                if self.dt > self.dt_min and np.logical_and(self.uphi[self.phidof] < self.dmg_limit,
                            (self.uphi[self.phidof] - self.uphi_old[self.phidof]) > self.delta_dmg_limit).any():
                    self.Reduce_Step(ratio=0.1)
                    break

                p_conv = self.Convergence(dp, 'PP')
                u_conv = self.Convergence(du, 'UU')
                if p_conv and u_conv:
                    print(dmg_increase)
                    if dmg_increase < 0.5*self.dmg_limit:
                        self.dt = min(0.9*self.dmg_limit/dmg_increase*self.dt, self.dt_max)
                    return
             
        
    def Solve_Coupled(self):
        """Solve each subsystem in coupled fashion
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        Free_u = self.stage['Free_u']
        Free_u_glob = self.stage['Free_u_glob']
        dx = np.zeros_like(self.uphi)
        dx_old = np.zeros_like(self.uphi)
        
        rNormOld = 0
        while self.dt >= self.dt_min:
            flag = 4
            for self.iter in range(self.iter_max):
                if self.iter < 5 or self.iter % 5 == 0:
                    flag = 0
                
                self.solves += 1
                Kuu = self.Global_Assembly('UU', uphi=self.uphi, Assemble=3+flag, couple=False)
                du = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                  -self.RHS[Free_u_glob])
                self.uphi[Free_u_glob] += du
                
                Kpp = self.Global_Assembly('PP', uphi=self.uphi, Assemble=3+flag, couple=True)
                dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                self.uphi[self.phidof] += dp
                
#                if self.iter >= 5:
##                    alignU = np.dot(dx[Free_u_glob].T/np.linalg.norm(dx[Free_u_glob]), du)
#                    alignP = np.dot(dx[self.phidof].T/np.linalg.norm(dx[self.phidof]), dp/np.linalg.norm(dp))
#                    if alignP > 0.7:
##                        du += alignU * dx[Free_u_glob]
#                        dp += dx[self.phidof]
#                self.uphi[self.phidof] += dp
                
#                rNorm =  np.linalg.norm(self.RHS)
#                if self.iter >= 5:
#                    rInc = False
#                    if rNorm > rNormOld:
#                        rInc = True
##                    # Current residual
##                    self.Global_Assembly('UU', uphi=self.uphi, Assemble=2, couple=False)
##                    self.Global_Assembly('PP', uphi=self.uphi, Assemble=2, couple=True)
##                    r0 = np.linalg.norm(self.RHS[self.phidof])
##                    print('r0: %1.6g' % rNorm)
#                    
#                    # Update direction
##                    self.uphi[Free_u_glob] -= du
##                    self.uphi[self.phidof] -= dp
#                    dx = np.zeros_like(self.uphi)
#                    dx[Free_u_glob] = du
#                    dx[self.phidof] = dp
#                    dpMax = dp.max()
#                    critical = dp > 1e-2*dpMax
#                    
##                    # Initial guess for line search
##                    smax = (1 - self.uphi[self.phidof])/dp
##                    smax = np.min(smax[smax > 0])
##                    sold = 1
##                    s = 2
##                    rlist = [r0]
##                    set = False
##                    r0 = rNorm
#                    s = 1
#                    uphi = self.uphi.copy()
#                    rlist = [np.linalg.norm(self.RHS)]
#                    for i in range(10):
##                        self.uphi += dx
#                        self.uphi += s * dx
#                        self.Global_Assembly('UU', uphi=self.uphi, Assemble=2, couple=False)
#                        self.Global_Assembly('PP', uphi=self.uphi, Assemble=2, couple=True)
#                        r = np.linalg.norm(self.RHS)
#                        rlist.append(r)
#                        print('r%i-%i: %1.6g' % (self.iter+1, i+1, r))
#                        with open("Rhis_%i-%i" % (self.step, self.iter), 'a') as fh:
#                            fh.write('%1.16g\n' % r)
##                        if not set and r > r0:
##                            uphi = self.uphi - dx
##                            set = True
##                        if self.uphi[self.phidof].max() >= (self.uphi[self.phidof]+dp).max():
##                            rinc = False
##                        if (r > r0 and not rInc) or solver.uphi[solver.phidof][critical].max() > 1:
##                            self.uphi -= dx
##                            break
##                        elif r < r0:
##                            rInc = False
##                            r0 = r
##                        else:
##                            r0 = r
##                        if (i+1) % 10 == 0:
##                            s *= 3
#                    plt.figure("Residual History", figsize=(12,12), clear=True)
#                    plt.plot(rlist, 'b-x')
#                    plt.savefig(self.Folder + '\\r_step_%i_iter_%i'%(self.step, self.iter))
#                    self.uphi = uphi
#                    self.Global_Assembly('UU', uphi=self.uphi, Assemble=2, couple=False)
#                    self.Global_Assembly('PP', uphi=self.uphi, Assemble=2, couple=True)
                
                p_conv = self.Convergence(dp, 'PP')
                u_conv = self.Convergence(du, 'UU')
#                rNormOld = rNorm
#                
                dx = np.zeros_like(self.uphi)
                dx[Free_u_glob] = du
                dx[self.phidof] = dp
#                
##                change = dx/np.linalg.norm(dx) - dx_old
###                print("Change in update: %1.6g" % np.linalg.norm(change))
##                dx_old = dx/np.linalg.norm(dx)
##                alpha = np.array([el.E for el in self.Elements])
##                alpha /= alpha.max()
#                self.plot(data={'update':dx},#, 'RHS':self.RHS, 'uphi':self.uphi},
#                          save=True, suffix='_iter_%i'%(self.iter))
                
                if self.dt > self.dt_min and np.logical_and(self.uphi[self.phidof] < 1+self.dmg_limit,
                            (self.uphi[self.phidof] - self.uphi_old[self.phidof]) > self.delta_dmg_limit).any():
                    self.Reduce_Step(ratio=0.1)
                    break
                if p_conv and u_conv and flag == 0:
                    if self.time_increase and self.t >= self.time_increase[-1] - self.dt/10:
                        self.time_increase.pop()
                        self.dt *= 10
                        print("Increasing timestep to: %1.2g" % self.dt)
                    self.iterations.append(self.iter)
#                    raise ValueError("Stop Here")
                    return
                elif p_conv and u_conv:
                    flag = 0
                else:
                    flag = 4
            if self.iter >= self.iter_max-1:
                raise ValueError("No convergence")
                
#        raise ValueError("Step size too small")
        
        
    def Solve_BFGS(self):
        """Solve each subsystem in coupled using BFGS approximated Jacobian
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        Free_u = self.stage['Free_u']
        Free_u_glob = self.stage['Free_u_glob']
        
        while self.dt >= self.dt_min:
            Kuu = self.Global_Assembly('UU', uphi=self.uphi, Assemble=3, couple=False)
            du = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                              -self.RHS[Free_u_glob])
            self.uphi[Free_u_glob] += du
            for self.iter in range(self.iter_max):
                if self.iter % 8 == 0 or self.iter < 5:
                    Kuu = self.Global_Assembly('UU', uphi=self.uphi, Assemble=1, couple=False)
                    Kpp = self.Global_Assembly('PP', uphi=self.uphi, Assemble=1, couple=True)
                    BFGS = BFGS_Operator(Kuu, Kpp, Free_u)
#                print "Step: ", i
                
                b = np.concatenate([-self.RHS[self.udof], -self.RHS[self.phidof]])
                dx = BFGS.Apply(b)
                print("Update norm: %1.6g" % np.linalg.norm(dx))
                self.solves += 1
                self.uphi[Free_u_glob] += dx[Free_u]
                self.uphi[self.phidof] += dx[self.udof.size:]
                
                self.Global_Assembly('UU', uphi=self.uphi, Assemble=2, couple=False)
                self.Global_Assembly('PP', uphi=self.uphi, Assemble=2, couple=True)
#                BFGS.Kuu = Kuu
#                BFGS.Kpp = Kpp
                
#                BFGS = BFGS_Operator(Kuu, Kpp, Free_u)
                
                p_conv = self.Convergence(dx[self.udof.size:], 'PP')
                u_conv = self.Convergence(dx[Free_u], 'UU')
                
                dgrad = np.concatenate([-self.RHS[self.udof], -self.RHS[self.phidof]]) - b
                BFGS.Update(dx, dgrad)
                alpha = np.array([el.E for el in self.Elements])
                alpha /= alpha.max()
                update = 0*dx
                update[self.udof] = dx[:Kuu.shape[0]]
                update[self.phidof] = dx[Kuu.shape[0]:]
                delta = self.uphi - self.uphi_old
                self.plot(data={'update':update, 'delta':delta, 'RHS':self.RHS}, alpha=alpha,#, 'RHS':self.RHS, 'uphi':self.uphi},
                          save=True, suffix='_iter_%i'%(self.iter))
                
                print(p_conv, u_conv)
                if self.dt > self.dt_min and np.logical_and(self.uphi[self.phidof] < self.dmg_limit,
                            (self.uphi[self.phidof] - self.uphi_old[self.phidof]) > self.delta_dmg_limit).any():
                    self.Reduce_Step(ratio=0.1)
                    break
                if p_conv and u_conv:
                    if self.time_increase and self.t >= self.time_increase[-1] - self.dt/10:
                        self.time_increase.pop()
                        self.dt *= 10
                    return
                
            if self.iter >= self.iter_max-1:
                raise ValueError("No convergence")
                
    def Solve_Global_Local(self):
        """Solve each subsystem in coupled global-local fashion
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if not hasattr(self, 'Prox_el'):
            radius = 5.*self.Elements[0].lc
            if self.Parallel:
                loc_list = self.pool.map(Dist_Func, [(point, self.Nodes, radius) for point in self.Nodes])
            else:
                loc_list = [Dist_Func((point, self.Nodes, radius)) for point in self.Nodes]
            cols = np.concatenate([loc for loc in loc_list])
            rows = np.concatenate([[i]*loc.size for i, loc in enumerate(loc_list)])
            data = np.concatenate([[True]*loc.size for loc in loc_list])
            self.Prox_nd = sparse.csr_matrix((data, (rows,cols)))
            Tripi = []
            Tripj = []
            Tripk = []
            for i, el in enumerate(self.Elements):
                Tripi += el.nodes.tolist()
                Tripj += [i for j in el.nodes]
                Tripk += [True for j in el.nodes]
            self.Prox_el = (self.Prox_nd*sparse.csr_matrix((Tripk, (Tripi, Tripj)))).T
            
        
        Free_u = self.stage['Free_u']
        Free_u_glob = self.stage['Free_u_glob']
        dx = np.zeros_like(self.uphi)

        while self.dt > self.dt_min:
            for self.iter in range(self.iter_max):
    #            print "Step: ", i
                self.solves += 1
                Kuu = self.Global_Assembly('UU', uphi=self.uphi, Assemble=3, couple=False)
                t0 = time.time()
                dx[Free_u_glob] = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                  -self.RHS[Free_u_glob])
#                print "Time to solve section UU: %1.4g"%(time.time()-t0)
                self.uphi[Free_u_glob] += dx[Free_u_glob]
                
                Kpp = self.Global_Assembly('PP', uphi=self.uphi, Assemble=3, couple=True)
                t0 = time.time()
                dx[self.phidof] = spla.spsolve(Kpp, -self.RHS[self.phidof])
#                print "Time to solve section PP: %1.4g"%(time.time()-t0)
                self.uphi[self.phidof] += dx[self.phidof]
                    
                p_conv = self.Convergence(dx[self.phidof], 'PP')
                u_conv = self.Convergence(dx[Free_u_glob], 'UU')
                    
                dp = dx[self.phidof]
                p_x_dp = np.abs((self.uphi[self.phidof]>0.5)*dp)#*self.uphi[self.phidof])
                p_x_dp = p_x_dp > 0.9*np.max(p_x_dp)
                near_nodes = np.where(self.Prox_nd * p_x_dp)[0]
                active_elems = np.where(self.Prox_el * p_x_dp)[0]
                big_change_nodes = np.abs(dp) > 0.05*np.max(np.abs(dp))
                big_change = dp[big_change_nodes]
                concentration = np.sum(np.abs(dp[near_nodes]))/np.sum(np.abs(big_change))
                
    #            max_x = np.max(np.abs(dx[0::3]))
    #            max_y = np.max(np.abs(dx[1::3]))
    #            max_p = np.max(np.abs(dx[2::3]))
    #            dy = np.zeros_like(dx)
    ##            dy[0::3][big_change_nodes] = dx[0::3][big_change_nodes]/max_x
    ##            dy[1::3][big_change_nodes] = dx[1::3][big_change_nodes]/max_y
    #            dy[2::3][big_change_nodes] = dx[2::3][big_change_nodes]/max_p
    ##            dy[0::3][near_nodes] = 0
    ##            dy[1::3][near_nodes] = 0
    #            dy[2::3][near_nodes] = 0
    #            
    #            alpha = 0.3*np.ones(len(self.Elements))
    #            alpha[active_elems] = 1
    #            self.plot(data={'uphi':self.uphi, 'RHS':self.RHS, 'update':dx, 'sm_update':dy}, save=True,
    #                      suffix='_iter_%i'%(self.iter), alpha=alpha)
    
                # Energy to add lc to the crack length
                crack_add = 4./3*self.Elements[0].Gc*self.Elements[0].lc
                if p_conv and u_conv:
                    Surface_Eng = 0
                    for el in self.Elements:
                        Surface_Eng += el.Surface_Eng
                    if Surface_Eng - self.Surface_Eng < 10*crack_add:
                        if not self.wait2increase:
                            self.dt = min(10*self.dt, self.dt_max)
                        elif self.wait2increase[-1] == 0:
                            self.wait2increase.pop()
                            self.dt = min(10*self.dt, self.dt_max)
                            if self.wait2increase:
                                self.wait2increase[-1] -= 1
                        else:
                            self.wait2increase[-1] -= 1
                    return
    
                Surface_Eng = 0
                for el in self.Elements:
                    Surface_Eng += el.Surface_Eng
                print("Current Energy: %1.4g, Previous Energy: %1.4g, threshold: %1.4g" % (
                        Surface_Eng, self.Surface_Eng, 20*crack_add))
                if Surface_Eng - self.Surface_Eng > 20*crack_add:
                    if self.dt*0.1 < self.dt_min:
                        self.wait2increase.append(999)
                        self.wait2increase[-4] -= 1
                        self.uphi = self.uphi_old.copy()
                        self.RHS = self.RHS_old.copy()
                        amplitude = self.Amplitudes[self.stage['AMP']]
                        self.uphi[self.stage['FixDof']] = (self.stage_end[self.stage['FixDof']] +
                                  np.interp(self.t, amplitude[:,0], amplitude[:,1]) *
                                  self.stage['Fixed_Inc'])
                        raise ValueError("Step Size too small")
                        
                    self.Reduce_Step(ratio=0.1)
                    break
                
                if concentration > 0.8:
                    self.Local_Update(dp)

    def Local_Update(self, dp):
        """The local update routine for the global-local solver
        
        Parameters
        ----------
        dp : array_like
            Last update to the damage variable at each node
        
        Returns
        -------
        None
        """

        Free_u = self.stage['Free_u']
        Free_u_glob = self.stage['Free_u_glob']
        dx = np.zeros_like(self.uphi)
        dx[self.phidof] = dp
        for self.sub_iter in range(self.sub_iter_max):
            p_x_dp = np.abs(dx[self.phidof]*self.uphi[self.phidof])
            core_nodes = p_x_dp > 0.9*np.max(p_x_dp)
            active_nodes = np.where(self.Prox_nd * core_nodes)[0]
            active_elems = np.where(self.Prox_el * core_nodes)[0]
            Focus_Free = 3*np.outer(active_nodes,np.ones(3,int)) + np.outer(np.ones(active_nodes.size,int),np.arange(3))
            Focus_Free = Focus_Free.reshape(-1)
            Focus_Fix = np.setdiff1d(np.arange(3*self.Nodes.shape[0]), Focus_Free, assume_unique=True)
            Focus_Free_phi = Focus_Free[2::3]
            Focus_Free_phi = np.intersect1d(Focus_Free_phi, self.stage['FreeDof'], assume_unique=True)
            Focus_Free_u = np.delete(Focus_Free, np.arange(2,Focus_Free.size, 3))
            Focus_Free_u = np.intersect1d(Focus_Free_u, self.stage['FreeDof'], assume_unique=True)
            Local_Free_phi = (Focus_Free_phi//3)
            Local_Free_u = 2*(Focus_Free_u//3) + (Focus_Free_u % 3)
            
            dx.fill(0.)
            Kuu = self.Global_Assembly('UU', uphi=self.uphi, Assemble=3, couple=False, elems=active_elems)
            t0 = time.time()
            dx[Focus_Free_u] = spla.spsolve(Kuu[Local_Free_u[:,np.newaxis], Local_Free_u],
                              -self.RHS[Focus_Free_u])
#            print "Time to solve section UU-sub: %1.4g"%(time.time()-t0)
            self.uphi[Focus_Free_u] += dx[Focus_Free_u]
        
            Kpp = self.Global_Assembly('PP', uphi=self.uphi, Assemble=3, couple=True, elems=active_elems)
            t0 = time.time()
            dx[Focus_Free_phi] = spla.spsolve(Kpp[Local_Free_phi[:,np.newaxis], Local_Free_phi],
                              -self.RHS[Focus_Free_phi])
#            print "Time to solve section PP-sub: %1.4g"%(time.time()-t0)
            self.uphi[Focus_Free_phi] += dx[Focus_Free_phi]
            
            self.RHS[Focus_Fix] = 0.
            p_conv = self.Convergence(dx[self.phidof], 'PP')
            u_conv = self.Convergence(dx[Free_u_glob], 'UU')
            if p_conv and u_conv:
                return
            
#            alpha = 0.3*np.ones(len(self.Elements))
#            alpha[active_elems] = 1
#            self.plot(data={'uphi':self.uphi, 'RHS':self.RHS, 'update':dx}, save=True, alpha=alpha,
#                      suffix='_iter_%i-%i'%(self.iter,self.sub_iter))
            
        return
            
    def Solve_Full(self):
        """Solve each subsystem in coupled fashion
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        FreeDof = self.stage['FreeDof']
        
        du = 0*self.RHS
        while self.dt > self.dt_min:
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                K = self.Global_Assembly('ALL', uphi=self.uphi, Assemble=3, couple=True)
                du[FreeDof] = spla.spsolve(K[FreeDof[:,np.newaxis], FreeDof],
                                  -self.RHS[FreeDof])
#                if self.t > 0.1:
#                    scale = min(1./np.max(du[self.phidof]/(1.-self.uphi[self.phidof])),1)
#                    print scale, np.max(self.uphi[self.phidof])
#                    if scale < 0.:
#                        scale = 1.
#                    elif scale == 0.:
#                        np.save('Toy_Peak.npy', {'K':K, 'uphi':self.uphi, 'RHS':self.RHS})
#                        raise ValueError("No Change")
#                    du *= scale
                self.uphi += du
                        
                p_conv = self.Convergence(du[self.phidof], 'PP')
                u_conv = self.Convergence(du[self.stage['Free_u_glob']], 'UU')
#                self.plot(data=['uphi','RHS','change' save=True, suffix='_iter_%i'%self.iter)
                if p_conv and u_conv:
                    if self.iter < 5:
                        pass
#                        self.dt = min(2*self.dt, self.dt_max)
                    return
             
#            if self.Plotting:
            raise ValueError("Step size too small")
            self.Reduce_Step()
            
        raise ValueError("Step size too small")

    def Solve_Hybrid(self):
        """Solve each subsystem in hybrid (modified Newton-Raphson) fashion
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        Free_u = self.stage['Free_u']
        Free_u_glob = self.stage['Free_u_glob']
        du = np.zeros(self.Nodes.size)
        
        for i in range(10000):
            print("correction step %i" % i)
            p_conv = False
            u_conv = False
            uphi_u = self.uphi.copy()
            uphi_p = self.uphi.copy()
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                if True:#not u_conv:
                    uphi_u[self.udof] = self.uphi[self.udof]
                    Kuu = self.Global_Assembly('UU', uphi=uphi_u, Assemble=3, couple=False)
                    du[Free_u] = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                      -self.RHS[Free_u_glob])
                    self.uphi[Free_u_glob] += du[Free_u]
                
                if True:#not p_conv:
                    uphi_p[self.phidof] = self.uphi[self.phidof]
                    Kpp = self.Global_Assembly('PP', uphi=uphi_p, Assemble=3, couple=False)
                    dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                    self.uphi[self.phidof] += dp

                self.iter = 0
                p_conv = self.Convergence(dp, 'PP', hold=(i>0))
                u_conv = self.Convergence(du[Free_u], 'UU', hold=(i>0))
                if i > 20 and i % 10 == 0:
                    self.plot(data={'uphi':self.uphi, 'RHS':self.RHS}, save=True, suffix='_iter_%i-%i'%(i,self.iter))
                if p_conv and u_conv:
                    if self.iter < 5:
                        pass
    #                        self.dt = min(2*self.dt, self.dt_max)
                    break
                
            if i == 0:
                last_change = self.uphi - self.uphi_old
                for el in self.Elements:
                    el.Update_Energy(self.uphi[el.dof])
            else:
                new_change = self.uphi - self.uphi_old
                diff = new_change - last_change
                urel = np.linalg.norm(diff[self.udof]) / np.linalg.norm(last_change[self.udof])
                prel = np.linalg.norm(diff[self.phidof]) / np.linalg.norm(last_change[self.phidof])
                uabs = np.linalg.norm(new_change[self.udof])
                pabs = np.linalg.norm(new_change[self.phidof])
                if (urel < 1e-2 or uabs < 1e-12) and (prel < 1e-2 or pabs < 1e-12):
                    print("Converged after %i corrections" % i)
                    return
                else:
                    print("u_rel: %8.4e\tphi_rel: %8.4e" % (urel, prel))
                    print("phi_abs: %8.4e\tphi_abs: %8.4e" % (uabs, pabs))
                    for el in self.Elements:
                        el.Update_Energy(self.uphi[el.dof])
                last_change = new_change
            continue
             
#            self.Reduce_Step()
            
        raise ValueError("Step size too small")

    def Convergence(self, du, section, hold=False):
        """Check if nonlinear iterations have converged
        
        Parameters
        ----------
        du : array_like
            Change to field variables in last increment
        section : string
            Which subset of problem is being updated ('UU', 'PP', or 'ALL')
        hold : boolean
            Flag indicating that this is is a correction step and criteria are different
        
        Returns
        -------
        converged : bool
            True if iterations have converged, false otherwise
        """
        
        if section == 'UU':
            sections = ['UU']
            subsets = [self.stage['Free_u_glob']]
            dus = [du]
        elif section == 'PP':
            sections = ['PP']
            subsets = [self.phidof]
            dus = [du]
        elif section == 'ALL':
            sections = ['UU', 'PP']
            subsets = [self.stage['Free_u_glob'], self.phidof]
            dus = [du[self.stage['Free_u_glob']], du[self.phidof]]
        else:
            print("Section specified = %s" % section)
            raise ValueError("Unknown section specified in convergence")
        
        force_check = True
        corr_check = True
        for section, subset, du in zip(sections, subsets, dus):
            assert du.size == subset.size
            if not self.flux['crnt_%s' % section] and self.flux[section] > self.flux['avg_%s' % section]:
                self.flux['avg_%s' % section] = self.flux[section]
                self.flux['cnt_%s' % section] = 1
                self.flux['crnt_%s' % section] = True
            elif self.flux[section] > self.flux['avg_%s' % section]:
                self.flux['avg_%s' % section] += self.flux[section]
                self.flux['cnt_%s' % section] += 1
                
            res_max = np.max(np.abs(self.RHS[subset]))
            force_check = force_check and res_max < self.ftol * self.flux['avg_%s' % section]
    #        if not hold and self.iter == 0:
    #            active = np.abs(self.RHS[subset]) > self.eps * self.flux['max_%s' % section]
    #            self.flux['avg_%s' % section] = np.sum(np.abs(self.RHS[subset][active]))
    #        else:
    #            self.flux['max_%s' % section] += np.max(np.abs(self.RHS[subset]))
    #            active = np.abs(self.RHS[subset]) > self.eps * self.flux['max_%s' % section]
    #            self.flux['avg_%s' % section] += np.sum(np.abs(self.RHS[subset][active]))
    ##            self.flux['avg_%s' % section] = max(np.sum(np.abs(self.RHS[subset])), self.flux['avg_%s' % section])
    #        if self.flux['avg_%s' % section] == 0:
    #            force_check = True
    #        else:
    #            force_check = np.max(np.abs(self.RHS[subset])) < (self.ftol *
    #                                self.flux['avg_%s' % section]/(self.iter+1))
            increment = self.uphi[subset] - self.uphi_old[subset]
            if np.max(np.abs(increment)) == 0:
                pass
            else:
                corr_check = corr_check and np.max(np.abs(du)) < self.ctol * np.max(np.abs(increment)) or np.max(abs(du)) < 1e-12
                
            print(section, force_check, corr_check)
            
        return force_check and corr_check
        
    def run(self, plot_frequency=np.Inf, Method='Decoupled'):
        """Run the phase field solver
        
        Parameters
        ----------
        plot_frequncy : scalar
            How often to plot the displaced shape and damage status
        Method : string
            How to setup stiffness matrix/RHS, one of 'Staggered', 'Decoupled', or 'Coupled', 'Full'
        
        Returns
        -------
        None
        """
        if Method == 'Decoupled':
            Solve = self.Solve_Decoupled
        elif Method == 'Staggered':
            Solve = self.Solve_Staggered
        elif Method == 'Coupled':
            Solve = self.Solve_Coupled
        elif Method == 'BFGS':
            Solve = self.Solve_BFGS
        elif Method == 'Global_Local':
            Solve = self.Solve_Global_Local
        elif Method == 'Full':
            Solve = self.Solve_Full
        elif Method == 'Hybrid':
            Solve = self.Solve_Hybrid
        else:
            raise ValueError("Unknown solver method specified")
            
        self.plot()
        self.Disp = []
        self.Reaction = []
        for self.stage in self.Steps:
            while self.t < self.t_max:
                self.Increment()
                Solve()
                
                self.Disp.append( np.max(self.uphi[(self.dim+1)*np.array(self.NSet['TOP'])+1]) )
                self.Reaction.append( np.linalg.norm(self.residual[(self.dim+1)*np.array(self.NSet['TOP'])+1]) )
                print("Time: ", self.t, self.time_increase)
                if True or int(self.t / plot_frequency) > int((self.t - self.dt) / plot_frequency):
                    uphi_copy = self.uphi.copy()
                    uphi_copy[self.phidof] = self.Filter * uphi_copy[self.phidof]
                    alpha = np.array([el.E for el in self.Elements])
                    alpha /= alpha.max()
                    self.plot(data={'uphi':self.uphi.copy()}, alpha=alpha)

            self.stage_end = self.uphi.copy()
        
        self.Save_Status()
        return self.Disp, self.Reaction
        
    def plot(self, amp=None, data=None, alpha=None, save=True, suffix='', update=None):
        """Plot the current displacements and damage status
        
        Parameters
        ----------
        amp : scalar, optional
            Amplification factor
        data : dict, optional
            Name-value pair of information to plot
        alpha : array_like, optional
            Indicates a transparency to fade out certain elements (assumed as all ones if None)
        update : array_like, optional
            Rows must match number of nodes.  If specified, will plot a heatmap
            of variables in each column (intended to visualize how data is being updated)
        
        Returns
        -------
        None
        """
        
        if alpha is None:
            alpha = np.ones(len(self.Elements))
        if data is None:
            data = {'uphi':self.uphi.copy()}
            data['uphi'][self.phidof] = self.Filter * data['uphi'][self.phidof]
            
        t0 = time.time()
        for datum in data:
            # Shape
            shape = np.zeros(self.Nodes.shape)
            shape[:,0] = data[datum][0::3]
            shape[:,1] = data[datum][1::3]
            if amp is None:
                if not (np.abs(shape).max() == 0).any():
                    dims = np.max(self.Nodes, axis=0) - np.min(self.Nodes, axis=0)
                    scale = 0.05*np.min(dims/np.max(np.abs(shape), axis=0))
                else:
                    scale = 1.
            else:
                scale = amp
            shape *= scale
            shape += self.Nodes
            minim = shape.min(axis=0)
            maxim = shape.max(axis=0)
                
            # Coloring
            colors = np.ones((len(self.Elements), 4))
            self.sm.set_clim(vmin=np.min(data[datum][self.phidof]),
                             vmax=np.max(data[datum][self.phidof]))
            colors[:,-1] = alpha
            for i, el in enumerate(self.Elements):
                el.patch.set_xy(shape[el.nodes,:])
                colors[i,:-1] = self.sm.to_rgba(np.mean(data[datum][el.dof[el.phidof]]))[:-1]
                el.patch.set_facecolor(colors[i])
                    
            fig = plt.figure("Display", figsize=(10,10))
            if self.ipython:
                fig.clf()
                ax = fig.gca()
                
                patch_collection = PatchCollection(self.patches, match_original=True)
                ax.add_collection(patch_collection)
                
                self.sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=np.min(data[datum][self.phidof]),
                                                                               vmax=np.max(data[datum][self.phidof])))
                self.sm._A = []
                self.cbar = plt.colorbar(self.sm, ax=ax)
            else:
                self.cbar.set_clim(vmin=np.min(data[datum][self.phidof]),
                                   vmax=np.max(data[datum][self.phidof]))
                self.cbar.draw_all()
            
            plt.axis('equal')
            plt.xlim(minim[0], maxim[0])
            plt.ylim(minim[1], maxim[1])
            plt.title('%s_%1.16g\nscale=%1.2g'%(datum, self.t, scale))
            if save:
                plt.savefig("%s\\%s_step_%i%s.png"%(self.Folder, datum, self.step, suffix))
            plt.draw()
            plt.pause(0.05)
        print("Spent %1.4g seconds plotting %i visualizations" % (time.time()-t0, len(data)))
            
        
if __name__ == "__main__":
#    for i in range(3, 7):
#        solver = Solver('C88P01S11_4.mat', getcwd() + '\\Staggered_%i' % i)
#        solver.dt = 10**(-i)
#        solver.dt_max = solver.dt
#        Disp, Reaction = solver.run(1e-6, 'Staggered')
#    solver = Solver('C88P01S11_4.mat', getcwd() + '\\Coupled6')
    solver = Solver('Small_Crack.inp', getcwd() + '\\SmallCrack_alt4')
#    solver.Resume(getcwd() + '\\ScalingTest3\\Step_45.npy', step=39)
#    solver.dt = solver.dt_min
#    solver.t_max=0.1
#    import cProfile
#    cProfile.run("Disp, Reaction = solver.run(1e-6, 'Coupled')", "prof_stats_small_new")
    Disp, Reaction = solver.run(1e-6, 'Coupled')
    
#    solver.Resume(getcwd() + '\\Asym_Global_Local\\Step_39.npy', step=39)
    
#    t0 = time.time()
#    solver.t0 = time.time()
#    Disp, Reaction = solver.run(1e-10, 'Coupled')
#    print time.time()-t0
#    plt.figure(figsize=(12,12))
#    plt.plot(Disp, np.array(Reaction), 'b-x')
#    plt.savefig(solver.Folder + '\\Force_Displacement.png')
#    print solver.solves        
    
#    solver.stage = solver.Steps[0]
#    Disp = []
#    Reaction = []
#    for (dirpath, dirnames, filenames) in walk(getcwd() + '\\Scaling2'):
#        for filename in filenames:
#            if filename[:4] == "Step":
#                data = np.load(getcwd() + "\\Scaling2\\" + filename, allow_pickle=True).item()
#                solver.uphi = data['uphi']
#                [el.Reset() for el in solver.Elements]
#                solver.Global_Assembly('UU', uphi=solver.uphi, Assemble=2, couple=False)
#                Disp.append(np.max(solver.uphi[(solver.dim+1)*np.array(solver.NSet['TOP'])+1]))
#                Reaction.append( np.linalg.norm(solver.residual[(solver.dim+1)*np.array(solver.NSet['TOP'])+1]) )
#            
#    Disp = np.array(Disp)
#    Reaction = np.array(Reaction)
#    order = np.argsort(Disp)
#    Disp = Disp[order]
#    Reaction = Reaction[order]
#    plt.figure(figsize=(12,12), clear=True)
#    plt.plot(Disp, np.array(Reaction), 'b-x')