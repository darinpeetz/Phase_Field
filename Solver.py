# -*- coding: utf-8 -*-
"""
Created on Tue May 08 11:26:00 2018

@author: Darin
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from os import getcwd, path, makedirs, unlink, listdir

import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from Initialization import Read_Mesh
from Element import Element

import multiprocessing as mp
def Container_Func(args):
    """ Provides an interface to do local element routines in parallel """
    if args[0] == 'Local_Assembly':
        return args[1].Local_Assembly(*args[2:])
    elif args[0] == 'Update_Energy':
        raise ValueError("Update energy function cannot be run in parallel"
                         "using the multiprocessing module")
        return args[1].Update_Energy(*args[2:])
    elif args[0] == 'Energy':
        return args[1].Energy(*args[2:])
    elif args[0] == 'Stress':
        return args[1].Stress(*args[2:])
    elif args[0] == 'patch.set_xy':
        raise ValueError("Patch.Set_XY function cannot be run in parallel"
                         "using the multiprocessing module")
        return args[1].patch.set_xy(*args[2:])

# For plotting
class UpdatablePatchCollection(PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self.patches = patches
        PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths
    
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
        
        Nodes, Elements, Steps, Amplitudes, NSet, ElSet = Read_Mesh(filename)
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
        
        self.patches = []
        for el in self.Elements:
            self.patches.append(el.patch)
        self.patch_collection = UpdatablePatchCollection(self.patches, cmap=cmap.jet)
        fig = plt.figure("Display", figsize=(10,10))
        fig.clf()
        ax = plt.gca()
        ax.add_collection(self.patch_collection)
        self.patch_collection.set_array(np.zeros(len(self.Elements)))
        self.cbar = fig.colorbar(self.patch_collection, ax=ax)
    
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
            step['FixDof'] = np.array(step['FixDof'])
            step['Fixed_Inc'] = np.array(step['Fixed_Inc'])
            step['FreeDof'] = np.arange(Nodes.shape[0]*(self.dim+1))
            step['FreeDof'] = np.delete(step['FreeDof'], step['FixDof'])
        
            step['Fix_u'] = step['FixDof']/(self.dim+1)*self.dim + (step['FixDof'] % (self.dim+1))
            step['Free_u'] = np.delete(np.arange(self.udof.shape[0]), step['Fix_u'])
            step['Free_u_glob'] = self.udof[step['Free_u']]
        
        self.RHS = np.zeros(Nodes.shape[0]*(self.dim+1), dtype=float)
        self.uphi = self.RHS.copy()
        self.uphi_old = self.uphi.copy()
        self.stage_end = self.uphi.copy()
        
        self.step = 0
        self.iter_max = 500
        self.t = 0.
        self.t_max = 1.
        self.dt = 2e-3
        self.dt_min = 1e-8
        self.dt_max = 2e-3
        self.ftol = 5e-3
        self.ctol = 1e-2
        self.flux = {}
        
        self.solves = 0
        
        # Save the mesh information for ease of use later
        np.save(self.Folder + "\\Mesh.npy", {'Elements':self.Elements,
                                             'Nodes':self.Nodes,
                                             'Steps':self.Steps,
                                             'ElSet':self.ElSet,
                                             'NSet':self.NSet})
    
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
        data = np.load(filename).item()
        self.uphi = data['uphi']
        self.RHS = data['RHS']
        self.t = data['time']
        self.stage = data['stage']
        self.stage_end = data['stage_end']
        self.step = step
        
        for el in self.Elements:
            el.Update_Energy(self.uphi[el.dof])
        
    def Init_Parallel(self, processes=None):
        if processes is None:
            processes = mp.cpu_count()
            
        if processes <= 1:
            self.Parallel = False
        else:
            print "Setting up %i processes to do parallel local assembly"%processes
            self.pool = mp.Pool(processes)
            self.Parallel = True
        
    def Setup_Directory(self, Folder):
        """ Prepares output direcdtory for storing results
        
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
    
    def Global_Assembly(self, section, uphi=None, Assemble=3, couple=False):
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
        
        Returns
        -------
        K : sparse matrix
            Tangent stiffness matrix
        """
        if uphi is None:
            uphi = self.uphi
        if Assemble > 3 or Assemble < 0:
            raise ValueError("Value of 'Assemble' must be between 0 and 3, but is %"%Assemble)

        init = False
        if section == 'ALL':
            self.RHS.fill(0.)
            if not hasattr(self, 'Ki_ALL'):
                init = True
                self.Ki_ALL = []
                self.Kj_ALL = []
                self.Kk_ALL = []
            Ki = self.Ki_ALL
            Kj = self.Kj_ALL
            Kk = self.Kk_ALL
        elif section == 'UU':
            self.RHS[self.udof] = 0.
            if not hasattr(self, 'Ki_UU'):
                init = True
                self.Ki_UU = []
                self.Kj_UU = []
                self.Kk_UU = []
            Ki = self.Ki_UU
            Kj = self.Kj_UU
            Kk = self.Kk_UU
        elif section == 'PP':
            self.RHS[self.phidof] = 0.
            if not hasattr(self, 'Ki_PP'):
                init = True
                self.Ki_PP = []
                self.Kj_PP = []
                self.Kk_PP = []
            Ki = self.Ki_PP
            Kj = self.Kj_PP
            Kk = self.Kk_PP

        index = 0
        if init or not self.Parallel:
            for el in self.Elements:
                K, F = el.Local_Assembly(uphi[el.dof], section, Assemble, couple)
                if section == 'ALL':
                    self.RHS[el.dof] += F
                    if init:
                        rows = el.dof
                        cols = el.dof
                elif section == 'UU':
                    self.RHS[el.dof[el.udof]]   += F
                    if init:
                        rows = el.dof[el.udof]/(self.dim+1)*self.dim + (el.dof[el.udof] % (self.dim+1))
                        cols = el.dof[el.udof]/(self.dim+1)*self.dim + (el.dof[el.udof] % (self.dim+1))
                elif section == 'PP':
                    self.RHS[el.dof[el.phidof]] += F
                    if init:
                        rows = el.dof[el.phidof]/(self.dim+1)
                        cols = el.dof[el.phidof]/(self.dim+1)
                
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
            Klist, Flist = zip(*self.pool.map(Container_Func,
                [('Local_Assembly', el, uphi[el.dof], section, Assemble, couple) for el in self.Elements]))
            Kk = np.concatenate([Ke.reshape(-1) for Ke in Klist])
            for el in range(len(self.Elements)):
                if section == 'ALL':
                    self.RHS[self.Elements[el].dof] += Flist[el]
                elif section == 'UU':
                    self.RHS[self.Elements[el].dof[self.Elements[el].udof]] += Flist[el]
                elif section == 'PP':
                    self.RHS[self.Elements[el].dof[self.Elements[el].phidof]] += Flist[el]
        
        K = sparse.csr_matrix((Kk, (Ki, Kj)))
        
        return K

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
        for el in self.Elements:
            el.Update_Energy(self.uphi[el.dof])
 
        np.save(self.Folder + "\\Step_%i.npy"%self.step, {'uphi':self.uphi,
                                                          'RHS':self.RHS,
                                                          'time':self.t,
                                                          'stage':self.stage,
                                                          'stage_end':self.stage_end})
        self.step += 1
        self.uphi_old = self.uphi.copy()
        self.RHS_old = self.RHS.copy()
        self.t += self.dt
        amplitude = self.Amplitudes[self.stage['AMP']]
        self.uphi[self.stage['FixDof']] = (self.stage_end[self.stage['FixDof']] +
                  np.interp(self.t, amplitude[:,0], amplitude[:,1]) *
                  self.stage['Fixed_Inc'])
            
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
        
        self.t -= self.dt
        
        self.dt *= ratio
        if self.dt < self.dt_min:
            raise ValueError("Step size too small")    
            
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
        
        while self.dt > self.dt_min:
            accelerate = None
            
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                uphi_p[self.phidof] = self.uphi[self.phidof]
                Kpp = self.Global_Assembly('PP', uphi=uphi_p, Assemble=3, couple=False)
                dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                self.uphi[self.phidof] += dp
                
                if self.Convergence(dp, 'PP'):
                    if self.iter < 5:
                        accelerate = True
                    else:
                        accelerate = False
                    break

            if accelerate is None:
                self.Reduce_Step()
                continue
            
            for self.iter in range(self.iter_max):
                Kuu = self.Global_Assembly('UU', Assemble=3, couple=False)
                du = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                  -self.RHS[Free_u_glob])
                self.uphi[Free_u_glob] += du
                
                if self.Convergence(du, 'UU'):
                    if self.iter < 5 and accelerate:
                        self.dt = min(2*self.dt, self.dt_max)
                    return
                
            self.Reduce_Step()
            
        
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
        
        while self.dt > self.dt_min:
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                if not u_conv:
                    uphi_u[self.udof] = self.uphi[self.udof]
                    Kuu = self.Global_Assembly('UU', uphi=uphi_u, Assemble=3, couple=False)
                    du = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                      -self.RHS[Free_u_glob])
                    self.uphi[Free_u_glob] += du
                
                if not p_conv:
                    uphi_p[self.phidof] = self.uphi[self.phidof]
                    Kpp = self.Global_Assembly('PP', uphi=uphi_p, Assemble=3, couple=False)
                    dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                    self.uphi[self.phidof] += dp

                p_conv = self.Convergence(dp, 'PP')
                u_conv = self.Convergence(du, 'UU')
                if p_conv and u_conv:
                    if self.iter < 5:
                        pass
#                        self.dt = min(2*self.dt, self.dt_max)
                    return
             
#            if self.Plotting:
            raise ValueError("Step size too small")
            self.Reduce_Step()
            
        raise ValueError("Step size too small")
        
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
        
        while self.dt > self.dt_min:
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                self.solves += 1
                Kuu = self.Global_Assembly('UU', uphi=self.uphi, Assemble=3, couple=False)
                du = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                  -self.RHS[Free_u_glob])
                self.uphi[Free_u_glob] += du
                
                Kpp = self.Global_Assembly('PP', uphi=self.uphi, Assemble=3, couple=True)
                dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                self.uphi[self.phidof] += dp
                    
                p_conv = self.Convergence(dp, 'PP')
                u_conv = self.Convergence(du, 'UU')
                self.plot(data=['RHS','change'], save=True, suffix='_iter_%i'%self.iter)
                if p_conv and u_conv:
                    if self.iter < 5:
                        pass
#                        self.dt = min(2*self.dt, self.dt_max)
                    return
             
#            self.Reduce_Step()
            
        raise ValueError("Step size too small")
        
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
                self.uphi += du
                
                p_conv = self.Convergence(du[self.phidof], 'PP')
                u_conv = self.Convergence(du[self.stage['Free_u_glob']], 'UU')
                self.plot(data=['uphi','RHS','change'], save=True, suffix='_iter_%i'%self.iter)
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
            p_conv = False
            u_conv = False
            uphi_u = self.uphi.copy()
            uphi_p = self.uphi.copy()
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                if not u_conv:
                    uphi_u[self.udof] = self.uphi[self.udof]
                    Kuu = self.Global_Assembly('UU', uphi=uphi_u, Assemble=3, couple=False)
                    du[Free_u] = spla.spsolve(Kuu[Free_u[:,np.newaxis], Free_u],
                                      -self.RHS[Free_u_glob])
                    self.uphi[Free_u_glob] += du[Free_u]
                
                if not p_conv:
                    uphi_p[self.phidof] = self.uphi[self.phidof]
                    Kpp = self.Global_Assembly('PP', uphi=uphi_p, Assemble=3, couple=False)
                    dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                    self.uphi[self.phidof] += dp

                p_conv = self.Convergence(dp, 'PP', hold=(i>0))
                u_conv = self.Convergence(du[Free_u], 'UU', hold=(i>0))
                self.plot(data=['uphi', 'RHS', 'change'], save=True, suffix='_iter_%i-%i'%(i,self.iter))
#                update = np.hstack([du.reshape(-1,2), dp.reshape(-1,1)])
#                self.plot(data=[], save=True, suffix='_iter_%i-%i'%(i,self.iter), update=update)
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
                if (urel < 1e-6 or uabs < 1e-12) and (prel < 1e-6 or pabs < 1e-12):
                    print "Converged after %i corrections"%i
                    return
                else:
                    print "u_rel: %6.4g\tphi_rel: %6.4g\nphi_abs: %6.4g\tphi_abs: %6.4g"%(urel, prel, uabs, pabs)
                    for el in self.Elements:
                        el.Update_Energy(self.uphi[el.dof])
                last_change = new_change
            continue
             
            self.Reduce_Step()
            
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
            subset = self.stage['Free_u_glob']
        elif section == 'PP':
            subset = self.phidof
        elif section == 'ALL':
            subset = self.stage['FreeDof']
        else:
            print "Section specified = %s"%section
            raise ValueError("Unknown section specified in convergence")
        assert du.size == subset.size
        
        if not hold and self.iter == 0:
            self.flux[section] = np.sum(np.abs(self.RHS[subset]))
        else:
            self.flux[section] += np.sum(np.abs(self.RHS[subset]))
#            self.flux[section] = max(np.sum(np.abs(self.RHS[subset])), self.flux[section])
            
        if self.flux[section] == 0:
            force_check = True
        else:
            force_check = np.max(np.abs(self.RHS[subset])) < self.ftol * self.flux[section]/(self.iter+1)
        increment = self.uphi[subset] - self.uphi_old[subset]
        if np.max(np.abs(increment)) == 0:
            corr_check = True
        else:
            corr_check = np.max(np.abs(du)) < self.ctol * np.max(np.abs(increment)) or np.max(abs(du)) < 1e-12

        print "It: %i, Sect: %s, Force: %i, Corr: %i"%(self.iter, section, force_check, corr_check)
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
        elif Method == 'Full':
            Solve = self.Solve_Full
        elif Method == 'Hybrid':
            Solve = self.Solve_Hybrid
        else:
            raise ValueError("Unknown solver method specified")
            
        self.plot()
        Disp = []
        Reaction = []
        for self.stage in self.Steps:
            while self.t < self.t_max:
                self.Increment()
    
                
                Disp.append( np.max(self.uphi[(self.dim+1)*np.array(self.NSet['TOP'])+1]) )
                Reaction.append( np.linalg.norm(self.RHS[(self.dim+1)*np.array(self.NSet['TOP'])+1]) )
                print "Time: ", self.t
                if True or int(self.t / plot_frequency) > int((self.t - self.dt) / plot_frequency):
                    self.plot()

            self.stage_end = self.uphi.copy()
                
        return Disp, Reaction
        
    def plot(self, amp=1e0, data=['uphi'], save=True, suffix='', update=None):
        """Plot the current displacements and damage status
        
        Parameters
        ----------
        amp : scalar, optional
            Amplification factor
        data : list, optional
            Indicate which values to plot, select one or more of:
                change - change in current step
                uphi - displacements/damage
                RHS - residuals
                el_eng - elastic energy
                el_ten_eng - elastic tensile energy
                eng - energy
        update : array_like, optional
            Rows must match number of nodes.  If specified, will plot a heatmap
            of variables in each column (intended to visualize how data is being updated)
        
        Returns
        -------
        None
        """
        
        # Checking if any energies are to be plotted
        if update is not None:
            if update.shape[0] == self.Nodes.shape[0]:
                data = ['update']
            else:
                print "Update vector has wrong shape, ignoring"
                
        types = 0
        active = []
        energies = np.zeros((len(self.Elements),3))
        if 'el_eng' in data:
            types += 1
            active.append(0)
        if 'el_ten_eng' in data:
            types += 2
            active.append(1)
        if 'eng' in data:
            types += 4
            active.append(2)
            
        # Calculate any required energies
        if types > 0:
            if self.Parallel:
                energies = np.vstack(self.pool.map(Container_Func, 
                    *[('Energy', el, self.uphi[el.dof], types, np.mean)
                        for el in self.Elements]))
            else:
                energies = np.zeros((len(self.Elements),3))
                k = 0
                for el in self.Elements:
                    energies[k,active] = el.Energy(self.uphi[el.dof], types=types, reduction=np.mean)
                    k += 1
                
        if 'stress_x' in data or 'stress_y' in data or 'stress_xy' in data:
            if self.Parallel:
                stress = np.vstack(self.pool.map(Container_Func, 
                    [('Stress', el, self.uphi[el.dof], np.mean)
                        for el in self.Elements]))
            else:
                stress = np.zeros((len(self.Elements), 3))
                k = 0
                for el in self.Elements:
                    stress[k,:] = el.Stress(self.uphi[el.dof], reduction=np.mean)
                    k += 1
            
            
        for datum in data:
            if (datum == 'uphi' or ('eng' in datum) or ('stress' in datum)):
                vec = self.uphi
            elif datum == 'change':
                vec = self.uphi - self.uphi_old
            elif datum == 'RHS':
                vec = self.RHS
            elif datum == 'update':
                vec = 0*self.RHS
            else:
                print "Unkown variable to plot %s"%datum
                raise ValueError("Unkown plotting variable")

            if datum == 'update':
                colors = np.zeros((len(self.Elements),3))
            else:
                colors = np.zeros(len(self.Elements))
            shape = self.Nodes.copy()
            shape[:,0] += amp*vec[0::3]
            shape[:,1] += amp*vec[1::3]
            minim = shape.min(axis=0)
            maxim = shape.max(axis=0)
            i = 0
            
            if datum == 'el_eng':
                colors = energies[:,0]
            elif datum == 'el_ten_eng':
                colors = energies[:,1]
            elif datum == 'eng':
                colors = energies[:,2]
            elif datum == 'stress_x':
                colors = stress[:,0]
            elif datum == 'stress_y':
                colors = stress[:,1]
            elif datum == 'stress_xy':
                colors = stress[:,2]
                
#            if self.Parallel:
#                self.pool.map(Container_Func, 
#                    [('patch.set_xy', el, shape[el.nodes,:]) for el in self.Elements])
#                if datum == 'uphi' or datum == 'RHS':
#                    colors = np.array(self.pool.map(np.mean,
#                                    [vec[el.dof[el.phidof]] for el in self.Elements]))
#            else:
            for el in self.Elements:
                el.patch.set_xy(shape[el.nodes,:])
                if datum == 'uphi' or datum == 'RHS' or datum == 'change':
#                    if self.Parallel:
#                        colors = np.array(self.pool.map(np.mean,
#                            [vec[el.dof[el.phidof]] for el in self.Elements]))
#                    else:
                    colors[i] = np.mean(vec[el.dof[el.phidof]])
                    i += 1
                elif datum == 'update':
                    uphi = self.uphi.reshape(-1,3)
                    colors[i,:] = np.mean(update[el.nodes,:], axis=0)/np.mean(uphi[el.nodes,:], axis=0)
                    i += 1
                    
                    
            
            fig = plt.figure("Display", figsize=(10,10))
            if self.ipython:
                self.patch_collection = PatchCollection(self.patches, cmap=cmap.jet)
                fig.clf()
                ax = fig.gca()
                ax.add_collection(self.patch_collection)
                self.patch_collection.set_array(colors)
                self.cbar = fig.colorbar(self.patch_collection, ax=ax)
            else:
                self.patch_collection.set_array(colors)
                self.cbar.set_clim(colors.min(), colors.max())
                self.cbar.draw_all()
                
            if datum == 'update':
                break
            
            plt.axis('equal')
            plt.xlim(minim[0], maxim[0])
            plt.ylim(minim[1], maxim[1])
            plt.title('%s_%f'%(datum, self.t))
            if save:
                plt.savefig("%s\\%s_step_%i%s.png"%(self.Folder, datum, self.step, suffix))
            plt.draw()
            plt.pause(0.05)
        
        if update is not None:
            for i in range(update.shape[1]):
                if self.ipython:
                    self.patch_collection = PatchCollection(self.patches, cmap=cmap.jet)
                    fig.clf()
                    ax = fig.gca()
                    ax.add_collection(self.patch_collection)
                    self.patch_collection.set_array(colors[:,i])
                    self.cbar = fig.colorbar(self.patch_collection, ax=ax)
                else:
                    self.patch_collection.set_array(colors[:,i])
                    self.cbar.set_clim(colors[:,i].min(), colors[:,i].max())
                    self.cbar.draw_all()
                
                plt.axis('equal')
                plt.xlim(minim[0], maxim[0])
                plt.ylim(minim[1], maxim[1])
                plt.title('%s_%i_%f'%(datum, i, self.t))
                if save:
                    plt.savefig("%s\\%s_%i_step_%i%s.png"%(self.Folder, datum, i, self.step, suffix), dpi=200)
                plt.draw()
                plt.pause(0.05)
                
            
            
        
if __name__ == "__main__":
    
    solver = Solver('Asymmetric_Coarse.inp', getcwd() + '\\DELETEME')
#    solver = Solver('Double_Crack.inp', getcwd() + '\\Symmetric_Coupled_1e-4')
#    solver = Solver('Simple_Test.inp')
#    solver = Solver('Coarse_Double_Crack.inp', getcwd() + '\\Coarse_Hybrid')
    
    solver.Resume(getcwd() + '\\Asymmetric_Coarse2_Coupled_2e-3\\Step_395.npy', step=395)
    
    import time
    t0 = time.time()
    Disp, Reaction = solver.run(1e-10, 'Full')
    print time.time()-t0
    plt.figure(figsize=(12,12))
    plt.plot(Disp, np.array(Reaction), 'b-x')
    plt.savefig(solver.Folder + '\\Force_Displacement.png')
    print solver.solves
    
    
            