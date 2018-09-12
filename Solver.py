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
            
    def __init__(self, filename, Folder=None):
        """Constructor from input file. Calls Read_Mesh then Setup()
        
        Parameters
        ----------
        filename : string
            Input filename
        Folder : string
            Directory where to store outputs
        """
        
        Nodes, Elements, Bound, NSet, ElSet = Read_Mesh(filename)
        self.Setup(Nodes, Elements, Bound, NSet, ElSet, Folder)
        
    def Setup(self, Nodes, Elements, Bound, NSet, ElSet, Folder=None):
        """Constructor
        
        Parameters
        ----------
        Nodes : array_like
            Coordinates of every node in the mesh
        Elements : list of Element objects
            All elements in the mesh
        Bound : list of boundary lists
            Provides all boundary condition information
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
        self.Bound = Bound
        self.NSet = NSet
        self.ElSet = ElSet
        
        patches = []
        for el in self.Elements:
            patches.append(el.patch)
        self.patch_collection = UpdatablePatchCollection(patches, cmap=cmap.jet)
        fig = plt.figure("Display", figsize=(10,10))
        fig.clf()
        ax = plt.gca()
        ax.add_collection(self.patch_collection)
        self.patch_collection.set_array(np.zeros(len(self.Elements)))
        self.cbar = fig.colorbar(self.patch_collection, ax=ax)
    
        self.FixDof = []
        self.Fixed_Inc = []
        dim = Nodes.shape[1]
        for bc in Bound:
            for dof in bc[1:-1]:
                self.FixDof += [(dim+1)*x+dof for x in NSet[bc[0]]]
                self.Fixed_Inc += [bc[-1] for x in NSet[bc[0]]]
        self.FixDof = np.array(self.FixDof)
        self.Fixed_Inc = np.array(self.Fixed_Inc)
        self.FreeDof = np.arange(Nodes.shape[0]*(self.dim+1))
        self.FreeDof = np.delete(self.FreeDof, self.FixDof)
        
        self.Fix_u = self.FixDof/(self.dim+1)*self.dim + (self.FixDof % (self.dim+1))
        self.Free_u = np.delete(np.arange(self.udof.shape[0]), self.Fix_u)
        self.Free_u_glob = self.udof[self.Free_u]
        
        self.RHS = np.zeros(Nodes.shape[0]*(self.dim+1), dtype=float)
        self.uphi = self.RHS.copy()
        self.uphi_old = self.uphi.copy()
        
        self.step = 0
        self.iter_max = 1000
        self.t = 0.
        self.t_max = 1.
        self.dt = 2e-3
        self.dt_min = 1e-8
        self.dt_max = 2e-3
        self.ftol = 5e-3
        self.ctol = 1e-2
        self.flux = {}
        
        self.Parallel = False
        self.solves = 0
    
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
        self.step = step
        
    def Init_Parallel(self, processes=None):
        if processes is None:
            processes = mp.cpu_count()
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
                                                          'time':self.t})
        self.step += 1
        self.uphi_old = self.uphi.copy()
        self.RHS_old = self.RHS.copy()
        self.t += self.dt
        self.uphi[self.FixDof] = self.t * self.Fixed_Inc
            
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
        self.uphi[self.FixDof] = self.t * self.Fixed_Inc
        
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
                du = spla.spsolve(Kuu[self.Free_u[:,np.newaxis], self.Free_u],
                                  -self.RHS[self.Free_u_glob])
                self.uphi[self.Free_u_glob] += du
                
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
        
        p_conv = False
        u_conv = False
        
        while self.dt > self.dt_min:
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                if not u_conv:
                    uphi_u[self.udof] = self.uphi[self.udof]
                    Kuu = self.Global_Assembly('UU', uphi=uphi_u, Assemble=3, couple=False)
                    du = spla.spsolve(Kuu[self.Free_u[:,np.newaxis], self.Free_u],
                                      -self.RHS[self.Free_u_glob])
                    self.uphi[self.Free_u_glob] += du
                
                if not p_conv:
                    uphi_p[self.phidof] = self.uphi[self.phidof]
                    Kpp = self.Global_Assembly('PP', uphi=uphi_p, Assemble=3, couple=False)
                    dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                    self.uphi[self.phidof] += dp

                p_conv = self.Convergence(dp, 'PP')
                u_conv = self.Convergence(du, 'UU')
                if p_conv and u_conv:
#                if np.linalg.norm(self.RHS[self.udof]) < self.tol and np.linalg.norm(self.RHS[self.phidof]) < self.tol:
#                    print ("Phi dof converged within %6.5f and u dof converged "
#                          "within %6.5f"%(phirat, urat))
                    if self.iter < 5:
                        self.dt = min(2*self.dt, self.dt_max)
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
        
        while self.dt > self.dt_min:
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                self.solves += 1
                Kuu = self.Global_Assembly('UU', uphi=self.uphi, Assemble=3, couple=False)
                du = spla.spsolve(Kuu[self.Free_u[:,np.newaxis], self.Free_u],
                                  -self.RHS[self.Free_u_glob])
                self.uphi[self.Free_u_glob] += du
                
                Kpp = self.Global_Assembly('PP', uphi=self.uphi, Assemble=3, couple=True)
                dp = spla.spsolve(Kpp, -self.RHS[self.phidof])
                self.uphi[self.phidof] += dp

#                print "Max change in damage %1.6g"%np.max(solver.uphi[self.phidof] - solver.uphi_old[self.phidof])
#                self.plot()
                if np.max(solver.uphi[self.phidof] - solver.uphi_old[self.phidof]) > 0.5 and solver.dt > 2e-4:
                    self.Reduce_Step(ratio=0.1)
                    break
                    
                if (self.iter % 1) == 0:
                    self.plot(data=['change'], suffix='_iter_%i'%self.iter)
                p_conv = self.Convergence(dp, 'PP')
                u_conv = self.Convergence(du, 'UU')
                if p_conv and u_conv:
#                if np.linalg.norm(self.RHS[self.udof]) < self.tol and np.linalg.norm(self.RHS[self.phidof]) < self.tol:
#                    print ("Phi dof converged within %6.5f and u dof converged "
#                          "within %6.5f"%(phirat, urat))
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
        
        du = 0*self.RHS
        while self.dt > self.dt_min:
            for self.iter in range(self.iter_max):
#                print "Step: ", i
                K = self.Global_Assembly('ALL', uphi=self.uphi, Assemble=3, couple=True)
                du[self.FreeDof] = spla.spsolve(K[self.FreeDof[:,np.newaxis], self.FreeDof],
                                  -self.RHS[self.FreeDof])

                du[self.phidof] = np.minimum(np.maximum(du[self.phidof], -self.uphi[self.phidof]), 1-self.uphi[self.phidof])
                self.uphi += du
                                
                if (self.iter % 5) == 0:
                    self.plot(data=['change'], suffix='_iter_%i'%self.iter)
                p_conv = self.Convergence(du[self.phidof], 'PP')
                u_conv = self.Convergence(du[self.Free_u_glob], 'UU')
                if p_conv and u_conv:
#                if np.linalg.norm(self.RHS[self.udof]) < self.tol and np.linalg.norm(self.RHS[self.phidof]) < self.tol:
#                    print ("Phi dof converged within %6.5f and u dof converged "
#                          "within %6.5f"%(phirat, urat))
                    if self.iter < 5:
                        self.dt = min(2*self.dt, self.dt_max)
                    return
             
#            if self.Plotting:
            raise ValueError("Step size too small")
            self.Reduce_Step()
            
        raise ValueError("Step size too small")
        
    def Convergence(self, du, section):
        """Check if nonlinear iterations have converged
        
        Parameters
        ----------
        du : array_like
            Change to field variables in last increment
        section : string
            Which subset of problem is being updated ('UU', 'PP', or 'ALL')
        
        Returns
        -------
        converged : bool
            True if iterations have converged, false otherwise
        """
        
        if section == 'UU':
            subset = self.Free_u_glob
        elif section == 'PP':
            subset = self.phidof
        elif section == 'ALL':
            subset = self.FreeDof
        else:
            print "Section specified = %s"%section
            raise ValueError("Unknown section specified in convergence")
        assert du.size == subset.size
        
        if self.iter == 0:
            self.flux[section] = np.sum(np.abs(self.RHS[subset]))
        else:
            self.flux[section] += np.sum(np.abs(self.RHS[subset]))
#            self.flux[section] = max(np.sum(np.abs(self.RHS[subset])), self.flux[section])
            
        if True:#self.flux[section] == 0:
            force_check = True
        else:
            force_check = np.max(np.abs(self.RHS[subset])) < self.ftol * self.flux[section]/(self.iter+1)
        increment = self.uphi[subset] - self.uphi_old[subset]
        if np.max(np.abs(increment)) == 0:
            corr_check = True
        else:
            corr_check = np.max(np.abs(du)) < self.ctol * np.max(np.abs(increment))

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
        else:
            raise ValueError("Unknown solver method specified")
            
        self.plot()
        Disp = []
        Reaction = []
        while self.t < self.t_max:
            self.Increment()
            Solve()
            if solver.step >= 400:
                solver.dt = 2e-4
#            if solver.dt < 9e-3 and (solver.t % (10*solver.dt) < 1e-4*solver.dt or 10*solver.dt-(solver.t % (10*solver.dt)) < 1e-4*solver.dt):
#                solver.dt *= 10
#            if solver.t > 0.44-1e-8:
#                solver.dt = 1e-3
#                if solver.t > 0.444-1e-8:
#                    solver.dt = 1e-4
#                    if solver.t > 0.4449-1e-8:
#                        solver.dt = 1e-5
#                        
#            if solver.t > 0.445-1e-8:
#                solver.dt = 1e-4
#                if solver.t > 0.445-1e-8:
#                    solver.dt = 1e-3
#                    if solver.t > 0.45-1e-8:
#                        solver.dt = 1e-2
            
            Disp.append(self.uphi[1::3].max())
            Reaction.append( np.linalg.norm(self.RHS[(self.dim+1)*np.array(self.NSet['TOP'])+1]) )
            print "Time: ", self.t
            if int(self.t / plot_frequency) > int((self.t - self.dt) / plot_frequency):
                self.plot()
                
        return Disp, Reaction
        
    def plot(self, amp=1e0, data=['uphi'], save=True, suffix=''):
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
        
        Returns
        -------
        None
        """
        
        # Checking if any energies are to be plotted
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
            else:
                print "Unkown variable to plot %s"%datum
                raise ValueError("Unkown plotting variable")

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
                
            plt.figure("Display", figsize=(10,10))
            self.patch_collection.set_array(colors)
            self.cbar.set_clim(colors.min(), colors.max())
            self.cbar.draw_all()
            plt.axis('equal')
            plt.xlim(minim[0], maxim[0])
            plt.ylim(minim[1], maxim[1])
            plt.title('%s_%f'%(datum, self.t))
            plt.draw()
            plt.pause(0.05)
            if save:
                plt.savefig("%s\\%s_step_%i%s.png"%(self.Folder, datum, self.step, suffix))
             
        
if __name__ == "__main__":
    
    solver = Solver('Asymmetric.inp', getcwd() + '\\Asym_Coupled_SmallerStepFull2')
#    solver = Solver('Simple_Test.inp')
#    solver = Solver('Coarse_Double_Crack.inp', getcwd() + '\\Steps')
    
    solver.Resume(getcwd() + '\\Asym_Coupled_SmallerStep\\Step_402.npy', step=402)
    
#    try:
#        get_ipython().__class__.__name__
#    except:
#        solver.Init_Parallel()
    import time
    t0 = time.time()
    Disp, Reaction = solver.run(1e-10, 'Full')
    print time.time()-t0
    plt.figure(figsize=(12,12))
    plt.plot(Disp, np.array(Reaction), 'b-x')
    plt.savefig(solver.Folder + '\\Force_Displacement.png')
    print solver.solves
    
    
            