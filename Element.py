# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 14:03:42 2018

@author: Darin
"""

import numpy as np
from matplotlib.patches import Polygon

class Element:
    """
    Functions avaiable in this class
    --------------------------------
    __init__ : constructor
    _Get_GP : Gets the Gauss points
    _Construct_Shape : Constructs shape functions and derivatives in mapped coordinates
    _Shape_Func : Gets shape function and derivatives in parent coordiates
    _Assemble_B : Assembles field-to-gradient matrices
    _EigenStressStrain : Calculates principle stresses and strains
    _EigStrain : Eigenvalue solver for 2D strain matrix
    _DiffEig : Calculates derivative of eigenvalues wrt to the matrix
    _Constitutive : Provides constitutive relation for stress, stiffness, and energy
    _Calc_Energy : Calculates strain energies
    Get_Energy : Returns strain energies
    Local_Assembly : Assembles local stiffness matrix and rhs vector
    Update_Energy : Updates maximum energy values
    
    Objects stored in this class
    ----------------------------
    nodes : array_like
        Node numbers attached to this element
    eltype : string
        Element type ('L2', 'Q4', 'B8', or 'T3')
    dim : integer
        Dimensions of the element (1, 2, or 3)
    order : integer
        Integration order for element
    weights : 1D array
        Weights associated with each Gauss point
    Hist : 1D array
        Maximum strain energy at each gauss point
    Gc : scalar
        Griffith's fractur energy
    lc : scalar
        Critical length factor
    E : scalar
        Young's Modulus
    nu : scalar
        Poisson's ratio
    lam, mu : scalars
        Lame paramters lambda and mu.  Only used if E,nu = None
    k : scalar
        Minimum stiffness of element
    aniso : bool
        Flag for using anisotropic damage (True) or uniform damage (False)
    N : array_like
        Shape function values at the Gauss points
    J : array_like
        Determinants of the Jacobian matrix at the Gauss points
    B_u : array_like
        Matrices to convert displacements to strain at Gauss points
    B_phi : array_like
        Matrices to convert damage to gradient of damage at Gauss points
    """
    def __init__(self, eltype, nodes, coords, order=None, Gc=500, lc=1, E=1000,
                 nu=0.3, lam=None, mu=None, k=1e-10, aniso=True):
        """Constructor for element class
        
        Parameters
        ----------
        eltype : string
            Element type ('L2', 'Q4', 'B8', or 'T3')
        nodes : array_like
            Node numbers attached to this element
        coords : array_like
            Coordinates of nodes attached to this element
        order : integer, optional
            Integration order for element, defaults to minimum for full
            accuracy in an undeformed element
        Gc : scalar, optional
            Griffith's fractur energy
        lc : scalar, optional
            Critical length factor
        E : scalar, optional
            Young's Modulus
        nu : scalar, optional
            Poisson's ratio
        lam, mu : scalars, optional
            Lame paramters lambda and mu.  Only used if E,nu = None
        k : scalar, optional
            Minimum stiffness of element
        aniso : bool, optional
            Flag for using anisotropic damage (True) or uniform damage (False)
        
        Adds to class object
        --------------------
        nodes : array_like
            Node numbers attached to this element
        eltype : string
            Element type ('L2', 'Q4', 'B8', or 'T3')
        dim : integer
            Dimensions of the element (1, 2, or 3)
        order : integer
            Integration order for element
        Gc : scalar
            Griffith's fractur energy
        lc : scalar
            Critical length factor
        E : scalar
            Young's Modulus
        nu : scalar
            Poisson's ratio
        lam, mu : scalars
            Lame paramters lambda and mu.  Only used if E,nu = None
        k : scalar
            Minimum stiffness of element
        aniso : bool
            Flag for using anisotropic damage (True) or uniform damage (False)
        """
        
        if eltype not in ['L2', 'Q4', 'B8', 'T3']:
            raise TypeError("Unknown element type (%s) specified"%eltype)
            
        self.nodes = nodes
        self.eltype = eltype
        self.dim = coords.shape[1]
        self.dof = np.array([d+nodes*(self.dim+1) for d in range(self.dim+1)]).T.reshape(-1)
        self.udof = np.arange(self.dof.size)
        self.phidof = self.udof[self.dim::self.dim+1]
        self.udof = np.delete(self.udof,self.phidof)
        if order is None:
            if eltype == 'L2' or eltype == 'T3':
                self.order = 1
            elif eltype == 'Q4':
                self.order = 2
            elif eltype == 'B8':
                self.order = 3
        else:
            self.order = order

        self.Set_Properties(Gc, lc, E, nu, lam, mu, k, aniso)
        self._Construct_Shape(coords)
        self.patch = Polygon(coords, True)
        
    def Set_Properties(self, Gc=500, lc=1, E=1000, nu=0.3, lam=None, mu=None,
                       k=1e-10, aniso=True):
        """Overrides initial settings for material parameters
        
        Parameters
        ----------
        Gc : scalar, optional
            Griffith's fractur energy
        lc : scalar, optional
            Critical length factor
        E : scalar, optional
            Young's Modulus
        nu : scalar, optional
            Poisson's ratio
        lam, mu : scalars, optional
            Lame paramters lambda and mu.  Only used if E,nu = None
        k : scalar, optional
            Minimum stiffness of element
        """
        self.Gc = Gc
        self.lc = lc
        self.k = k
        self.aniso = aniso
        if E is None or nu is None:
            if lam is None or mu is None:
                raise ValueError("Must specify either E and nu or lam and mu")
            else:
                self.E = mu*(3*lam+2*mu)/(lam+mu)
                self.nu = lam/(2*(lam+mu))
                self.lam = lam
                self.mu = mu
        else:
            self.E = E
            self.nu = nu
            self.lam = E*nu/((1+nu)*(1-2*nu))
            self.mu = E/(2*(1+nu))
        
    def _Get_GP(self):
        """Assumes L2, Q4, or B8 elements or 2D Tri elements
        
        Parameters
        ----------
        None
            
        Returns
        -------
        GP : 2D array
            Quadrature points
        
        Adds to class object
        --------------------
        weights : 1D array
            weights associated with each Gauss point
        """
        
        if self.eltype == 'T3':
            if self.dim != 2:
                raise ValueError('Can only use line, quad, box, or tri elements')
            # Tri elements
            if self.order == 1:
                GP = np.array([[1./3, 1./3]])
                self.weights = np.array([1./2])
            elif self.order == 2:
                GP = np.array([[1./6, 1./6], [2./3, 1./6], [1./6, 2./3]])
                self.weights = np.array([1./6, 1./6, 1./6])
            elif self.order == 3:
                GP = np.array([[1./3, 1./3], [1./5, 1./5], [1./5, 3./5], [3./5, 1./5]])
                self.weights = np.array([-27., 25., 25., 25.])/96
        else:
            # Quad elements
            GP = np.zeros((self.order*2**(self.dim-1), self.dim))
            if self.order == 1:
                pnts = np.array([0])
                wgts = np.array([2])
            elif self.order == 2:
                pnts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
                wgts = np.array([1, 1])
            elif self.order == 3:
                pnts = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)])
                wgts = np.array([5./9, 8./9, 5./9])
                
            GP[:,0] = np.tile(pnts,2**(self.dim-1))
            self.weights = np.tile(wgts, 2**(self.dim-1))
            for i in range(1,self.dim):
                GP[:,i] = GP[:,i-1].T.reshape(2,-1).T.reshape(-1)
                self.weights *= self.weights.T.reshape(2,-1).T.reshape(-1)
                
        return GP
                
    def _Construct_Shape(self, coords):
        """Constructs shape functions, derivatives and B matrices for an element
        
        Parameters
        ----------
        coords : array_like
            Coordinates of nodes attached to this element
            
        Returns
        -------
        None
        
        Adds to class object
        --------------------
        N : 2D array
            Shape function values (dim 1) at the Gauss points (dim 0)
        J : 1D array
            Determinant of jacobian at each Gauss point
        B_u : 3D array, optional
            B matrices for u
        B_phi : 2D array, optional
            B vectors for phi (same as dNdx since phi is a scalar)
        Hist : 1D array
            Maximum strain energy values (set to zero initally)
        """
        
        GP = self._Get_GP()
        self.N, dNdxi = self._Shape_Func(GP)
        self.J, self.B_u, self.B_phi = self._Assemble_B(dNdxi, coords)
        self.Hist = np.zeros(self.weights.size)
    
    def _Shape_Func(self, GP):
        """Assumes L2, Q4, or B8 elements or 2D Tri elements
        
        Parameters
        ----------
        GP : ND array
            Gauss point coordinates
            
        Returns
        -------
        N : 2D array
            Shape function values (dim 1) at the Gauss points (dim 0)
        dNdxi : 3D array
            Shape function derivatives (dim 1) wrt parent coordinates (dim 2) at the Gauss points (dim 0)
        """
        
        
        if self.eltype == 'L2':
            N = np.zeros((GP.shape[0], 2))
            dNdxi = np.zeros((GP.shape[0], 2, GP.shape[1]))
            xi = GP[:,0]
            # N1 = 1/2-xi/2, N2 = 1/2+xi/2
            N[:,0] = 1./2 - xi/2
            N[:,1] = 1./2 + xi/2
            
            dNdxi[:,0,0] = -1./2
            dNdxi[:,1,0] =  1./2
        elif self.eltype == 'Q4':
            N = np.zeros((GP.shape[0], 4))
            dNdxi = np.zeros((GP.shape[0], 4, GP.shape[1]))
            xi = GP[:,0]; eta = GP[:,1]
            # N1 = 1/4*(1-xi)(1-eta), N2 = 1/4*(1+xi)(1-eta),
            # N3 = 1/4*(1+xi)(1+eta), N4 = 1/4*(1-xi)(1+eta)
            N[:,0] = 1./4*(1-xi)*(1-eta)
            N[:,1] = 1./4*(1+xi)*(1-eta)
            N[:,2] = 1./4*(1+xi)*(1+eta)
            N[:,3] = 1./4*(1-xi)*(1+eta)
            
            dNdxi[:,0,0] = -1./4*(1-eta); dNdxi[:,0,1] = -1./4*(1-xi)
            dNdxi[:,1,0] =  1./4*(1-eta); dNdxi[:,1,1] = -1./4*(1+xi)
            dNdxi[:,2,0] =  1./4*(1+eta); dNdxi[:,2,1] =  1./4*(1+xi)
            dNdxi[:,3,0] = -1./4*(1+eta); dNdxi[:,3,1] =  1./4*(1-xi)
        elif self.eltype == 'B8':
            N = np.zeros((GP.shape[0], 8))
            dNdxi = np.zeros((GP.shape[0], 8, GP.shape[1]))
            xi = GP[:,0]; eta = GP[:,1]; zeta = GP[:,2]
            # N1 = 1/8*(1-xi)*(1-eta)*(1-zeta)
            N[:,0]       =  1.0/8*(1-xi)*(1-eta)*(1-zeta)
            dNdxi[:,0,0] = -1.0/8*(1-eta)*(1-zeta);
            dNdxi[:,0,1] = -1.0/8*(1-xi)*(1-zeta);
            dNdxi[:,0,2] = -1.0/8*(1-xi)*(1-eta);
            # N2 = 1/8*(1+xi)*(1-eta)*(1-zeta)
            N[:,1]      =  1.0/8*(1+xi)*(1-eta)*(1-zeta)
            dNdxi[:,1,0] =  1.0/8*(1-eta)*(1-zeta);
            dNdxi[:,1,1] = -1.0/8*(1+xi)*(1-zeta);
            dNdxi[:,1,2] = -1.0/8*(1+xi)*(1-eta);
            # N3 = 1/8*(1+xi)*(1+eta)*(1-zeta)
            N[:,2]       =  1.0/8*(1+xi)*(1+eta)*(1-zeta)
            dNdxi[:,2,0] =  1.0/8*(1+eta)*(1-zeta);
            dNdxi[:,2,1] =  1.0/8*(1+xi)*(1-zeta);
            dNdxi[:,2,2] = -1.0/8*(1+xi)*(1+eta);
            # N4 = 1/8*(1-xi)*(1+eta)*(1-zeta)
            N[:,3]       =  1.0/8*(1-xi)*(1+eta)*(1-zeta)
            dNdxi[:,3,0] = -1.0/8*(1+eta)*(1-zeta);
            dNdxi[:,3,1] =  1.0/8*(1-xi)*(1-zeta);
            dNdxi[:,3,2] = -1.0/8*(1-xi)*(1+eta);
            # N5 = 1/8*(1-xi)*(1-eta)*(1+zeta)
            N[:,4]       =  1.0/8*(1-xi)*(1-eta)*(1+zeta)
            dNdxi[:,4,0] = -1.0/8*(1-eta)*(1+zeta);
            dNdxi[:,4,1] = -1.0/8*(1-xi)*(1+zeta);
            dNdxi[:,4,2] =  1.0/8*(1-xi)*(1-eta);
            # N6 = 1/8*(1+xi)*(1-eta)*(1+zeta)
            N[:,5]       =  1.0/8*(1+xi)*(1-eta)*(1+zeta)
            dNdxi[:,5,0] =  1.0/8*(1-eta)*(1+zeta);
            dNdxi[:,5,1] = -1.0/8*(1+xi)*(1+zeta);
            dNdxi[:,5,2] =  1.0/8*(1+xi)*(1-eta);
            # N7 = 1/8*(1+xi)*(1+eta)*(1+zeta)
            N[:,6]       =  1.0/8*(1+xi)*(1+eta)*(1+zeta)
            dNdxi[:,6,0] =  1.0/8*(1+eta)*(1+zeta);
            dNdxi[:,6,1] =  1.0/8*(1+xi)*(1+zeta);
            dNdxi[:,6,2] =  1.0/8*(1+xi)*(1+eta);
            # N8 = 1/8*(1-xi)*(1+eta)*(1+zeta)
            N[:,7]       =  1.0/8*(1-xi)*(1+eta)*(1+zeta)
            dNdxi[:,7,0] = -1.0/8*(1+eta)*(1+zeta);
            dNdxi[:,7,1] =  1.0/8*(1-xi)*(1+zeta);
            dNdxi[:,7,2] =  1.0/8*(1-xi)*(1+eta);
        elif self.eltype == 'T3':
            N = np.zeros((GP.shape[0], 3))
            dNdxi = np.zeros((GP.shape[0], 3, GP.shape[1]))
            xi = GP[:,0]; eta = GP[:,1]
            # N1 = xi, N2 = eta, N3 = 1 - xi - eta
            N[:,0] = xi
            N[:,1] = eta
            N[:,2] = 1.0 - xi - eta
            
            dNdxi[:,0,0] =  1.0; dNdxi[:,0,1] =  0.0
            dNdxi[:,1,0] =  0.0; dNdxi[:,1,1] =  1.0
            dNdxi[:,2,0] = -1.0; dNdxi[:,2,1] = -1.0
            
            
        else:
            raise TypeError('Invalid element type specified')
            
        return N, dNdxi
    
    def _Assemble_B(self, dNdxi, coords):
        """Converts dNdxi to dNdx and assembles B matrix for u or phi
        
        Parameters
        ----------
        dNdxi : 3D array
            Shape function derivatives (dim 1) wrt parent coordinates (dim 2) at the Gauss points (dim 0)
        coords : array_like
            Coordinates of nodes attached to this element
            
        Returns
        -------
        J : 1D array
            Determinant of jacobian at each Gauss point
        B_u : 3D array, optional
            B matrices for u
        B_phi : 2D array, optional
            B vectors for phi (same as dNdx since phi is a scalar)
        """
        
        nGP = dNdxi.shape[0]
        
        J = np.zeros(nGP)
        dNdx = np.zeros((self.dim, dNdxi.shape[1], nGP))
        B_u = np.zeros(((self.dim**2+self.dim)/2, coords.size, nGP))
            
        for i in range(nGP):
            Jac = np.dot(dNdxi[i,:,:].T, coords)
            J[i] = np.linalg.det(Jac)
            
            dNdx[:,:,i] = np.linalg.solve(Jac, dNdxi[i,:,:].T)
            
            if self.dim == 1:
                B_u[:,:,i] = dNdx[:,:,i]
            if self.dim == 2:
                B_u[0,0::2,i] = dNdx[0,:,i]
                B_u[1,1::2,i] = dNdx[1,:,i]
                B_u[2,0::2,i] = dNdx[1,:,i]
                B_u[2,1::2,i] = dNdx[0,:,i]
            elif self.dim == 3:
                B_u[0,0::3,i] = dNdx[0,:,i]
                B_u[1,1::3,i] = dNdx[1,:,i]
                B_u[2,2::3,i] = dNdx[2,:,i]
                B_u[3,0::3,i] = dNdx[1,:,i]
                B_u[3,1::3,i] = dNdx[0,:,i]
                B_u[4,1::3,i] = dNdx[2,:,i]
                B_u[4,2::3,i] = dNdx[1,:,i]
                B_u[5,0::3,i] = dNdx[2,:,i]
    
        return J, B_u, dNdx
    
    def _EigenStressStrain(self, eps, phi, couple=False):
        """Gets the eigenstresses and eigenstrains
        
        Parameters
        ----------
        eps : array_like
            The strain tensor in VECTOR form
        phi : scalar
            The damage variable at the point stresses and strains are measured
        couple : boolean, optional
            Whether to include coupling terms
            
        Returns
        -------
        eps_princ : vector
            Principal strains
        sig_princ : vector
            Principal stresses
        dlam : array_like
            First derivative of principal strains with respect to all strains
        d2lam : array_like
            Second derivative of principal strains with respect to all strains
        H : scalar
            1 or 0 indicating whether trace of strains applies to strain energy
        Hi : vector
            1s or 0s indicating whether each principal strain applies to strain energy
        Lmat : array_like
            Matrix to convert principal strains to principl stresses
        """
        
        eps_princ = self._EigStrain(eps)
        dlam, d2lam = self._DiffEig(eps, eps_princ)
        
        H = 1
        if ((eps_princ[0] + eps_princ[1]) < 0 and self.aniso):
            H = 0
        
        Hi = np.empty(2, dtype=int)
        Hi.fill(1)
        if (eps_princ[0] < 0 and self.aniso):
            Hi[0] = 0
        if (eps_princ[1] < 0 and self.aniso):
            Hi[1] = 0
                
        Lmat = np.empty((2,2))    
        gamma = self.lam * ((1-self.k) * (1 - H*phi)**2 + self.k)
        Lmat.fill(gamma)
        Lmat[0,0] += 2*self.mu*((1-self.k) * (1-Hi[0]*phi)**2 + self.k)
        Lmat[1,1] += 2*self.mu*((1-self.k) * (1-Hi[1]*phi)**2 + self.k)
        
        sig_princ = np.dot(Lmat, eps_princ)
        
        if couple:
            # dLmat/dphi
            dLmat = 0 * Lmat
            gamma = -2 * self.lam * (1-self.k) * H * (1 - H*phi)
            dLmat += gamma
            dLmat[0,0] -= 4 * self.mu * (1-self.k) * Hi[0] * (1 - Hi[0]*phi)
            dLmat[1,1] -= 4 * self.mu * (1-self.k) * Hi[1] * (1 - Hi[1]*phi)
            
            return eps_princ, sig_princ, dlam, d2lam, H, Hi, Lmat, dLmat
        
        else:
            return eps_princ, sig_princ, dlam, d2lam, H, Hi, Lmat
    
    def _EigStrain(self, eps):
        """Gets the eigenstresses and eigenstrains
        
        Parameters
        ----------
        eps : vector
            The strain tensor in VECTOR form
            
        Returns
        -------
        lam : vector
            The principle strains
        """
        lam = np.empty(2)
        lam.fill((eps[0]+eps[1])/2)
        shift = np.sqrt((eps[0]+eps[1])**2 - 4*(eps[0]*eps[1]-eps[2]**2/4))/2
        lam[0] += shift
        lam[1] -= shift
        
        return lam
    
    def _DiffEig(self, eps, lam):
        """Returns the derivative of the eigenvalues wrt the matrix
        
        Parameters
        ----------
        eps : vector
            The strain tensor in VECTOR form
        lam : vector
            The principle strains
            
        Returns
        -------
        dlam : array_like
            First derivative of the eigenvalues
        d2lam : array_like
            Second derivatives of the eigenvalues
        """
        
        dlam = np.zeros((2,3))
        d2lam = np.zeros((3,3,2))
        
        R = np.sqrt((lam[0]+lam[1])**2 - 4*(lam[0]*lam[1]))
        if ((R < (1e-8*(eps[0]+eps[1]))) or (R == 0)):
            # Avoid breakdown that occurs when eigenvalue is multiple or eps = 0
            dlam[0,0] = 1.
            dlam[0,1] = 0.
            dlam[0,2] = 0.
            dlam[1,0] = 0.
            dlam[1,1] = 1.
            dlam[1,2] = 0.
            d2lam[2,2,0] = 1./4
            d2lam[2,2,1] = 1./4
        else:
            twoR = 2*R
            cost = (eps[0]-eps[1])/R
            sint = eps[2]/R
            
#            dlam[0,0] = (1+cost)/2
#            dlam[0,1] = (1-cost)/2
#            dlam[0,2] = sint/2
#            dlam[1,0] = (1-cost)/2
#            dlam[1,1] = (1+cost)/2
#            dlam[1,2] = -sint/2
#            
#            d2lam[0,0,0] = (1-cost**2)/twoR
#            d2lam[0,1,0] = (cost**2-1)/twoR
#            d2lam[0,2,0] = -sint*cost/twoR
#            d2lam[1,0,0] = (cost**2-1)/twoR
#            d2lam[1,1,0] = (1-cost**2)/twoR
#            d2lam[1,2,0] = sint*cost/twoR
#            d2lam[2,0,0] = -sint*cost/twoR
#            d2lam[2,1,0] = sint*cost/twoR
#            d2lam[2,2,0] = (1-sint**2)/twoR
            
            dlam[0,0] = (1+cost)/2
            dlam[0,1] = (1-cost)/2
            dlam[0,2] = sint/2
            dlam[1,0] = dlam[0,1]
            dlam[1,1] = dlam[0,0]
            dlam[1,2] = -dlam[0,2]
            
            d2lam[0,0,0] = (1-cost**2)/twoR
            d2lam[0,1,0] = -d2lam[0,0,0]
            d2lam[0,2,0] = -sint*cost/twoR
            d2lam[1,0,0] = -d2lam[0,0,0]
            d2lam[1,1,0] = d2lam[0,0,0]
            d2lam[1,2,0] = -d2lam[0,2,0]
            d2lam[2,0,0] = d2lam[0,2,0]
            d2lam[2,1,0] = -d2lam[0,2,0]
            d2lam[2,2,0] = (1-sint**2)/twoR
            
            d2lam[:,:,1] = -d2lam[:,:,0]
        
        return dlam, d2lam
    
    def _Constitutive(self, eps, phi, energy=True, couple=False):
        """Provides constitutive relations to get stress and tangent stiffness
        
        Parameters
        ----------
        eps : array_like
            The strain tensor in VECTOR form
        phi : scalar
            The damage variable at the point stresses and strains are measured
        energy : bool, optional
            Flag whether energies should be calculated as well
        couple : boolean, optional
            Whether to include coupling terms
            
        Returns
        -------
        stress : vector
            stress values in VECTOR form
        Cmat : array_like
            tangent stiffness matrix
        elast_eng : scalar
            Elastic energy
        elast_ten_eng : scalar
            Elastic tensile energy
        eng : scalar
            Inelastic energy
        """
        if (eps == 0.).all():
            # When no strain is present, no priniciple strains exist
            stress = 0*eps
            Cmat = np.array([[1-self.nu, self.nu, 0.], [self.nu, 1-self.nu, 0.],
                             [0., 0., 0.5-self.nu]])
            Cmat *= self.E / ( (1+self.nu) * (1-2*self.nu) )
            elast_eng = 0
            elast_ten_eng = 0
            eng = 0
            dstress = stress.copy()
            dten_eng = stress.copy()

        else:
            if couple:
                eps_princ, sig_princ, dlam, d2lam, H, Hi, Lmat, dLmat = self._EigenStressStrain(
                                            eps, phi, couple)
            else:
                eps_princ, sig_princ, dlam, d2lam, H, Hi, Lmat = self._EigenStressStrain(
                                            eps, phi, couple)
                    
            
            stress = np.dot(sig_princ, dlam)
            # Derivative of prinicipal stress wrt damage, only necessary for coupling
            if couple:
                dpsdphi = np.dot(dLmat, eps_princ)
                dstress = np.dot(dpsdphi, dlam)
            Cmat = np.dot(dlam.T,np.dot(Lmat,dlam)) + np.dot(d2lam, sig_princ)
            if energy:
                if couple:
                    elast_eng, elast_ten_eng, eng, dten_eng = self._Calc_Energy(
                            eps_princ, phi, H, Hi, couple=couple)
                    dten_eng = np.dot(dten_eng, dlam)
                else:
                    elast_eng, elast_ten_eng, eng = self._Calc_Energy(eps_princ, phi, H, Hi, couple=couple)
                    
        
        if energy:
            if couple:
                return stress, dstress, Cmat, elast_eng, elast_ten_eng, eng, dten_eng
            else:
                return stress, Cmat, elast_eng, elast_ten_eng, eng
        else:
            if couple:
                return stress, dstress, Cmat
            else:
                return stress, Cmat
    
    def _Calc_Energy(self, eps_princ, phi, H, Hi, types=7, couple=False):
        """Calculates elastic, elastic tensile, and inelastic energy
        
        Parameters
        ----------
        eps_princ : array_like
            The principle strains
        phi : scalar
            The damage variable at the point stresses and strains are measured
        H : scalar
            1 or 0 indicating whether trace of strains applies to strain energy
        Hi : vector
            1s or 0s indicating whether each principal strain applies to strain energy
        types : int, optional
            Which energies to return (types&1:elastic, types&2:elastic tensile,
                                      types&4:inelastic)
        couple : boolean, optional
            Whether to include coupling terms
            
        Returns
        -------
        elast_eng : scalar
            Elastic energy
        elast_ten_eng : scalar
            Elastic tensile energy
        eng : scalar
            Inelastic energy
        dten_eng : scalar
            Derivative of elastic tensile energy wrt eps_princ
        """
        if eps_princ.size == 1:
            eps_tr = eps_princ[0]
        elif eps_princ.size == 2:
            eps_tr = eps_princ[0] + eps_princ[1]
        elif eps_princ.size == 3:
            eps_tr = eps_princ[0] + eps_princ[1] + eps_princ[2]
        
        ret = []
        if types & 1:
            # elast_eng
            ret.append(self.lam/2*eps_tr**2 + self.mu*np.dot(eps_princ.T,eps_princ))
        if types & 2:
            # elast_ten_eng
            ret.append(self.lam/2*H*eps_tr**2 + self.mu*np.dot(eps_princ.T,eps_princ*Hi))
        if types & 4:
            # eng
            ret.append(self.lam/2*((1-self.k)*(1-H*phi)**2 + self.k)*eps_tr**2 +
                   self.mu*np.dot(eps_princ.T,eps_princ*((1-self.k)*(1-Hi*phi)**2 + self.k)))
        if couple:
            # dten_eng
            ret.append(self.lam * H * eps_tr + 2*self.mu * Hi * eps_princ)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret
    
    def Get_Energy(self, eps, phi, types=7):
        """ Returns elastic, elastic tensile, and inelastic energy 
            based on strain/damage values
        
        Parameters
        ----------
        eps : array_like
            The strain tensor in VECTOR form
        phi : scalar
            The damage variable at the point stresses and strains are measured
        types : int, optional
            Which energies to return (types&1:elastic, types&2:elastic tensile,
                                      types&3:inelastic)
            
        Returns
        -------
        elast_eng : scalar
            Elastic energy
        elast_ten_eng : scalar
            Elastic tensile energy
        eng : scalar
            Inelastic energy
        """
        
        eps_princ = self._EigStrain(eps)
        
        H = 1
        if ((eps_princ[0] + eps_princ[1]) < 0 and self.aniso):
            H = 0
        
        Hi = np.zeros(2, dtype=int)
        for i in range(2):
            Hi[i] = 1
            if (eps_princ[i] < 0 and self.aniso):
                Hi[i] = 0
        
        return self._Calc_Energy(eps_princ, phi, H, Hi, types=types, couple=False)
    
    def Energy(self, uphi, types=7, reduction=None):
        """ Returns elastic, elastic tensile, and inelastic energy at each
            Gauss point based on displacement/damage vector
        
        Parameters
        ----------
        uphi : array_like
            vector of displacements/damage for this element
        types : int, optional
            Which energies to return (types&1:elastic, types&2:elastic tensile,
                                      types&3:inelastic)
        reduction : operation, optional
            An operation to perform to reduce the values at each Gauss point
            to a single value
            
        Returns
        -------
        elast_eng : scalar
            Elastic energy
        elast_ten_eng : scalar
            Elastic tensile energy
        eng : scalar
            Inelastic energy
        """
        
        energies = np.zeros((self.weights.size, (types&4)/4 + (types&2)/2 + (types&1)))
        for i in range(self.weights.size):
            phi = np.dot(self.N[i,:], uphi[self.phidof])
            eps = np.dot(self.B_u[:,:,i], uphi[self.udof])
            energies[i,:] = self.Get_Energy(eps, phi, types=7)
        
        if reduction is None:
            return energies
        else:
            return reduction(energies, axis=0)

    def Stress(self, uphi, reduction=None):
        """ Returns stress at each Gauss point based on displacement/damage vector
        
        Parameters
        ----------
        uphi : array_like
            vector of displacements/damage for this element
        reduction : operation, optional
            An operation to perform to reduce the values at each Gauss point
            to a single value
            
        Returns
        -------
        stress : array_like
            stress components in VECTOR form
        """

        stress = np.zeros((self.weights.size, 3))
        for i in range(self.weights.size):
            phi = np.dot(self.N[i,:], uphi[self.phidof])
            eps = np.dot(self.B_u[:,:,i], uphi[self.udof])
            stress[i,:] = el._Constitutive(eps, phi, energy=False, couple=False)[0]
            
        if reduction is None:
            return stress
        else:
            return reduction(stress, axis=0)
    
    def Local_Assembly(self, uphi, section, Assemble=3, couple=False):
        """Assembles the local tangent stiffness matrix and internal force vector
        
        Parameters
        ----------
        uphi : 1D array
            The current approximation of u and phi
        section : scalar
            Part of the matrix to assemble ('uu', 'pp', 'up', or 'pu' or 'all')
            'uu' and 'pp' also return the internal force vectors.
        Assemble : integer, optional
            If Assemble & 1 == 1, assemble K
            If Assemble & 2 == 2, assemble RHS
            If both true, assemble both
        couple : boolean, optional
            Whether to include coupling terms
            
        Returns
        -------
        K_loc : 2D array
            Local tangent stiffness matrix
        F_loc : 1D array
            Local internal force vector
        """
        
        section = section.upper()
        
        # Initialization        
        if section == 'ALL':
            K = np.zeros((uphi.size,uphi.size))
            F = np.zeros(uphi.size)
        if section == 'UU':
            K = np.zeros((self.udof.size,self.udof.size))
            F = np.zeros(self.udof.size)
        elif section == 'UP':
            K = np.zeros((self.udof.size,self.phidof.size))
            F = None
        elif section == 'PU':
            K = np.zeros((self.phidof.size,self.udof.size))
            F = None
        elif section == 'PP':
            K = np.zeros((self.phidof.size,self.phidof.size))
            F = np.zeros(self.phidof.size)

        # Loop over integration points
        for i in range(self.weights.size):
            phi = np.dot(self.N[i,:], uphi[self.phidof])
            
            if section == 'ALL' or section == 'UU' or couple:
                eps = np.dot(self.B_u[:,:,i], uphi[self.udof])
            if couple and section == 'ALL':
                stress, dstress, D, elast_eng, elast_ten_eng, eng, dten_eng = self._Constitutive(eps, phi, energy=True, couple=couple)
            elif section == 'ALL':
                stress, D, elast_eng, elast_ten_eng, eng = self._Constitutive(eps, phi, energy=True, couple=False)
            elif section == 'UU':
                stress, D = self._Constitutive(eps, phi, energy=False, couple=False)
            
            coeff = self.J[i]*self.weights[i]
            if section == 'ALL':
                # UU
                if Assemble & 1 == 1:
                    K[self.udof[:,np.newaxis],self.udof] += np.dot(self.B_u[:,:,i].T, 
                                         np.dot(D, self.B_u[:,:,i]))*coeff
                if Assemble & 2 == 2:
                    F[self.udof] += np.dot(self.B_u[:,:,i].T, stress)*coeff
                
                # PP
                energy = max(elast_ten_eng, self.Hist[i])
                if Assemble & 1 == 1:
                    K[self.phidof[:,np.newaxis],self.phidof] += (self.Gc * self.lc * 
                             np.dot(self.B_phi[:,:,i].T, self.B_phi[:,:,i]) * coeff)
                    K[self.phidof[:,np.newaxis],self.phidof] += ((self.Gc / self.lc +
                            2*(1-self.k)*energy) * np.outer(self.N[i,:].T, self.N[i,:])
                            * coeff)
                if Assemble & 2 == 2:
                    gradphi = np.dot(self.B_phi[:,:,i], uphi[self.phidof])
                    F[self.phidof] += self.Gc * self.lc * np.dot(self.B_phi[:,:,i].T, gradphi) * coeff
                    F[self.phidof] += (self.Gc / self.lc + 2*(1-self.k)*energy) * self.N[i,:].T * phi * coeff
                    F[self.phidof] -= 2*(1-self.k)*energy * self.N[i,:].T * coeff
                    
                if couple and Assemble & 1 == 1:
                    # UP
                    K[self.udof[:,np.newaxis],self.phidof] += np.outer(np.dot(self.B_u[:,:,i].T, 
                                         dstress), self.N[i,:])*coeff
                    # PU
                    if elast_ten_eng >= self.Hist[i]:
                        K[self.phidof[:,np.newaxis],self.udof] += (2*(phi-1) * 
                         (1-self.k) * np.outer(self.N[i,:].T, np.dot(dten_eng, 
                         self.B_u[:,:,i])) * coeff)
                
                
            elif section == 'UU':
#                if Assemble & 1 == 1:
#                    K += np.dot(self.B_u[:,:,i].T, np.dot(D, self.B_u[:,:,i]))*coeff
#                if Assemble & 2 == 2:
#                    F += np.dot(self.B_u[:,:,i].T, stress)*coeff
                self.Assemble_U(K, F, D, stress, i, coeff, Assemble)
            
            elif section == 'UP':
                pass
            
            elif section == 'PU':
                pass
            
            elif section == 'PP':
                if couple:
                    elast_ten_eng = self.Get_Energy(eps, phi, types=2)
                    # Use current estimated energy for calculation
                    energy = max(elast_ten_eng, self.Hist[i])
                else:
                    # Decoupled, ignore currrent approximated energy
                    energy = self.Hist[i]
#                if Assemble & 1 == 1:
#                    K += self.Gc * self.lc * np.dot(self.B_phi[:,:,i].T, self.B_phi[:,:,i]) * coeff
#                    K += (self.Gc / self.lc + 2*(1-self.k)*energy) * np.outer(self.N[i,:].T, self.N[i,:]) * coeff
#                if Assemble & 2 == 2:
#                    gradphi = np.dot(self.B_phi[:,:,i], uphi[self.phidof])
#                    F += self.Gc * self.lc * np.dot(self.B_phi[:,:,i].T, gradphi).reshape(-1) * coeff
#                    F += (self.Gc / self.lc + 2*(1-self.k)*energy) * self.N[i,:].T * phi * coeff
#                    F -= 2*(1-self.k)*energy * self.N[i,:].T * coeff
                self.Assemble_P(K, F, uphi, phi, i, coeff, energy, Assemble)

        return K, F
    
    def Assemble_U(self, K, F, D, stress, i, coeff, Assemble=3):
        if Assemble & 1 == 1:
            K += np.dot(self.B_u[:,:,i].T, np.dot(D, self.B_u[:,:,i]))*coeff
        if Assemble & 2 == 2:
            F += np.dot(self.B_u[:,:,i].T, stress)*coeff
            
    def Assemble_P(self, K, F, uphi, phi, i, coeff, energy, Assemble=3):
        if Assemble & 1 == 1:
            K += self.Gc * self.lc * np.dot(self.B_phi[:,:,i].T, self.B_phi[:,:,i]) * coeff
            K += (self.Gc / self.lc + 2*(1-self.k)*energy) * np.dot(self.N[i,None].T, self.N[i,None]) * coeff
        if Assemble & 2 == 2:
            gradphi = np.dot(self.B_phi[:,:,i], uphi[self.phidof])
            F += self.Gc * self.lc * np.dot(self.B_phi[:,:,i].T, gradphi) * coeff
            F += (self.Gc / self.lc + 2*(1-self.k)*energy) * self.N[i,:].T * phi * coeff
            F -= 2*(1-self.k)*energy * self.N[i,:].T * coeff

        
    def RHS_FD(self, uphi, section, delta=1e-8):
        """Calculates the right hand side using finite differences
        
        Parameters
        ----------
        uphi : 1D array
            The current approximation of u and phi
        section : scalar, optional
            Part of the RHS to assemble ('uu', 'pp' or 'all')
        delta : scalar, optional
            Finite difference step size
            
        Returns
        -------
        F_loc : 1D array
            Local internal force vector
        """
        
        F_loc = np.zeros(uphi.size)
        section = section.upper()
        if section == 'ALL':
            dof = np.arange(uphi.size)
        elif section == 'UU':
            dof = np.arange(0, uphi.size, self.dim+1).tolist()
            for i in range(1,self.dim):
                dof += np.arange(i, uphi.size, self.dim+1).tolist()
            dof = np.sort(dof)
        elif section == 'PP':
            dof = np.arange(self.dim, uphi.size, self.dim+1)
            
        for d in dof:
            plus = 0
            uphi_del = uphi.copy()
            uphi_del[d] += delta
            for i in range(self.weights.size):
                coeff = self.J[i]*self.weights[i]
                
                phi = np.dot(self.N[i,:],uphi_del[self.phidof])
                u = np.zeros((self.dim*uphi_del.size)/(self.dim+1))
                for j in range(self.dim):
                    u[j::self.dim] = np.dot(self.N[i,:],uphi_del[self.udof[j::self.dim]])
                    
                gradphi = np.dot(self.B_phi[:,:,i], uphi_del[self.phidof])
                eps = np.dot(self.B_u[:,:,i], uphi_del[self.udof])
                stress, D, elast_eng, elast_ten_eng, eng = self._Constitutive(eps, phi, energy=True, couple=False)
                temp1 = eng
                temp2 = self.Gc/2*(phi**2/self.lc + self.lc*np.dot(gradphi,gradphi))
                plus += (temp1 + temp2)*coeff

            minus = 0
            uphi_del = uphi.copy()
            uphi_del[d] -= delta
            for i in range(self.weights.size):
                coeff = self.J[i]*self.weights[i]
                
                phi = np.dot(self.N[i,:],uphi_del[self.phidof])
                u = np.zeros((self.dim*uphi_del.size)/(self.dim+1))
                for j in range(self.dim):
                    u[j::self.dim] = np.dot(self.N[i,:],uphi_del[self.udof[j::self.dim]])
                    
                gradphi = np.dot(self.B_phi[:,:,i], uphi_del[self.phidof])
                eps = np.dot(self.B_u[:,:,i], uphi_del[self.udof])
                stress, D, elast_eng, elast_ten_eng, eng = self._Constitutive(eps, phi, energy=True, couple=False)
                temp1 = eng
                temp2 = self.Gc/2*(phi**2/self.lc + self.lc*np.dot(gradphi,gradphi))
                minus += (temp1 + temp2)*coeff
          
            F_loc[d] = (plus-minus)/2/delta;
            
        return F_loc[dof]
    
    def K_FD(self, uphi, section, delta=1e-8):
        """Calculates the tangent stiffness matrix using finite differences on
        the RHS vector
        
        Parameters
        ----------
        uphi : 1D array
            The current approximation of u and phi
        section : scalar, optional
            Part of K to assemble ('uu', 'pp' or 'all')
        delta : scalar, optional
            Finite difference step size
            
        Returns
        -------
        K_loc : 1D array
            Local internal force vector
        """
        
        section = section.upper()
        if section == 'ALL':
            dof = np.arange(uphi.size)
        elif section == 'UU':
            dof = np.arange(0, uphi.size, self.dim+1).tolist()
            for i in range(1,self.dim):
                dof += np.arange(i, uphi.size, self.dim+1).tolist()
            dof = np.sort(dof)
        elif section == 'PP':
            dof = np.arange(self.dim, uphi.size, self.dim+1)
            
        K_loc = np.zeros((dof.size,uphi.size))
        for d in dof:
            uphi_del = uphi.copy()
            uphi_del[d] += delta
            plus = self.Local_Assembly(uphi_del, section)[1]

            uphi_del = uphi.copy()
            uphi_del[d] -= delta
            minus = self.Local_Assembly(uphi_del, section)[1]
          
            K_loc[:,d] = (plus-minus)/2/delta;
            
        return K_loc[:,dof]
        
    def K_FD2(self, uphi, section, delta=1e-8):
        """Calculates the tangent stiffness matrix using finite differences on
        the energy term
        
        Parameters
        ----------
        uphi : 1D array
            The current approximation of u and phi
        section : scalar, optional
            Part of K to assemble ('uu', 'pp' or 'all')
        delta : scalar, optional
            Finite difference step size
            
        Returns
        -------
        K_loc : 1D array
            Local internal force vector
        """
        
        section = section.upper()
        if section == 'ALL':
            dof = np.arange(uphi.size)
        elif section == 'UU':
            dof = np.arange(0, uphi.size, self.dim+1).tolist()
            for i in range(1,self.dim):
                dof += np.arange(i, uphi.size, self.dim+1).tolist()
            dof = np.sort(dof)
        elif section == 'PP':
            dof = np.arange(self.dim, uphi.size, self.dim+1)
            
        K_loc = np.zeros((uphi.size,uphi.size))
        for d in dof:
            for f in dof:
                for signs in [[1, 1], [1, -1], [-1, 1], [-1, -1]]:
                    uphi_del = uphi.copy()
                    uphi_del[d] += signs[0]*delta
                    uphi_del[f] += signs[1]*delta
                    for i in range(self.weights.size):
                        coeff = self.J[i]*self.weights[i]
                        
                        phi = np.dot(self.N[i,:],uphi_del[self.phidof])
                        u = np.zeros((self.dim*uphi_del.size)/(self.dim+1))
                        for j in range(self.dim):
                            u[j::self.dim] = np.dot(self.N[i,:],uphi_del[self.udof[j::self.dim]])
                            
                        gradphi = np.dot(self.B_phi[:,:,i], uphi_del[self.phidof])
                        eps = np.dot(self.B_u[:,:,i], uphi_del[self.udof])
                        stress, D, elast_eng, elast_ten_eng, eng = self._Constitutive(eps, phi, energy=True, couple=False)
                        temp1 = eng
                        temp2 = self.Gc/2*(phi**2/self.lc + self.lc*np.dot(gradphi,gradphi))
                        K_loc[d,f] += signs[0]*signs[1] * (temp1 + temp2)*coeff
        
        K_loc /= 4*delta*delta

        return K_loc[dof[:,np.newaxis],dof]
    
    def Update_Energy(self, uphi):
        """Updates maximum energy value
        
        Parameters
        ----------
        uphi : 1D array
            The current approximation of u and phi
            
        Returns
        -------
        None
        """
        
        for i in range(self.weights.size):
            phi = np.dot(self.N[i,:],uphi[self.phidof])
            eps = np.dot(self.B_u[:,:,i], uphi[self.udof])
            eps_princ = self._EigStrain(eps)

            H = 1
            if ((eps_princ[0] + eps_princ[1]) < 0 and self.aniso):
                H = 0
            
            Hi = np.zeros(2, dtype=int)
            for j in range(2):
                Hi[j] = 1
                if (eps_princ[j] < 0 and self.aniso):
                    Hi[j] = 0
                    
            elast_ten_eng = self._Calc_Energy(eps_princ, phi, H, Hi, types=2, couple=False)
            if self.Hist[i] > elast_ten_eng:
                Warning("Energy drop detected")
            self.Hist[i] = max(self.Hist[i], elast_ten_eng)