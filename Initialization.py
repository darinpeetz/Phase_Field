# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 11:37:07 2018

@author: Darin
"""

import numpy as np
from Element import Element
import scipy.io as sio
import scipy.stats as stats

def activate(line, active='none'):
    """Checks if we are reading nodes, elements, or something else
    
    Parameters
    ----------
    line : string
        The line to check for activation
        
    Returns
    -------
    active : string
        'nodes' if we are going to read nodes, 'elems' if we are going to read
        elements, 'prop' if properties, or 'none' otherwise
    """
    if active == 'step' or active == 'boundary':
        if 'END' in line.upper() and 'STEP' in line.upper():
            return 'none'
        elif '*BOUNDARY' in line.upper():
            return 'boundary'
        else:
            return 'step'

    active = 'none'
    if line[:5].upper() == '*NODE':
        active = 'nodes'
    elif line[:14].upper() == '*UEL PROPERTY,':
        active = 'prop'
    elif line[:9].upper() == '*ELEMENT,':
        active = 'elems'
    elif line[:5].upper() == '*NSET':
        if 'GENERATE' in line.upper():
            active = 'gennset'
        else:
            active = 'nset'
    elif line[:6].upper() == '*ELSET':
        if 'GENERATE' in line.upper():
            active = 'genelset'
        else:
            active = 'elset'
    elif line[:5].upper() == '*STEP':
        active = 'step'
    elif line[:10].upper() == '*AMPLITUDE':
        active = 'amplitude'
    
    return active
    
def Read_Mesh(filename):
    """Reads a mesh from a .inp file
    
    Parameters
    ----------
    filename : string
        name of the file
        
    Returns
    -------
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
    """
    
    node_list = []
    elem_nodes = []
    elem_types = []
    ElSet = {}
    NSet = {}
    Steps = []
    reset = False
    Amplitude = {'DEFAULT':np.array([[0., 0.], [1., 1.]])}
    with open(filename, 'r') as fh:
        active = 'none'
        for line in fh:
            if line[0:2] == '**':
                pass
            elif line[0] == '*':
                if active == 'step' or active == 'boundary':
                    newstep = False
                else:
                    newstep = True
                    
                active = activate(line.strip(), active)
                
                if 'nset' in active:
                    name = line.upper().strip('\n').strip('\r')[line.upper().index('NSET=')+5:].split(',')[0]
                    NSet[name] = []
                elif 'elset' in active:
                    name = line.upper().strip('\n').strip('\r')[line.upper().index('ELSET=')+6:].split(',')[0]
                    ElSet[name] = []
                elif active == 'elems':
                    if 'CPE3' in line or 'CPS3' in line:
                        eltype = 'T3'
                    elif 'CPE4' in line or 'CPS4' in line:
                        eltype = 'Q4'
                    else:
                        index = line.index('=')
                        typestring = line[index+1:].split(',')
                        print('Ignoring elements of type %s in input file' % typestring[0])
                        active = 'none'
                elif active == 'step' and newstep:
                    Steps.append({'INFO':[]})
                elif active == 'amplitude':
                    name = line.upper().strip('\n').strip('\r')[line.upper().index('NAME=')+5:].split(',')[0]
                elif active == 'boundary':
                    if 'AMPLITUDE' in line.upper():
                        amplitude = line.upper().split('AMPLITUDE=')[0].split(',')[0]
                    else:
                        amplitude = 'DEFAULT'
                        
            elif active == 'nodes':
                node_list.append([float(x) for x in line.split(',')[1:]])
            elif active == 'elems':
                elem_nodes.append([int(x)-1 for x in line.split(',')[1:]])
                elem_types.append(eltype)
            elif active == 'prop':
                # Note that this assumes uniform material properties
                if reset == True:
                    print("Warning, overriding material properties. Current "
                          "framework does not allow for different material "
                          "properties for different elements. Most recently "
                          "set properties will be used.")
                reset = True
                prop_list = line.strip('\n').strip('\r').split(',')
                lc = float(prop_list[0])
                Gc = float(prop_list[1])
                E =  float(prop_list[2])
                nu = float(prop_list[3])
                k =  float(prop_list[4])
                aniso = prop_list[6].strip(' ') not in ['0', 'false', 'False', 'f', 'F']
            elif active == 'nset':
                NSet[name] += [int(x)-1 for x in line.split(',')]
            elif active == 'gennset':
                rng = [int(x) for x in line.split(',')]
                NSet[name] += [x for x in range(rng[0]-1, rng[1], rng[2])]
            elif active == 'elset':
                ElSet[name] += [int(x)-1 for x in line.split(',')]
            elif active == 'genelset':
                rng = [int(x) for x in line.split(',')]
                ElSet[name] += [x for x in range(rng[0]-1, rng[1], rng[2])]
            elif active == 'step':
                pass
            elif active == 'boundary': 
                temp = line.split(',')
                Bound = {'Set':temp[0].upper()}
                Bound['DOF'] = ([int(x) for x in range(int(temp[1])-1,int(temp[1]))])
                Bound['AMP'] = amplitude
                if len(temp) > 3:
                    Bound['VAL'] = float(temp[-1])
                else:
                    Bound['VAL'] = 0.
                Steps[-1]['INFO'].append(Bound)
            elif active == 'amplitude':
                Amplitude[name] = np.array([float(x) for x in line.split(',')]).reshape(-1,2)
    
    Nodes = np.array(node_list)
    Elements = []
    for node, eltype in zip(elem_nodes, elem_types):
        Elements.append(Element(eltype, np.array(node), Nodes[node,:], Gc=Gc, 
                                lc=lc, E=E, nu=nu, k=k, aniso=aniso))
        
    return Nodes, Elements, Steps, Amplitude, NSet, ElSet

def Read_Matlab(filename):
    """Reads a Matlab optimization result
    
    Parameters
    ----------
    filename : string
        name of the file
        
    Returns
    -------
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
    """
    
    data = sio.loadmat(filename)
    
    Nodes = data['fem']['Node'][0][0]
    elem_nodes = data['fem']['Element'][0][0]
    elem_types = elem_nodes.shape[0]*['Q4']
    ElSet = {}
    F = data['fem']['F'][0][0]
    dim = Nodes.shape[1]
    try:
        FixDofs = np.setdiff1d(np.arange(F.size), data['fem']['Springlessdof'][0][0]-1)
    except:
        FixDofs = np.setdiff1d(np.arange(F.size), data['fem']['FreeDofs'][0][0]-1)
    NSet = {'TOP':F.nonzero()[0]//dim, 'BOT':FixDofs//dim}
    Steps = [{'INFO':[]}]
    Amplitude = {'DEFAULT':np.array([[0., 0.], [1., 1.]])}
    amplitude = 'DEFAULT'

    lc = 2*max(Nodes[1]-Nodes[0])
    Gc = 0.003
    E0 = float(data['fem']['E0'])
    nu = float(data['fem']['Nu0'])
    k = 1e-8
#    lc, Gc, E0, nu, k = 0.0075, 0.0027, 210.0, 0.3, 1.e-7
    aniso = True
    try:
        V = data['opt']['P'][0][0] * data['opt']['x'][0][0].ravel()
    except:
        V = data['opt']['P'][0][0] * data['opt']['x'][0][0].ravel()[data['opt']['Mirror'][0][0].ravel()-1]
    E = k + (1-k)*V**4
        
    Bound = {'Set':'TOP'}
    Bound['DOF'] = [stats.mode(F.nonzero()[0] % dim).mode[0]]
    Bound['AMP'] = amplitude
    Bound['VAL'] = 0.1*(Nodes[:,Bound['DOF']].max() - Nodes[:,Bound['DOF']].min())
    Steps[-1]['INFO'].append(Bound)
    
    Bound = {'Set':'BOT'}
    Bound['DOF'] = np.unique(FixDofs % dim)
    Bound['AMP'] = amplitude
    Bound['VAL'] = 0.
    Steps[-1]['INFO'].append(Bound)

    Elements = []
#    real_nodes = np.zeros(Nodes.shape[0], dtype=bool)
#    for node, eltype, stiff in zip(elem_nodes, elem_types, E):
#        if stiff > 1e-2:
#            real_nodes[node-1] = 1
#    real_nodes = real_nodes * np.cumsum(real_nodes)
#    Nodes = Nodes[real_nodes > 0]
#    for ns in NSet:
#        NSet[ns] = real_nodes[NSet[ns]]
#        NSet[ns] = NSet[ns][NSet[ns] > 0] - 1
    for node, eltype, stiff in zip(elem_nodes, elem_types, E):
#        stiff = (stiff > 1e-1)*(1-1e-10) + 1e-10
#        stiff = 1
        if stiff > -1e-2:
            Elements.append(Element(eltype, np.array(node-1), Nodes[node-1,:], Gc=Gc, 
                                    lc=lc, E=E0*stiff, nu=nu, k=k, aniso=aniso))
#            Elements.append(Element(eltype, real_nodes[np.array(node)-1]-1, Nodes[real_nodes[np.array(node)-1]-1,:], Gc=Gc, 
#                                    lc=lc, E=E0, nu=nu, k=k, aniso=aniso))
        
    return Nodes, Elements, Steps, Amplitude, NSet, ElSet