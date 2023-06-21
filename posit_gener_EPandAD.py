# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:18:58 2023

@author: urbiz
"""

import numpy as np
import functions as fn

def load_elements(path_data, filename):
    elements = []
    with open(path_data + filename, 'r') as file:
        for line in file:
            value = line.split()
            if value[0] != '#':
                elements.append(value)
    return elements

def load_energies(path_data, filename):
    energies = []
    with open(path_data + filename, 'r') as file:
        for line in file:
            value = line.split()
            if value[0] != '#':
                energies.append(value)
    return energies

def load_positions(path_data, energies):
    pos_atom = []
    merged_pos_atom = []

    for n in range(1, len(energies) + 1):
        filename = "positions_" + str(n)
        with open(path_data + filename, 'r') as file:
            for line in file:
                value = line.split()
                if value[0] != '#':
                    pos_atom.append(value)
        merged_pos_atom.append(pos_atom[:])
        pos_atom = []

    return merged_pos_atom


print("-----------------------------------")
print("COLLECTING DATA ...")
print("====================================")

path_data = "data_AD_plusNNposit/"

elements = load_elements(path_data, "elements")
print("Elementos cargados!")
energies = load_energies(path_data, "Energies")
print("Energias cargadas!")
merged_pos_atom = load_positions(path_data, energies)
print("Posiciones cargadas!")
print("====================================")


merged_pos_atom_bulk = np.array(merged_pos_atom, dtype=float)


num_atoms=129

num_x=50
num_y=50
num_z=3

z_ini=0.56
z_end=0.8

X=np.linspace(0,1,num_x)
Y=np.linspace(0,1,num_y)
Z=np.linspace(z_ini,z_end,num_z)

posit_XYZ=[]
posit_gener_all=[]
for k in range(0,len(Z)):
    for i in range(0,len(X)):
        for j in range(0,len(Y)):

            print(i,X[i],j,Y[j],k,Z[k])
            posXYZ=X[i],Y[j],Z[k]
            posXYZ=np.array(posXYZ,dtype='float')
            posXYZ=np.reshape(posXYZ,(1,3))
            
            cut_A=merged_pos_atom_bulk[0,0:121,:]
            cut_B=merged_pos_atom_bulk[0,122:len(merged_pos_atom_bulk[0,:,0]),:]
            posit_gener=np.concatenate((cut_A,posXYZ,cut_B))
            posit_gener_all.append(posit_gener)
            posit_XYZ.append(posXYZ)

posit_gener_all=np.array(posit_gener_all,dtype=float)
posit_XYZ=np.array(posit_XYZ,dtype=float)


charge_vector = fn.charg_vect_funcZ1(elements,num_atoms)
matrix_module = fn.module(posit_gener_all,num_atoms,"base")
matrix_coulomb = fn.coulomb(matrix_module,charge_vector,"base")
matrix_coulomb_bulk=np.array(matrix_coulomb,dtype=float)

matriz_norm_0=matrix_coulomb_bulk/np.max(matrix_coulomb_bulk)
matriz_norm_1_1=fn.norm2(matrix_coulomb_bulk,0.95,0.1,num_atoms)
matriz_norm_1_2=fn.norm2(matrix_coulomb_bulk,0.97,0.2,num_atoms)

matriz_norm_0=matriz_norm_0.reshape(matriz_norm_0.shape[0],matriz_norm_0.shape[1],matriz_norm_0.shape[2],1)
matriz_norm_1_1=matriz_norm_1_1.reshape(matriz_norm_1_1.shape[0],matriz_norm_1_1.shape[1],matriz_norm_1_1.shape[2],1)
matriz_norm_1_2=matriz_norm_1_2.reshape(matriz_norm_1_2.shape[0],matriz_norm_1_2.shape[1],matriz_norm_1_2.shape[2],1)
matrix = np.concatenate((matriz_norm_0,matriz_norm_1_1,matriz_norm_1_2),axis=3)

np.save('matrix_AD.npy',matrix)
np.save('positXYZ.npy',posit_XYZ)
