# -*- coding: utf-8 -*-
"""
Created on Wed May 24 08:52:17 2023

@author: urbiz
"""

import numpy as np
import functions as fn



path_data = "datos_AgClCuO/"

elements=[]
filename="elements"
for line in open(path_data+filename, 'r'):       
    value=line.split()
    if(value[0] != '#'):
        elements.append(value)

print("Elementos cargados!")






posit_ini = np.load('posit_ini.npy')

num_atoms=122

num_x=25
num_y=25
num_z=1

z_ini=0.56
z_end=0.8

X=np.linspace(0,1,num_x)
Y=np.linspace(0,1,num_y)
Z=np.linspace(z_ini,z_end,num_z)



posit_XYZ=[]
posit_gener_all=[]
for i in range(0,len(X)):
    for j in range(0,len(Y)):
        for k in range(0,len(Z)):
            print(i,X[i],j,Y[j],k,Z[k])
            posXYZ=X[i],Y[j],Z[k]
            posXYZ=np.array(posXYZ,dtype='float')
            posXYZ=np.reshape(posXYZ,(1,3))
            
            cut_A=posit_ini[0:120,:]
            cut_B=posit_ini[121:len(posit_ini[:,0]),:]
            posit_gener=np.concatenate((cut_A,posXYZ,cut_B))
            posit_gener_all.append(posit_gener)
            posit_XYZ.append(posXYZ)

posit_gener_all=np.array(posit_gener_all,dtype=float)
posit_XYZ=np.array(posit_XYZ,dtype=float)

print("Calculando charge_vectorsss...")
charge_vectorZ1 = fn.charg_vect_funcZ1(elements,num_atoms)
charge_vectorZ2 = fn.charg_vect_funcZ2(elements,num_atoms)
charge_vectorZ3 = fn.charg_vect_funcZ3(elements,num_atoms)
print("Calculando modulo...")
matrix_module = fn.module(posit_gener_all,num_atoms,"base")
print("Calculando matriz de coulomb...")
matrix_coulombZ1 = fn.coulomb(matrix_module,charge_vectorZ1,"base_Z1")
matrix_coulombZ2 = fn.coulomb(matrix_module,charge_vectorZ2,"base_Z2")
matrix_coulombZ3 = fn.coulomb(matrix_module,charge_vectorZ3,"base_Z3")

matrix_coulomb_bulkZ1=np.array(matrix_coulombZ1,dtype=float)
matrix_coulomb_bulkZ2=np.array(matrix_coulombZ2,dtype=float)
matrix_coulomb_bulkZ3=np.array(matrix_coulombZ3,dtype=float)

matrix_coulomb_Z1_norm = matrix_coulomb_bulkZ1 / np.max(matrix_coulomb_bulkZ1)
matrix_coulomb_Z2_norm = matrix_coulomb_bulkZ2 / np.max(matrix_coulomb_bulkZ2)
matrix_coulomb_Z3_norm = matrix_coulomb_bulkZ3 / np.max(matrix_coulomb_bulkZ3)

matrix_coulomb_Z1_norm = matrix_coulomb_Z1_norm.reshape(matrix_coulomb_Z1_norm.shape[0],matrix_coulomb_Z1_norm.shape[1],matrix_coulomb_Z1_norm.shape[2],1)
matrix_coulomb_Z2_norm = matrix_coulomb_Z2_norm.reshape(matrix_coulomb_Z2_norm.shape[0],matrix_coulomb_Z2_norm.shape[1],matrix_coulomb_Z2_norm.shape[2],1)
matrix_coulomb_Z3_norm = matrix_coulomb_Z3_norm.reshape(matrix_coulomb_Z3_norm.shape[0],matrix_coulomb_Z3_norm.shape[1],matrix_coulomb_Z3_norm.shape[2],1)
matrix = np.concatenate((matrix_coulomb_Z1_norm,matrix_coulomb_Z2_norm,matrix_coulomb_Z3_norm),axis=3)

np.save('matrix_posit_gener_01',matrix)