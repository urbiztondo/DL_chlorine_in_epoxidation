# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:34:25 2021

@author: urbiz
"""

###################################
###   IMPORTS  ####################
###################################

import functions as fn
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import datetime
import time
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
import shutil
import inspect
import statsmodels.api as sm
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2

###################################
###########  VARS #################
###################################

start_time = time.time()

print("-----------------------------------")
print("-----------------------------------")
print("DATOS DE ENTRADA...")
print("-----------------------------------")

path_data = "datos_AgClCuO/"

id_matriz = 2 #0- matriz 2D, 1-matriz3D con todos los id_zs, 3-matriz3D con normalizacion por partes en cada capa
id_calcu = 0 # 0- no se calculan, 1- se calculan
id_Z = 1 # calculo matriz de coulomb 1-Z pequenos, 2-Zgrandes, 3-peso a mas bajos
id_reducc =0 # 0-no se reduce, 1-se reduce a los atomos mas relevantes
id_flatten = 0 # 0-no se aplana la matriz, 1-se aplana para meter en vector

# Sobre las normalizaciones
norm_ener_type = 0 # 0-de 0a1, 1- por partes, 2-media/desvest, 4-concatenada
norm_coulomb = 1 # 1-promedio, 2-por partes

#Parametros de normalizacion para cada tipo de matriz si va por partes
if (id_matriz==0):
    y1_norm_ener = 0.7
    perc_norm_ener = 0.3
if (id_matriz==1):
    y1_norm_ener = 0.7
    perc_norm_ener = 0.3
if (id_matriz==2):
    y1_norm_ener = 0.5
    perc_norm_ener = 0.3
    

num_atoms = 122
frac_train=0.8
frac_val=0.1


epochs = 5000
BATCH_SIZE = 16

NN_type=14

verbose=2


pathresults="NN14_def_norm_energ/"



for norm_ener_type in [0,1,2]:

    
        
    path_exper=pathresults+"norm_energ_"+str(norm_ener_type)+"/"
    
    # Fijando semilla para obtener resultados mas estables en tf
    seed = 1
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
    
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("PROBANDO CON "+str(epochs)+" EPOCHS")    
    
    print("PATH: ",path_data)
    print("id_matriz: ", id_matriz)
    print("tipo de normalizacion de energia: ", norm_ener_type)
    print("valor de y1 para normalizacion: ", y1_norm_ener)
    print("valor de perc para normalizacion: ", perc_norm_ener)
    print("Normalización del coulomb: ", norm_coulomb)
    print("Numero de atomos: ",num_atoms) 
    print("frac train: ",frac_train) 
    print("frac val: ",frac_val) 
    print("epochs: ",epochs) 
    print("Batch size: ", BATCH_SIZE)
    
    
    print("====================================")
    print("Tipo de matriz de coulomb")
    
    if (id_matriz==0):
        print("matriz 2d")
        if(id_flatten == 1):
            print("matriz flateada")
        print("que tipo de Z quieres?")
        if (id_Z==1):
            print("matriz de 1d con Z1")
        if (id_Z==2):
            print("matriz de 1d con Z2")
        if (id_Z==3):
            print("matriz de 1d con Z3")
    if (id_matriz==1):
        print("matriz 3d recopilando todos los Zs")
    if (id_matriz==2):
        print("matriz 3d normalizando por partes")
    if(id_reducc==0):
        print("Van todos los atomos")
    if(id_reducc==1):
        print("Aplica la reduccion a los atomos")
    print("====================================")
    print("SE HACEN CALCULOS NUEVOS???")
    
    if (id_calcu==0):
        print("No se solicita el calculo // carga de datos")
    if (id_calcu==1):
        print("A calcular... puede tomar bastante tiempo")
    
    print("-----------------------------------")
    print("-----------------------------------")
    print("-----------------------------------")
    
    
    #########################################
    ### COLLECTING DATA  ####################
    #########################################
    
    print("-----------------------------------")
    print("COLLECTING DATA ...")
    print("====================================")
    
    
    elements=[]
    filename="elements"
    for line in open(path_data+filename, 'r'):       
        value=line.split()
        if(value[0] != '#'):
            elements.append(value)
    
    print("Elementos cargados!")
    
    energies=[]
    filename="Energies"
    
    for line in open(path_data+filename, 'r'):       
        value=line.split()
        val=np.array(value,dtype='float')           
        if(value[0] != '#'):
            energies.append(value)
    
    print("Energias cargadas!")
    
    pos_atom=[]
    merged_pos_atom=[]
    
    for n in range(1,len(energies)+1):
        filename="positions_"+str(n)
        for line in open(path_data+filename, 'r'):       
            value=line.split()
            val=np.array(value,dtype='float')           
            if(value[0] != '#'):
                pos_atom.append(value)
        merged_pos_atom.append(pos_atom[:])
        pos_atom = [] 
           
    print("Posiciones cargadas!")
    print("====================================")
    
    
    
    #########################################
    ### CALCULO COULOMB   ##
    #########################################
    
    merged_pos_atom_bulk=np.array(merged_pos_atom,dtype=float)
    
    # PARA LA MATRIZ 2D
    if (id_matriz==0):
        if (id_calcu==1):
            print("Calculando charge_vector...")
            if (id_Z==1):
                charge_vector = fn.charg_vect_funcZ1(elements,num_atoms)
            if (id_Z==2):
                charge_vector = fn.charg_vect_funcZ2(elements,num_atoms)
            if (id_Z==3):
                charge_vector = fn.charg_vect_funcZ3(elements,num_atoms)
            print("Calculando modulo...")
            matrix_module = fn.module(merged_pos_atom_bulk,num_atoms,"base")        
            print("Calculando matriz de coulomb...")
            matrix_coulomb = fn.coulomb(matrix_module,charge_vector,"base")    
    
        if (id_calcu==0):
            print("loading external data:  ...  matrix_coulomb")
         
            with open ('coulomb_base.dat','r') as filehandle:
                matrix_coulomb=json.load(filehandle)
                print("matrix coulomb have been loaded!")    
               
            with open ('modulo_base.dat','r') as filehandle:
                matrix_module=json.load(filehandle)
                print("matrix module low have been loaded!")
    
        # Convirtiendo en matrices
        matrix_coulomb_bulk=np.array(matrix_coulomb,dtype=float)
          
        # Normalizando
        if (norm_coulomb == 1): 
            matrix_coulomb_norm = matrix_coulomb_bulk / np.max(matrix_coulomb_bulk)
            
            
            
        if (norm_coulomb == 2):
            maximo = np.max(matrix_coulomb_bulk)
            minimo = np.min(matrix_coulomb_bulk)
            percenta = 0.95
            y1_coulomb = 0.2
            x1_coulomb = maximo + (minimo - maximo)*percenta
            
            matrix_coulomb_norm = []
            for i in range(0,len(matrix_coulomb_bulk)):
                vector = matrix_coulomb_bulk[i].flatten()
                vector_norm = fn.energ_norm_vector(percenta,y1_coulomb,minimo,maximo,x1_coulomb,vector)
                matrix_uni = vector_norm.reshape(num_atoms,num_atoms)
                matrix_coulomb_norm.append(matrix_uni)
            matrix_coulomb_norm=np.array(matrix_coulomb_norm,dtype=float)
            
            
            
    
        if (id_reducc==1):
        # Reduciendo los atomos que menos aportan  
            atoms = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                     13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                     26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                     39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                     52,  53,  54,  55,  56,  57,  58,  59, 81, 82, 98, 101, 104, 106, 108,
                     113, 115, 117, 120, 121]
            
            
            matrix_coulomb_norm_cutted = fn.coulomb_short(matrix_coulomb_norm, atoms)    
            matrix = matrix_coulomb_norm_cutted.reshape(matrix_coulomb_norm_cutted.shape[0],matrix_coulomb_norm_cutted.shape[1],matrix_coulomb_norm_cutted.shape[2],1)
        if(id_reducc==0):
            matrix = matrix_coulomb_norm.reshape(matrix_coulomb_norm.shape[0],matrix_coulomb_norm.shape[1],matrix_coulomb_norm.shape[2],1)
    
        if(id_flatten==1):
            matrix_flatted = []
            for i in range(0,len(matrix)):
            #for i in range(0,200):            
                matrix_element = matrix[i].flatten()
                matrix_flatted.append(matrix_element)
            matrix = matrix_flatted
    
    
    # MATRIZ 3D CON TODOS LOS ZS
    if (id_matriz==1):
    
        
        ### charge vector ###########################
        #############################################
        if (id_calcu==1):       
            print("Calculando charge_vectorsss...")
            charge_vectorZ1 = fn.charg_vect_funcZ1(elements,num_atoms)
            charge_vectorZ2 = fn.charg_vect_funcZ2(elements,num_atoms)
            charge_vectorZ3 = fn.charg_vect_funcZ3(elements,num_atoms)
            
        ### module matrix  ##########################
        #############################################
        
            print("Calculando modulo...")
            matrix_module = fn.module(merged_pos_atom_bulk,num_atoms,"base")
        
        ### coulomb matrix ##########################
        ############################################# 
        
            print("Calculando matriz de coulomb...")
            matrix_coulombZ1 = fn.coulomb(matrix_module,charge_vectorZ1,"base_Z1")
            matrix_coulombZ2 = fn.coulomb(matrix_module,charge_vectorZ2,"base_Z2")
            matrix_coulombZ3 = fn.coulomb(matrix_module,charge_vectorZ3,"base_Z3")
        
        if (id_calcu==0):
            print("loading external data:  ...  matrix_coulomb")
         
            with open ('coulomb_base_Z1.dat','r') as filehandle:
                matrix_coulombZ1=json.load(filehandle)
                print("matrix coulomb Z1 have been loaded!")
            
            with open ('coulomb_base_Z2.dat','r') as filehandle:
                matrix_coulombZ2=json.load(filehandle)
                print("matrix coulomb Z2 have been loaded!")
        
            with open ('coulomb_base_Z3.dat','r') as filehandle:
                matrix_coulombZ3=json.load(filehandle)
                print("matrix coulomb Z3 have been loaded!")
                
            with open ('modulo_base.dat','r') as filehandle:
                matrix_module=json.load(filehandle)
                print("matrix module have been loaded!")
    
    
    
        # Convirtiendo en matrices
        
        matrix_coulomb_bulkZ1=np.array(matrix_coulombZ1,dtype=float)
        matrix_coulomb_bulkZ2=np.array(matrix_coulombZ2,dtype=float)
        matrix_coulomb_bulkZ3=np.array(matrix_coulombZ3,dtype=float)
        
        
        # Normalizando
        if (norm_coulomb == 1): 
            matrix_coulomb_Z1_norm = matrix_coulomb_bulkZ1 / np.max(matrix_coulomb_bulkZ1)
            matrix_coulomb_Z2_norm = matrix_coulomb_bulkZ2 / np.max(matrix_coulomb_bulkZ2)
            matrix_coulomb_Z3_norm = matrix_coulomb_bulkZ3 / np.max(matrix_coulomb_bulkZ3)
        
        
        if (id_reducc==1):
        # Reduciendo los atomos que menos aportan  
            atoms = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                     13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                     26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                     39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                     52,  53,  54,  55,  56,  57,  58,  59, 120, 121]
            
            
            matrix_coulomb_Z1_norm_cutted = fn.coulomb_short(matrix_coulomb_Z1_norm, atoms)
            matrix_coulomb_Z2_norm_cutted = fn.coulomb_short(matrix_coulomb_Z2_norm, atoms)
            matrix_coulomb_Z3_norm_cutted = fn.coulomb_short(matrix_coulomb_Z3_norm, atoms)
            
            # Redimensionando y concatenando
            matrix_coulomb_Z1_norm_cutted = matrix_coulomb_Z1_norm_cutted.reshape(matrix_coulomb_Z1_norm_cutted.shape[0],matrix_coulomb_Z1_norm_cutted.shape[1],matrix_coulomb_Z1_norm_cutted.shape[2],1)
            matrix_coulomb_Z2_norm_cutted = matrix_coulomb_Z2_norm_cutted.reshape(matrix_coulomb_Z2_norm_cutted.shape[0],matrix_coulomb_Z2_norm_cutted.shape[1],matrix_coulomb_Z2_norm_cutted.shape[2],1)
            matrix_coulomb_Z3_norm_cutted = matrix_coulomb_Z3_norm_cutted.reshape(matrix_coulomb_Z3_norm_cutted.shape[0],matrix_coulomb_Z3_norm_cutted.shape[1],matrix_coulomb_Z3_norm_cutted.shape[2],1)
            matrix = np.concatenate((matrix_coulomb_Z1_norm_cutted,matrix_coulomb_Z2_norm_cutted,matrix_coulomb_Z3_norm_cutted),axis=3)
    
        if(id_reducc==0):
            matrix_coulomb_Z1_norm = matrix_coulomb_Z1_norm.reshape(matrix_coulomb_Z1_norm.shape[0],matrix_coulomb_Z1_norm.shape[1],matrix_coulomb_Z1_norm.shape[2],1)
            matrix_coulomb_Z2_norm = matrix_coulomb_Z2_norm.reshape(matrix_coulomb_Z2_norm.shape[0],matrix_coulomb_Z2_norm.shape[1],matrix_coulomb_Z2_norm.shape[2],1)
            matrix_coulomb_Z3_norm = matrix_coulomb_Z3_norm.reshape(matrix_coulomb_Z3_norm.shape[0],matrix_coulomb_Z3_norm.shape[1],matrix_coulomb_Z3_norm.shape[2],1)
            matrix = np.concatenate((matrix_coulomb_Z1_norm,matrix_coulomb_Z2_norm,matrix_coulomb_Z3_norm),axis=3)
     
    
    # MATRIZ3D NORMALIZANDO POR PARTES TODOS LOS ZS
    
    if (id_matriz==2):
        if (id_calcu==1):
            print("Calculando charge_vector...")
            if (id_Z==1):
                charge_vector = fn.charg_vect_funcZ1(elements,num_atoms)
            if (id_Z==2):
                charge_vector = fn.charg_vect_funcZ2(elements,num_atoms)
            if (id_Z==3):
                charge_vector = fn.charg_vect_funcZ3(elements,num_atoms)
            print("Calculando modulo...")
            matrix_module = fn.module(merged_pos_atom_bulk,num_atoms,"base")        
            print("Calculando matriz de coulomb...")
            matrix_coulomb = fn.coulomb(matrix_module,charge_vector,"base")    
    
        if (id_calcu==0):
            print("loading external data:  ...  matrix_coulomb")
         
            with open ('coulomb_base.dat','r') as filehandle:
                matrix_coulomb=json.load(filehandle)
                print("matrix coulomb have been loaded!")    
               
            with open ('modulo_base.dat','r') as filehandle:
                matrix_module=json.load(filehandle)
                print("matrix module low have been loaded!")
    
        # Convirtiendo en matrices
        matrix_coulomb_bulk=np.array(matrix_coulomb,dtype=float)
        
        matriz_norm_0=matrix_coulomb_bulk/np.max(matrix_coulomb_bulk)
        matriz_norm_1_1=fn.norm2(matrix_coulomb_bulk,0.95,0.1,num_atoms)
        matriz_norm_1_2=fn.norm2(matrix_coulomb_bulk,0.97,0.2,num_atoms)
        
        matriz_norm_0=matriz_norm_0.reshape(matriz_norm_0.shape[0],matriz_norm_0.shape[1],matriz_norm_0.shape[2],1)
        matriz_norm_1_1=matriz_norm_1_1.reshape(matriz_norm_1_1.shape[0],matriz_norm_1_1.shape[1],matriz_norm_1_1.shape[2],1)
        matriz_norm_1_2=matriz_norm_1_2.reshape(matriz_norm_1_2.shape[0],matriz_norm_1_2.shape[1],matriz_norm_1_2.shape[2],1)
        matrix = np.concatenate((matriz_norm_0,matriz_norm_1_1,matriz_norm_1_2),axis=3)
    
    
    
    
    
    
    # Comprobando necesidad de calculo nuevo de matrices de coulomb
    
    if(id_matriz==0):
        if (len(energies) != len(matrix_coulomb)):
            print("************************************************************************")
            print("ERROR!!!!! NECESITAS EJECUTAR EL CALCULO PORQUE NO COINCIDEN LOS VALORES")
            print("Solicitas para el calculo: "+str(len(energies))+" tienes en la matriz de coulomb: "+str(len(matrix_coulombZ1)))
            sys.exit(0)
            
    if(id_matriz==1):
        if (len(energies) != len(matrix_coulombZ1)):
            print("************************************************************************")
            print("ERROR!!!!! NECESITAS EJECUTAR EL CALCULO PORQUE NO COINCIDEN LOS VALORES")
            print("Solicitas para el calculo: "+str(len(energies))+" tienes en la matriz de coulomb: "+str(len(matrix_coulombZ1)))
            sys.exit(0)
    
    if(id_matriz==2):
        if (len(energies) != len(matriz_norm_0)):
            print("************************************************************************")
            print("ERROR!!!!! NECESITAS EJECUTAR EL CALCULO PORQUE NO COINCIDEN LOS VALORES")
            print("Solicitas para el calculo: "+str(len(energies))+" tienes en la matriz de coulomb: "+str(len(matriz_norm_0)))
            sys.exit(0)
    
    
    
    
    
    
    #########################################
    ### NORMALIZANDO ENERGIA  #################
    #########################################
    
    energies_bulk=np.array(energies,dtype=float)
    
    
    
    print("Normalizing energies!!!")
    energies_bulk = energies_bulk[:,1]
    
    
    if (norm_ener_type == 0):
        energ_0a1=1-((energies_bulk-np.min(energies_bulk))/(np.max(energies_bulk)-np.min(energies_bulk)))
        energ_0a1=np.reshape(energ_0a1,(len(energ_0a1),1))
        energies = energ_0a1
    
    if (norm_ener_type == 1):
    
    #Esto es para normalizar con el 0-0,1
        #perc_norm_ener = 0.79
        #y1_norm_ener = 0.28
        energ_norm,x1,y1,max,min = fn.energ_norm(perc_norm_ener,y1_norm_ener,energies_bulk)
        energ_norm = np.reshape(energ_norm,(len(energ_norm),1))
        energies = energ_norm
    
    if (norm_ener_type == 2):
        mean = np.mean(energies_bulk)
        stddev = np.std(energies_bulk)
        energies = (energies_bulk - mean) / stddev
        
    if (norm_ener_type == 3):
    
    #Esto es para normalizar con el 0-0,1
        energ_norm,x1,y1,max,min = fn.energ_norm(0.88,0.1,energies_bulk)
        #energ_norm = np.reshape(energ_norm,(len(energ_norm),1))
        energ_0a1=1-((energies_bulk-np.min(energies_bulk))/(np.max(energies_bulk)-np.min(energies_bulk)))
        energ_sum = energ_norm +energ_0a1
        energies = energ_sum/np.max(energ_sum)
        energies = energies.reshape(energies.shape[0],1)
    
    if (norm_ener_type == 4):
        energ_norm005,x1,y1,max,min = fn.energ_norm(0.75,0.05,energies_bulk)
        energ_norm03,x1,y1,max,min = fn.energ_norm(0.85,0.05,energies_bulk)
        energ_norm09,x1,y1,max,min = fn.energ_norm(0.95,0.05,energies_bulk)
        energies=np.concatenate((energ_norm005.reshape(energ_norm005.shape[0],1),energ_norm03.reshape(energ_norm03.shape[0],1),energ_norm09.reshape(energ_norm09.shape[0],1)),axis=1)
        
       
    
    #########################################
    ### mezclado   ##
    #########################################
    
    orden = np.linspace(0,len(energies_bulk)-1,len(energies_bulk)).reshape(len(energies_bulk),1)
       
    
    print("Collecting all data and shuffle and unzip")
    merged_all=list(zip(matrix,merged_pos_atom,energies,orden))
    random.shuffle(merged_all)
    matrix,merged_pos_atom,energies,orden=zip(*merged_all)
    
    ##### Pasando todo a np array ##########
    energies = np.array(energies,dtype=float)
    matrix=np.array(matrix,dtype=float)
    posit_atoms=np.array(merged_pos_atom,dtype=float)
    
    
    
    
    
    matrix_inlet = matrix
    #matrix_inlet = matrix_coulomb_Z2_norm
    
    
    
    
    
    """ *************************************** """
    """ *************************************** """
    """ *************************************** """   
    
    print ("..........")
    print ("..........")
    print ("..........")
    print ("entrando en la preparacion de vectores de entrada...")
    
    
    
    frac_test=1-frac_train-frac_val
    
    
    itrain=0;ftrain=int(len(matrix_inlet)*frac_train)
    ival=ftrain;fval=int(len(matrix_inlet)*(frac_train+frac_val))
    itest=fval;ftest=len(matrix_inlet)
    
    x_train = matrix_inlet[itrain:ftrain]
    y_train = energies[itrain:ftrain]
    x_val   = matrix_inlet[ival:fval]
    y_val   = energies[ival:fval]
    x_test  = matrix_inlet[itest:ftest]
    y_test  = energies[itest:ftest]
    orden_train = orden[itrain:ftrain]
    orden_test = orden[itest:ftest]
    orden_val = orden[ival:fval]
    
    if (id_flatten == 1):
        x_train=np.reshape(x_train,(len(x_train),len(x_train[0]),1))
        x_val=np.reshape(x_val,(len(x_val),len(x_val[0]),1))
        x_test=np.reshape(x_test,(len(x_test),len(x_test[0]),1))
    
    y_train=np.reshape(y_train,(len(y_train),1))
    y_val=np.reshape(y_val,(len(y_val),1))
    y_test=np.reshape(y_test,(len(y_test),1))
    
    if id_matriz==1:
        print("x_train: "+str(x_train.shape[0])+" "+str(x_train.shape[1])+" "+str(x_train.shape[2])+" "+str(x_train.shape[3]))
        print("x_val: "+str(x_val.shape[0])+" "+str(x_val.shape[1])+" "+str(x_val.shape[2])+" "+str(x_val.shape[3]))
        print("x_test: "+str(x_test.shape[0])+" "+str(x_test.shape[1])+" "+str(x_test.shape[2])+" "+str(x_test.shape[3]))
    
    if id_matriz==0:
        
        if (id_flatten == 1):
    
            print("x_train: "+str(x_train.shape[0])+" "+str(x_train.shape[1])+" "+str(x_train.shape[2]))
            print("x_val: "+str(x_val.shape[0])+" "+str(x_val.shape[1])+" "+str(x_val.shape[2]))
            print("x_test: "+str(x_test.shape[0])+" "+str(x_test.shape[1])+" "+str(x_test.shape[2]))
    
        #x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
        #x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
        #x_val=x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
        else:
            print("x_train: "+str(x_train.shape[0])+" "+str(x_train.shape[1])+" "+str(x_train.shape[2])+" "+str(x_train.shape[3]))
            print("x_val: "+str(x_val.shape[0])+" "+str(x_val.shape[1])+" "+str(x_val.shape[2])+" "+str(x_val.shape[3]))
            print("x_test: "+str(x_test.shape[0])+" "+str(x_test.shape[1])+" "+str(x_test.shape[2])+" "+str(x_test.shape[3]))
    
    
    print("y_train: "+str(y_train.shape[0])+" "+str(y_train.shape[1]))
    print("y_val: "+str(y_val.shape[0])+" "+str(y_val.shape[1]))
    print("y_test: "+str(y_test.shape[0])+" "+str(y_test.shape[1]))
    
    
    end_time = time.time()
    print("Tiempo para carga/normalizacion de datos de entreno: ", end_time-start_time, " segundos")
    
    
    start_time_1 = time.time()
    
    
    
    print ("..........")
    print ("..........")
    print ("..........") 
    print("entrando en el nucleo de la red neuronal")
    
    
    
    
       
    if (NN_type==14):
        model = Sequential()
        model.add(Conv2D(64, (3,3), strides=(1,1),
                         activation='selu',
                         input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]),
                         kernel_initializer='HeUniform',
                         bias_initializer='zeros'))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(128, (3,3), strides=(1,1),
                         activation='selu',
                         kernel_initializer='HeUniform',
                         bias_initializer='zeros'))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(256, (3,3), strides=(1,1),
                         activation='selu',
                         kernel_initializer='HeUniform',
                         bias_initializer='zeros'))
        #model.add(MaxPooling2D(2,2))
        model.add(Flatten())
        model.add(Dense(256, activation='sigmoid',
                        kernel_initializer='he_normal',
                        bias_initializer='zeros',
                        kernel_regularizer=regularizers.L1L2(l1=1e-8, l2=1e-6)))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='sigmoid',
                        kernel_initializer='he_normal',
                        bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid',
                        kernel_initializer='he_normal',
                        bias_initializer='zeros'))   
     
         
    
    model = tf.keras.models.clone_model(model)        
        
    model.summary()
    
    
    def epoch_lr(epoch):
        if (epoch<1000):
            lr = 9.9e-7*epoch+1e-5
        if (epoch>=1000 and epoch<2000):
            lr = -9.9e-7*epoch+1.99e-3
        if (epoch>=2000 and epoch<3000):
            lr = 9.9e-7*epoch-1.97e-3
        if (epoch>=3000 and epoch<4000):
            lr = -9.9e-7*epoch+3.97e-3   
        if (epoch>=4000):
            lr = 9.9e-7*epoch-3.95e-3
        return lr
    
    
    
    #lr_schedule=tf.keras.callbacks.LearningRateScheduler(
    #    lambda epoch: epoch_lr(epoch))
    
    opt = tf.keras.optimizers.Nadam(
        learning_rate=0.00001,
        beta_1=0.99,
        beta_2=0.999,
        epsilon=1e-08,
        name='Nadam')
    
    #opt=tf.keras.optimizers.SGD(momentum=0.9)
    
    model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=['mae'])
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test,y_test),
        #callbacks=[lr_schedule],
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=verbose,
        shuffle=True)
    
    
    
    
      
    history_dict=history.history
    training_cost=np.log(history_dict['loss'])
    evaluation_cost=np.log(history_dict['val_loss'])
    training_accuracy=np.log(history_dict['mae'])
    evaluation_accuracy=np.log(history_dict['val_mae'])
    
    print("FINAL SCORE:")
    print("training_cost=",training_cost[-1])
    print("evaluation_cost=",evaluation_cost[-1])
    print("training_accuracy=",training_accuracy[-1])
    print("evaluation_accuracy=",evaluation_accuracy[-1])
    
    
    
    print(".............................................................") 
    print("......*******GRAPHING****************...")
    print(".............................................................")
    
    # creamos los directorios de trabajo
    
    
    isfile=os.path.exists(pathresults)    
    if (not(isfile)):  
        os.mkdir(pathresults)
        
        
    
    isfile=os.path.exists(path_exper)
    if (not(isfile)):  
        os.mkdir(path_exper)
    
    
    ###############################
    #### GRAFICA CON 4 EN 1  DATOS DE AJUSTE EN EPOCHS######
    ###############################
    
    
    xx = np.linspace(0,epochs-1,epochs)
    training_MAE = training_accuracy.reshape(len(training_accuracy),1)
    xx_MAE = xx.reshape(len(xx),1)
    training_to_der = np.concatenate((xx_MAE,training_MAE),axis=1)
    train_dev=fn.deriv(training_to_der)
    train_dev = np.array(train_dev)
    
    val_MAE = evaluation_accuracy.reshape(len(evaluation_accuracy),1)
    val_to_der = np.concatenate((xx_MAE,val_MAE),axis=1)
    val_dev = fn.deriv(val_to_der)
    val_dev = np.array(val_dev)
    
    training_accuracy=np.log(history_dict['mae'])
    evaluation_accuracy=np.log(history_dict['val_mae'])
    
    fig2, ax2 = plt.subplots(2,2, figsize=(10,10)) #sharex='col', sharey='row',
    ax2[0,0].plot(xx,evaluation_cost, color="orange", label="evaluation cost (log)")
    ax2[0,0].plot(xx,training_cost, color="blue", label="training cost (log)")
    ax2[0,1].plot(xx,evaluation_accuracy, color="orange", label="evaluation MAE (log)")
    ax2[0,1].plot(xx,training_accuracy, color="blue", label="training MAE (log)")
    ax2[1,0].plot(train_dev[:,0],train_dev[:,1],color="blue", label="pendiente error training",fillstyle='none',linewidth=0, marker='.', markersize=10)
    ax2[1,0].plot(val_dev[:,0],val_dev[:,1],color="orange", label="pendiente error valida",fillstyle='none',linewidth=0, marker='.', markersize=10)
    ax2[1,1].plot(training_accuracy,np.log((10**training_accuracy-10**evaluation_accuracy)**2),color="blue", label="diference",fillstyle='none',linewidth=0, marker='.', markersize=10)
    txt = "$epoch$"
    ax2[0,0].set_xlabel(txt)
    ax2[0,0].legend()
    ax2[0,1].set_xlabel(txt)
    ax2[1,0].legend()
    ax2[1,0].set_xlabel(txt)
    ax2[0,1].legend()
    ax2[1,1].set_xlabel("training_accuracy MAE")
    ax2[1,1].legend()
    filename=str(path_exper)+"/"+"im_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".png"  
    fig2.savefig(filename, dpi=400, facecolor="#f1f1f1")
    
    
    ###############################
    #### GRAFICA 4 en 1 de energias pred y real y y=y con r2  ######
    ###############################
    
    #Calculo de los valores de y en la red   
    y_pred_val = model.predict(x_val)     
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    
    
    def rsquared(y_pred,y):
        X = sm.add_constant(y_pred)
        mod = sm.OLS(y,X).fit()
        R2 = mod.rsquared_adj
        return R2
    
    if norm_ener_type ==1:
        y_pred_val = fn.norm_energ (y_pred_val,y1,x1,max,min)
        y_pred_test = fn.norm_energ (y_pred_test,y1,x1,max,min)
        y_pred_train = fn.norm_energ (y_pred_train,y1,x1,max,min)
        
        y_val = fn.norm_energ (y_val,y1,x1,max,min)
        y_test = fn.norm_energ (y_test,y1,x1,max,min)
        y_train = fn.norm_energ (y_train,y1,x1,max,min)
    
    
    
    R2_train = rsquared(y_pred_train,y_train)
    R2_val = rsquared(y_pred_val,y_val)
    R2_test = rsquared(y_pred_test,y_test)
    
    fig3,ax3=plt.subplots(2,2, figsize=(10,10))
    ax3[0,0].plot(orden_val,y_val,color='blue',label="Actual energy",fillstyle='none',linewidth=0, marker='.', markersize=15)
    ax3[0,0].plot(orden_val,y_pred_val,color='red',label="NN predicted",linewidth=0, marker='.', markersize=10)
    ax3[0,1].plot(orden_test,y_test,color='blue',label="Actual energy",fillstyle='none',linewidth=0, marker='.', markersize=15)
    ax3[0,1].plot(orden_test,y_pred_test,color='red',label="NN predicted",linewidth=0, marker='.', markersize=10) 
    ax3[1,0].plot(orden_train,y_train,color='blue',label="Actual energy",fillstyle='none',linewidth=0, marker='.', markersize=15)
    ax3[1,0].plot(orden_train,y_pred_train,color='red',label="NN predicted",linewidth=0, marker='.', markersize=10) 
    ax3[1,1].plot(y_pred_train,y_train,color='blue',label="R2 train: "+str(R2_train),fillstyle='none',linewidth=0, marker='.', markersize=5)
    ax3[1,1].plot(y_pred_val,y_val,color='red',label="R2 val: "+str(R2_val),fillstyle='none',linewidth=0, marker='.', markersize=5)
    ax3[1,1].plot(y_pred_test,y_test,color='green',label="R2 test: "+str(R2_test),fillstyle='none',linewidth=0, marker='.', markersize=5)
    
    ax3[0,0].set_xlabel("epoch")
    ax3[0,0].set_ylabel("energy validation")
    ax3[0,0].legend()
    ax3[0,1].set_xlabel("epoch")
    ax3[0,1].set_ylabel("energy test")
    ax3[0,1].legend()
    ax3[1,0].set_xlabel("epoch")
    ax3[1,0].set_ylabel("energy training")
    ax3[1,0].legend()
    ax3[1,1].set_xlabel("energy")
    ax3[1,1].set_ylabel("energy")
    ax3[1,1].legend()
    
    filename=str(path_exper)+"/estimations.png"  
    fig3.savefig(filename, dpi=400, facecolor="#f1f1f1")
    
    
    
    
    
    
    
    dif_train=100*(np.abs(y_train-y_pred_train))/y_train
    dif_test=100*(np.abs(y_test-y_pred_test))/y_test
    dif_val=100*(np.abs(y_val-y_pred_val))/y_val
    
    
    fig4,ax4=plt.subplots(2,2, figsize=(10,10))
    ax4[0,0].plot(orden_val,dif_val,color='blue',label="Actual energy",fillstyle='none',linewidth=0, marker='.', markersize=15)
    ax4[0,1].plot(orden_test,dif_test,color='blue',label="Actual energy",fillstyle='none',linewidth=0, marker='.', markersize=15)
    ax4[1,0].plot(orden_train,dif_train,color='blue',label="Actual energy",fillstyle='none',linewidth=0, marker='.', markersize=15)
    ax4[1,1].plot(y_pred_train,y_train,color='blue',label="R2 train: "+str(R2_train),fillstyle='none',linewidth=0, marker='.', markersize=5)
    ax4[1,1].plot(y_pred_val,y_val,color='red',label="R2 val: "+str(R2_val),fillstyle='none',linewidth=0, marker='.', markersize=5)
    ax4[1,1].plot(y_pred_test,y_test,color='green',label="R2 test: "+str(R2_test),fillstyle='none',linewidth=0, marker='.', markersize=5)
    
    
    ax4[0,0].set_xlabel("epoch")
    ax4[0,0].set_ylabel("energy validation")
    #ax4[0,0].legend()
    ax4[0,1].set_xlabel("epoch")
    ax4[0,1].set_ylabel("energy test")
    #ax4[0,1].legend()
    ax4[1,0].set_xlabel("epoch")
    ax4[1,0].set_ylabel("energy training")
    #ax4[1,0].legend()
    ax4[1,1].set_xlabel("energy")
    ax4[1,1].set_ylabel("energy")
    #ax4[1,1].legend()
    
    
    filename=str(path_exper)+"/estimations2.png"  
    fig4.savefig(filename, dpi=400, facecolor="#f1f1f1")
    
    
    
    
    
    
    
    
    
    print(".............................................................") 
    print("......*******DATA SAVING.....****************...")
    print(".............................................................")
    
    
    # Guardado de los datos de fitting de la red
    epoch_x = np.linspace(1,epochs,epochs)
    fitting_data = [epoch_x,training_cost, training_accuracy, evaluation_cost, evaluation_accuracy] 
    fitting_data = np.array(fitting_data).transpose()
    filename1 = str(path_exper)+"/"+"datosajuste.txt"
    np.savetxt(filename1, fitting_data)
    
    
    
    # Guardamos la estructura de la red
    filename_sum = str(path_exper)+"/"+"NN summary.dat"
    with open(filename_sum, 'w') as fh:
        #json.dump(model_summary,filehandle)
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        print ("model_summary have been created!")    
    
    
    # Guardado de la red
    model.save(path_exper)
    
    
    #Compilacion de datos de prediccion
    datos_pred_val = np.concatenate((orden_val,y_val,y_pred_val), axis=1) 
    datos_pred_test = np.concatenate((orden_test,y_test,y_pred_test), axis=1)
    datos_pred_train = np.concatenate((orden_train,y_train,y_pred_train), axis=1)
    
    #Guardado de datos en archivo
    filename1 = str(path_exper)+"/"+"y_val_pred.txt"
    np.savetxt(filename1, datos_pred_val)
    
    filename1 = str(path_exper)+"/"+"y_test_pred.txt"
    np.savetxt(filename1, datos_pred_test)
    
    filename1 = str(path_exper)+"/"+"y_train_pred.txt"
    np.savetxt(filename1, datos_pred_train)
    
    
    
    #Guardando el codigo utilizado
    code_path = inspect.getframeinfo(inspect.currentframe()).filename
    code_name = os.path.basename(code_path)
    print(code_path,code_name)
    shutil.copy(code_path, path_exper)
    
    
    #Guardado de datos iniciales
    data_INI=[path_data, norm_ener_type, norm_coulomb, num_atoms, frac_train, frac_val, epochs, BATCH_SIZE, id_matriz, id_Z, id_calcu, id_reducc, NN_type,perc_norm_ener,y1_norm_ener]
    filename = str(path_exper)+"/datosINI.txt"
    archivo = open(filename, "w")
    archivo.write(str((data_INI)))
    archivo.close()
    print(data_INI)
    
    '''
    image_number = 2
    
    
    pathimages=str(path_exper)+"/images_layers"
    isfile=os.path.exists(pathimages)    
    if (not(isfile)):  
        os.mkdir(pathimages)
    
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    for x in range(0,4):
        f1 = activation_model.predict(x_test[image_number].reshape(1,122,122,3))[x]
        for filter in range(0,len(f1[0,0,0,:])):
            plt.imshow(f1[0,:,:,filter],cmap='inferno')
            filename=str(pathimages)+"/imagenum_"+str(image_number)+"_layer_"+str(x)+"_filter_"+str(filter)+".png"  
            plt.savefig(filename, dpi=400, facecolor="#f1f1f1")
            
    '''  
    
    
    tf.keras.backend.clear_session()
    
    '''
    from numba import cuda
    cuda.select_device(0)
    cuda.close()
    '''
    
    end_time_1 = time.time()
    print("Tiempo para carga/normalizacion de datos de entreno: ", end_time_1-start_time_1, " segundos")
    

