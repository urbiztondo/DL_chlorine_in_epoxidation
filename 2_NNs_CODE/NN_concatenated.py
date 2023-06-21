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

path_data = "data_lejos_preparado/"

id_matriz = 2 #0- matriz 2D, 1-matriz3D con todos los id_zs, 3-matriz3D con normalizacion por partes en cada capa
id_calcu = 0 # 0- no se calculan, 1- se calculan
id_Z = 1 # calculo matriz de coulomb 1-Z pequenos, 2-Zgrandes, 3-peso a mas bajos
id_reducc =0 # 0-no se reduce, 1-se reduce a los atomos mas relevantes
id_flatten = 0 # 0-no se aplana la matriz, 1-se aplana para meter en vector

# Sobre las normalizaciones
norm_ener_type = 1 # 0-de 0a1, 1- por partes, 2-media/desvest, 4-concatenada
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
    

num_atoms = 129
frac_train=0.8
frac_val=0.1


epochs = 5000
BATCH_SIZE = 16

NN_type=14

verbose=0


pathresults="NN14_concate_datosNlejosN/"


#factor = 0.98
intensity = 0.3

for factor in [0.98,0.95,0.9,0.8,0.7,0.6]:
    print(factor)

    
    
        
    path_exper=pathresults+"NN14_conca_factor_"+str(factor)+"/"
    
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
    
    
    
    # Guarda energias y a partir del dato de energia limite se dividen en dos grupos de max y min, los que 
    #van por encima y por debajo de esos valores.
    
    energies_old = energies
    
    energies_bulk=np.array(energies,dtype=float)
    energies_bulk = energies_bulk[:,1]
    
    
    orden = np.linspace(0,len(energies_bulk)-1,len(energies_bulk)).reshape(len(energies_bulk),1)
    
    
    
    #factor = 0.98
    energ_valor_limit = np.max(energies_bulk)-((np.max(energies_bulk)-np.min(energies_bulk))*factor)
    
    energ_maximos_coef = np.where(energies_bulk > energ_valor_limit)
    energ_minimos_coef = np.where(energies_bulk < energ_valor_limit)
    
    
    
    ener_max = energies_bulk[energ_maximos_coef]
    ener_min = energies_bulk[energ_minimos_coef]
    ener_max = np.array(ener_max).reshape(len(ener_max),1)
    ener_min = np.array(ener_min).reshape(len(ener_min),1)
    
    print("longitud energias max: ",len(ener_max)," longitud energias min: ",len(ener_min))
    
    # Normalizaciones de los dos tramos de energia
    energ_max_norm,x1_max,y1_max,max_max,min_max = fn.energ_norm(perc_norm_ener,y1_norm_ener,ener_max)
    energ_max_norm = np.reshape(energ_max_norm,(len(energ_max_norm),1))
    energ_min_norm,x1_min,y1_min,max_min,min_min = fn.energ_norm(perc_norm_ener,y1_norm_ener,ener_min)
    energ_min_norm = np.reshape(energ_min_norm,(len(energ_min_norm),1))
    
    # con los ordenes de las energias se sacan los valores de las matrices separadas en las dos max y min
    orden_max = orden[energ_maximos_coef] 
    orden_min = orden[energ_minimos_coef]
    
    matrix_max = matrix[energ_maximos_coef,:,:,:]
    matrix_max = matrix_max.reshape(matrix_max.shape[1],matrix_max.shape[2],matrix_max.shape[3],matrix_max.shape[4])
    matrix_min = matrix[energ_minimos_coef,:,:,:]
    matrix_min = matrix_min.reshape(matrix_min.shape[1],matrix_min.shape[2],matrix_min.shape[3],matrix_min.shape[4])
    
    # con el coeficiente intensity se atenua la contribucion de las matrices una respecto a la otra.
    #intensity = 0.3
    
    matrix_inlet_max = np.concatenate((matrix_max*intensity,matrix_min),axis=0)
    matrix_inlet_min = np.concatenate((matrix_max,matrix_min*intensity),axis=0)
    energies_max_min = np.concatenate((energ_max_norm,energ_min_norm),axis=0)
    orden = np.concatenate((orden_max,orden_min),axis=0)
    
    #Aqui ojo, al juntar las energias las len(ener_max) tendran unos valores maxymin propios
    #y los len(ener_min) tendran otros minymax para la conversion a unidades de energia
    
    
    # mezclado de las matrices y energias y demas... se denotan como _sh como shuffled
    print("Collecting all data and shuffle and unzip")
    merged_all=list(zip(matrix_inlet_max,matrix_inlet_min,energies_max_min,orden))
    random.shuffle(merged_all)
    matrix_inlet_max_sh,matrix_inlet_min_sh,energies_max_min_sh,orden_sh=zip(*merged_all)
    
    ##### Pasando todo a np array ##########
    matrix_inlet_max_sh = np.array(matrix_inlet_max_sh,dtype=float)
    matrix_inlet_min_sh = np.array(matrix_inlet_min_sh,dtype=float)
    energies_max_min_sh = np.array(energies_max_min_sh,dtype=float)
    orden_sh = np.array(orden_sh,dtype=float)
    
    
    # Deduce el orden en el que se han mezclado los valores
    #♠util para luego sacar las energias con los dos coeficentes min y max diferentes
    list_shuffle = []
    for i in range(0,len(orden)):
        A = np.where(orden_sh[:,0]==orden[i])
        #print(A)
        list_shuffle.append(A)
    list_shuffle = np.array(list_shuffle).reshape(len(list_shuffle),1)
    
    
    
    # preparacion de datos de entrada a la red
    frac_test=1-frac_train-frac_val
    
    
    
    itrain=0;ftrain=int(len(matrix_inlet_max_sh)*frac_train)
    ival=ftrain;fval=int(len(matrix_inlet_max_sh)*(frac_train+frac_val))
    itest=fval;ftest=len(matrix_inlet_max_sh)
    
    x_train_max = matrix_inlet_max_sh[itrain:ftrain]
    x_val_max   = matrix_inlet_max_sh[ival:fval]
    x_test_max  = matrix_inlet_max_sh[itest:ftest]
    
    x_train_min = matrix_inlet_min_sh[itrain:ftrain]
    x_val_min  = matrix_inlet_min_sh[ival:fval]
    x_test_min  = matrix_inlet_min_sh[itest:ftest]
    
    y_train = energies_max_min_sh[itrain:ftrain]
    y_val = energies_max_min_sh[ival:fval]
    y_test = energies_max_min_sh[itest:ftest]
    
    orden_train = orden_sh[0,itrain:ftrain]
    orden_val = orden_sh[0,ival:fval]
    orden_test= orden_sh[0,itest:ftest]
    
    
    
    # modelo
    
    
    input_A = keras.layers.Input(shape=(x_train_max.shape[1],x_train_max.shape[2],x_train_max.shape[3]), name="max_input")
    input_B = keras.layers.Input(shape=(x_train_min.shape[1],x_train_min.shape[2],x_train_min.shape[3]), name="min_input")
    
    
    
    hidden0_A = keras.layers.Conv2D(64,(3,3),
                                  activation='selu',
                                  kernel_initializer='HeUniform',
                                  bias_initializer='zeros')(input_A)
    hidden1_A = keras.layers.MaxPool2D(2,2)(hidden0_A)
    hidden2_A = keras.layers.Conv2D(128,(3,3),
                                  activation='selu',
                                  input_shape=(x_train_max.shape[1],x_train_max.shape[2],x_train_max.shape[3]),
                                  kernel_initializer='HeUniform',
                                  bias_initializer='zeros')(hidden1_A)
    hidden3_A = keras.layers.MaxPool2D(2,2)(hidden2_A)
    hidden4_A = keras.layers.Conv2D(256,(3,3),
                                  activation='selu',
                                  input_shape=(x_train_max.shape[1],x_train_max.shape[2],x_train_max.shape[3]),
                                  kernel_initializer='HeUniform',
                                  bias_initializer='zeros')(hidden3_A)
    hidden5_A = keras.layers.MaxPool2D(2,2)(hidden4_A)
    hidden6_A = keras.layers.Flatten()(hidden5_A)
    hidden7_A = keras.layers.Dense(256,activation='sigmoid',
                    kernel_initializer='he_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=regularizers.L1L2(l1=1e-8, l2=1e-6))(hidden6_A)
    hidden8_A = keras.layers.Dropout(0.1)(hidden7_A)
    hidden9_A = keras.layers.Dense(128,activation='sigmoid',
                    kernel_initializer='he_normal',
                    bias_initializer='zeros')(hidden8_A)
    
    hidden0_B = keras.layers.Conv2D(64,(3,3),
                                  activation='selu',
                                  kernel_initializer='HeUniform',
                                  bias_initializer='zeros')(input_B)
    hidden1_B = keras.layers.MaxPool2D(2,2)(hidden0_B)
    hidden2_B = keras.layers.Conv2D(128,(3,3),
                                  activation='selu',
                                  input_shape=(x_train_max.shape[1],x_train_max.shape[2],x_train_max.shape[3]),
                                  kernel_initializer='HeUniform',
                                  bias_initializer='zeros')(hidden1_B)
    hidden3_B = keras.layers.MaxPool2D(2,2)(hidden2_B)
    hidden4_B = keras.layers.Conv2D(256,(3,3),
                                  activation='selu',
                                  input_shape=(x_train_max.shape[1],x_train_max.shape[2],x_train_max.shape[3]),
                                  kernel_initializer='HeUniform',
                                  bias_initializer='zeros')(hidden3_B)
    hidden5_B = keras.layers.MaxPool2D(2,2)(hidden4_B)
    hidden6_B = keras.layers.Flatten()(hidden5_B)
    hidden7_B = keras.layers.Dense(256,activation='sigmoid',
                    kernel_initializer='he_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=regularizers.L1L2(l1=1e-8, l2=1e-6))(hidden6_B)
    hidden8_B = keras.layers.Dropout(0.1)(hidden7_B)
    hidden9_B = keras.layers.Dense(128,activation='sigmoid',
                    kernel_initializer='he_normal',
                    bias_initializer='zeros')(hidden8_B)
    
    
    concat = keras.layers.concatenate([hidden9_A,hidden9_B])
    
    output = keras.layers.Dense(1, name="output", activation='sigmoid',
                    kernel_initializer='he_normal',
                    bias_initializer='zeros')(concat)
    
    
    model = keras.Model(inputs=[input_A,input_B],outputs=[output])
    
    
    #model.summary()
    
    opt = tf.keras.optimizers.Nadam(
        learning_rate=0.00001,
        beta_1=0.99,
        beta_2=0.999,
        epsilon=1e-08,
        name='Nadam')
    
    model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=['mae'])
    
    
    history = model.fit(
        (x_train_max,x_train_min),
        y_train,
        validation_data=((x_val_max,x_val_min),y_val),
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
    print("training_cost=",training_cost[-1],"    Sumatorio="+str(np.sum(training_cost)))
    print("evaluation_cost=",evaluation_cost[-1],"    Sumatorio="+str(np.sum(evaluation_cost)))
    print("training_accuracy=",training_accuracy[-1],"    Sumatorio="+str(np.sum(training_accuracy)))
    print("evaluation_accuracy=",evaluation_accuracy[-1],"    Sumatorio="+str(np.sum(evaluation_accuracy)))
    
    
    
    print(".............................................................") 
    print("......*******GRAPHING****************...")
    print(".............................................................")
    
    #Creamos los directorios de trabajo
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
    plt.show()
    filename=str(path_exper)+"/"+"im_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".png"  
    fig2.savefig(filename, dpi=400, facecolor="#f1f1f1")
    
    
    
    ###############################
    #### GRAFICA 4 en 1 de energias pred y real y y=y con r2  ######
    ###############################
    
    ################# max ########################
    
    #Calculo de los valores de y en la red   
    y_pred_val = model.predict_on_batch((x_val_max,x_val_min))     
    y_pred_test = model.predict_on_batch((x_test_max,x_test_min))
    y_pred_train = model.predict_on_batch((x_train_max,x_train_min))
    
    y_pred = np.concatenate((y_pred_train,y_pred_val,y_pred_test),axis=0)
    y_pred_unshuffled = y_pred[list_shuffle].reshape(len(y_pred),1)
    y_pred_unshu_max = y_pred_unshuffled[0:len(matrix_max),0]
    y_pred_unshu_min = y_pred_unshuffled[len(matrix_max):,0]
    
    y_pred_unsh_max_eV = fn.norm_energ(y_pred_unshu_max,y1_max,x1_max,max_max,min_max).reshape(len(y_pred_unshu_max),1)
    y_pred_unsh_min_eV = fn.norm_energ(y_pred_unshu_min,y1_min,x1_min,max_min,min_min).reshape(len(y_pred_unshu_min),1)
    
    
    
    
    
    R2_max = fn.rsquared(y_pred_unsh_max_eV, ener_max)
    R2_min = fn.rsquared(y_pred_unsh_min_eV, ener_min)
    print(R2_max,R2_min)
    
    
    plt.plot(y_pred_unsh_max_eV[:,0],ener_max[:,0],color='blue',label="max "+str(R2_max),fillstyle='none',linewidth=0, marker='.', markersize=3)
    plt.plot(y_pred_unsh_min_eV[:,0],ener_min[:,0],color='red',label="min "+str(R2_min),fillstyle='none',linewidth=0, marker='.', markersize=3)
    plt.xlabel('real energy eV')
    plt.ylabel('stimated energy eV')
    plt.title("energy vs energy")
    plt.legend()
    filename=str(path_exper)+"/R2s.png"
    plt.savefig(filename, dpi=400, facecolor="#f1f1f1")
    plt.show()
    
    
    dif_max = abs(y_pred_unsh_max_eV - ener_max)
    dif_min = abs(y_pred_unsh_min_eV - ener_min)
    sum_max =np.sum(dif_max)
    sum_min = np.sum(dif_min)
    
    plt.plot(orden_max,dif_max,color='blue',label="max "+str(sum_max),fillstyle='none',linewidth=0, marker='.', markersize=3)
    plt.plot(orden_min,dif_min,color='red',label="max "+str(sum_min),fillstyle='none',linewidth=0, marker='.', markersize=3)
    plt.xlabel('order')
    plt.ylabel('diference energy eV')
    plt.title("diferencia entre energia real-estimada")
    plt.legend()
    filename=str(path_exper)+"/diferences.png"
    plt.savefig(filename, dpi=400, facecolor="#f1f1f1")
    plt.show()
    
    
    
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
    datos_pred_max = np.concatenate((orden_max,ener_max,y_pred_unsh_max_eV), axis=1)
    datos_pred_min = np.concatenate((orden_min,ener_min,y_pred_unsh_min_eV), axis=1)
    
    #Guardado de datos en archivo
    filename1 = str(path_exper)+"/"+"y_max_pred.txt"
    np.savetxt(filename1, datos_pred_max)
    
    filename1 = str(path_exper)+"/"+"y_min_pred.txt"
    np.savetxt(filename1, datos_pred_min)
    
    
    #Guardando el codigo utilizado
    code_path = inspect.getframeinfo(inspect.currentframe()).filename
    code_name = os.path.basename(code_path)
    print(code_path,code_name)
    shutil.copy(code_path, path_exper)
    
    
    
    R2_train = fn.rsquared(y_train, y_pred_train) 
    R2_test = fn.rsquared(y_test, y_pred_test) 
    R2_val = fn.rsquared(y_val, y_pred_val) 
    
    
    #Guardado de datos iniciales
    data_INI=[factor,R2_train,R2_test,R2_val, path_data, norm_ener_type, norm_coulomb, num_atoms, frac_train, frac_val, epochs, BATCH_SIZE, id_matriz, id_Z, id_calcu, id_reducc, NN_type,perc_norm_ener,y1_norm_ener]
    filename = str(path_exper)+"/datosINI.txt"
    archivo = open(filename, "w")
    archivo.write(str((data_INI)))
    archivo.close()
    print(data_INI)
    
    
    
    
    
    
    tf.keras.backend.clear_session()
    
