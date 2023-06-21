# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:06:02 2022

@author: urbiz
"""




import numpy as np
from tensorflow.keras import models
import functions as fn
import matplotlib.pyplot as plt



# Loading experiment set

matrix_EP = np.load('../NN_DFT_EPAgClCuO/matrix_EP.npy')
matrix_AD = np.load('../NN_DFT_ADAgClCuO/matrix_AD.npy')
positXYZ = np.load('../NN_DFT_EPAgClCuO/positXYZ.npy')


# Loading and evaluating model EP and AD

path_results = "../NN_DFT_EPAgClCuO/NN14_def/NN14_10000epochs_16/"
model=models.load_model(path_results)
model.summary()


EnerEP_out = model.predict(matrix_EP)


path_results = "../NN_DFT_ADAgClCuO/NN14_def/NN14_10000epochs_16_NNpreparados/"
model=models.load_model(path_results)
model.summary()


EnerAD_out = model.predict(matrix_AD)

#Calculating eV through normalization coeficients
EnerEP_out_real=y_pred_val = fn.norm_energ (EnerEP_out,0.5,-503.266947,-502.14672,-505.88081)
EnerAD_out_real=y_pred_val = fn.norm_energ (EnerAD_out,0.5,-499.62241,-496.45495,-507.01315)





positXYZ=np.array(positXYZ,dtype=float)
positXYZ=np.reshape(positXYZ,(len(positXYZ[:,0,0]),len(positXYZ[0,0,:])))

difference=EnerEP_out_real-EnerAD_out_real
difference_norm=difference-min(difference)
main = np.concatenate((positXYZ,EnerEP_out_real,EnerAD_out_real,difference_norm),axis=1)



# Reorganization of data in 3D matrix
num_x=50
num_y=50
num_xy=num_x*num_y
num_z=3

energ_y=[]
energ_xy=[]
energ=[]

cont=0
for i in range(0,num_z):
    for j in range(0,num_x):
        for j in range(0,num_y):
            energ_y.append(main[cont,5])
            cont=cont+1
        energ_xy.append(energ_y)
        energ_y=[]
    energ.append(energ_xy)
    
    energ_xy=[]
    
        
        
energ=np.array(energ,dtype=float)


# Saving data
np.save('energies.npy', energ)
np.save('main.npy',main)



repetitions = (2,2)
tiled_energ = np.tile(energ[0],repetitions)



plt.imshow(tiled_energ, cmap='viridis')



  