# -*- coding: utf-8 -*-
"""
Created on Tue May  2 07:42:13 2023

@author: urbiz
"""
import numpy as np
from tensorflow.keras import models
import functions as fn

#Loading the coulomb matrix created for the selected positions
matrix_AgClCuO = np.load('../NN_DFT_AgClCuO/matrix_posit_gener_05.npy')


# Loading the model fine optimized
path_results ='../NN_DFT_AgClCuO/NN14_def/NN14_10000epochs_16/'
model=models.load_model(path_results)
model.summary()

# Energy calculation
EnerAgClCuO_out = model.predict(matrix_AgClCuO)
# Energy calculation in eV using normalization coeficients
EnerAgClCuO_out_real = fn.norm_energ (EnerAgClCuO_out,0.5,-464.285295,-463.12086,-467.00231)


# Reorganization of data in 3D matrix to graphical representation
def line_2_array(main):
    num_x=25
    num_y=25
    num_z=1
    
    energ_y=[]
    energ_xy=[]
    energ=[]
    
    cont=0
    for i in range(0,num_z):
        for j in range(0,num_x):
            for j in range(0,num_y):
                energ_y.append(main[cont])
                cont=cont+1
            energ_xy.append(energ_y)
            energ_y=[]
        energ.append(energ_xy)
        
        energ_xy=[]
        
            
            
    return np.array(energ,dtype=float)

#Saving data
EnerAgClCuO = line_2_array(EnerAgClCuO_out_real)
EnerAgClCuO = np.squeeze(EnerAgClCuO,axis=-1)
np.save('EnerAgClCuO_05.npy',EnerAgClCuO)
np.save('energ_agClCuO_05.npy',EnerAgClCuO_out_real)
