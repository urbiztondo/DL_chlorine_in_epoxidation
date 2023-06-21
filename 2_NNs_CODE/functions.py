# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:20:37 2021

@author: urbiz
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm


def coulomb_short(matrix,atoms):
    matrix_cut = matrix[:,atoms,:]
    matrix_cut_cut = matrix_cut[:,:,atoms]

    return matrix_cut_cut


def charg_vect_func(elements,num_atoms):
    
    carga_Cu = 29
    carga_O = 8
    carga_Ag = 47
    carga_Cl = 37
    carga_C = 6
    carga_H = 1
   
  
    charge_vector = []
    
    for i in range(0,num_atoms):
        element = elements[i]
    
        if element == ['Cu']:
            #print(str(i)+"   "+"es cobre")
            charge_vector.append(carga_Cu)
        if element == ['O']:
            #print(str(i)+"   "+"es oxigeno")
            charge_vector.append(carga_O)
        if element == ['Ag']:
            #print(str(i)+"   "+"es plata") 
            charge_vector.append(carga_Ag)
        if element == ['Cl']:
            #print(str(i)+"   "+"es cloro")
            charge_vector.append(carga_Cl)
        if element == ['C']:
            #print(str(i)+"   "+"es carbono")
            charge_vector.append(carga_C)
        if element == ['H']:
            #print(str(i)+"   "+"es hidrogeno")
            charge_vector.append(carga_H)
            
        
    with open('charge_vector.dat', 'w') as filehandle:
        json.dump(charge_vector,filehandle)
        print ("charge_vector have been created!")
        
    return charge_vector
        
def charg_vect_funcZ1(elements,num_atoms):
    
    carga_Cu = 29
    carga_O = 8
    carga_Ag = 47
    carga_Cl = 37
    carga_C = 6
    carga_H = 1
   
  
    charge_vector = []
    
    for i in range(0,num_atoms):
        element = elements[i]
    
        if element == ['Cu']:
            #print(str(i)+"   "+"es cobre")
            charge_vector.append(carga_Cu)
        if element == ['O']:
            #print(str(i)+"   "+"es oxigeno")
            charge_vector.append(carga_O)
        if element == ['Ag']:
            #print(str(i)+"   "+"es plata") 
            charge_vector.append(carga_Ag)
        if element == ['Cl']:
            #print(str(i)+"   "+"es cloro")
            charge_vector.append(carga_Cl)
        if element == ['C']:
            #print(str(i)+"   "+"es carbono")
            charge_vector.append(carga_C)
        if element == ['H']:
            #print(str(i)+"   "+"es hidrogeno")
            charge_vector.append(carga_H)
            
        
    with open('charge_vector.dat', 'w') as filehandle:
        json.dump(charge_vector,filehandle)
        print ("charge_vector have been created!")
        
    return charge_vector

def charg_vect_funcZ2(elements,num_atoms):
    
    carga_Cu = 9
    carga_O = 6
    carga_Ag = 1
    carga_Cl = 7
    carga_C = 4
    carga_H = 1
   
  
    charge_vector = []
    
    for i in range(0,num_atoms):
        element = elements[i]
    
        if element == ['Cu']:
            #print(str(i)+"   "+"es cobre")
            charge_vector.append(carga_Cu)
        if element == ['O']:
            #print(str(i)+"   "+"es oxigeno")
            charge_vector.append(carga_O)
        if element == ['Ag']:
            #print(str(i)+"   "+"es plata") 
            charge_vector.append(carga_Ag)
        if element == ['Cl']:
            #print(str(i)+"   "+"es cloro")
            charge_vector.append(carga_Cl)
        if element == ['C']:
            #print(str(i)+"   "+"es carbono")
            charge_vector.append(carga_C)
        if element == ['H']:
            #print(str(i)+"   "+"es hidrogeno")
            charge_vector.append(carga_H)
            
        
    with open('charge_vector.dat', 'w') as filehandle:
        json.dump(charge_vector,filehandle)
        print ("charge_vector have been created!")
        
    return charge_vector

def charg_vect_funcZ3(elements,num_atoms):
    
    carga_Cu = 2
    carga_O = 2
    carga_Ag = 10
    carga_Cl = 10
    carga_C = 8
    carga_H = 6
   
  
    charge_vector = []
    
    for i in range(0,num_atoms):
        element = elements[i]
    
        if element == ['Cu']:
            #print(str(i)+"   "+"es cobre")
            charge_vector.append(carga_Cu)
        if element == ['O']:
            #print(str(i)+"   "+"es oxigeno")
            charge_vector.append(carga_O)
        if element == ['Ag']:
            #print(str(i)+"   "+"es plata") 
            charge_vector.append(carga_Ag)
        if element == ['Cl']:
            #print(str(i)+"   "+"es cloro")
            charge_vector.append(carga_Cl)
        if element == ['C']:
            #print(str(i)+"   "+"es carbono")
            charge_vector.append(carga_C)
        if element == ['H']:
            #print(str(i)+"   "+"es hidrogeno")
            charge_vector.append(carga_H)
            
        
    with open('charge_vector.dat', 'w') as filehandle:
        json.dump(charge_vector,filehandle)
        print ("charge_vector have been created!")
        
    return charge_vector

def ener_index(energies,min_dif):

    energies=np.array(energies,dtype=float)
    energies=energies[:,1]
    
    indices_ex = []
    indices = []
    for i in range(0,len(energies)):
        dif = energies[i] - energies[i-1]
        #print (str(i)+" diferencia"+str(dif))
        if (dif < min_dif):
            #print("mayor que diferencia establecida"+str(i)+"diferencia"+str(dif))
            indices.append(i)
        if (dif >= min_dif):
            indices_ex.append(i)
    
    '''
    titulo = "valor de min_dif y de longitud de datos"+str(min_dif)+" "+str(len(indices))
    plt.title(titulo)
    plt.bar(indices,1,align="center")
    filename = "fig_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".png"
    plt.savefig(filename)
    '''
    return indices,indices_ex



def module(merged_pos_atom,num_atom,filename):

    M_merged3d=[]
    M_merged2d=[]
    M=[]
    for exp in range(0,len(merged_pos_atom)):
        for i in range(0,num_atom):
            for j in range(0,num_atom):
                #print(str(exp)+"   "+str(i)+"  "+str(j))
                distX=(float(merged_pos_atom[exp][j][0])-float(merged_pos_atom[exp][i][0]))**2
                distY=(float(merged_pos_atom[exp][j][1])-float(merged_pos_atom[exp][i][1]))**2
                distZ=(float(merged_pos_atom[exp][j][2])-float(merged_pos_atom[exp][i][2]))**2
                #print("distX= "+str(distX)+"distY= "+str(distY)+"distZ= "+str(distZ))
                dist=(distX+distY+distZ)**(1/2)
                M.append(dist)
            M_merged2d.append(M)
            M=[]
        M_merged3d.append(M_merged2d)
        M_merged2d=[]
    #Matrix = np.array(M_merged3d)
    

    Matrix = M_merged3d
    
    plt.imshow(Matrix[0])
    
    
    filename = "modulo_" + filename + ".dat"

    
    
    with open(filename, 'w') as filehandle:
        json.dump(Matrix,filehandle)
        print ("module matrix have been created!")
    return Matrix
    


def coulomb(matrix,charge_vector,filename):

    matrix=np.array(matrix,dtype=float)
    
    
    matrix_calculated=[]
    matrix2d_calculated=[]
    matrix3d_calculated=[]
    for exp in range(0,matrix.shape[0]):
            for j in range(0,matrix.shape[1]):
                for i in range(0,matrix.shape[2]):
                    if i == j:
                        val = 0.5 * charge_vector[i]**2.4
                        #print("Experimento: "+str(exp)+"  "+"Columna: "+str(j)+"  "+"Fila: "+str(i)+"  "+"valor: "+str(val))
                        matrix_calculated.append(val)
                    else:
                        val = (charge_vector[i]*charge_vector[j])/matrix[exp,j,i]
                        #print("Experimento: "+str(exp)+"  "+"Columna: "+str(j)+"  "+"Fila: "+str(i)+"  "+"valor: "+str(val))
                        matrix_calculated.append(val)
                matrix2d_calculated.append(list(matrix_calculated))
                matrix_calculated = []
            matrix3d_calculated.append(matrix2d_calculated)
            matrix2d_calculated = []
    
    
    matrix = np.array(matrix3d_calculated,dtype=float)
    matrix=matrix/np.amax(matrix)
    plt.imshow(matrix[0])


    filename = "coulomb_" + filename + ".dat"
    
    with open(filename, 'w') as filehandle:
        json.dump(matrix3d_calculated,filehandle)
        print ("matrix3d_coulomb have been created!")
    
    return matrix3d_calculated
    


def toeV(energy,y1,x1,max,min):
    
    energy_eV = []
    
    for i in range (0,len(energy)):
    
        if (energy[i] <= y1):
            energy_eV_ind = ((energy[i]/y1)*(x1-max))+max
        if (energy[i] > y1):
            energy_eV_ind = (((energy[i]-y1)/(1-y1))*(min-x1))+x1
        energy_eV.append(energy_eV_ind)
    
    return energy_eV



def NN(model,matrix_inlet,energies,orden,y1,x1,max,min,path_exper):
    
    matrix_fullposit = matrix_inlet.reshape(matrix_inlet.shape[0],matrix_inlet.shape[1],matrix_inlet.shape[2],1)
    energies = np.array(energies)
    energies = energies.reshape(energies.shape[0],1)
    
    posit_calcula = matrix_fullposit
    datos_ener = energies
    #orden_calc = np.array(orden)
    orden_calc = np.array(orden).reshape(len(datos_ener),1)
    model_out = model.predict(posit_calcula)
   
    #y_test_matrix = np.array(y_test).reshape(len(y_test),1)
    model_out = np.array(model_out)
    Energias_estimadas=toeV(model_out[:,0],y1,x1,max,min)
    Energias_reales=toeV(datos_ener,y1,x1,max,min)
    Energias_reales = np.array(Energias_reales).reshape(len(datos_ener),1)
    
    
    orden = np.array(orden_calc).reshape(len(datos_ener),1)
    
    
    Porcentaje_error=abs((Energias_estimadas-Energias_reales)/Energias_reales)*100
      
    
    datos = np.concatenate((orden_calc,Energias_reales,Energias_estimadas,Porcentaje_error), axis=1) 
    #datos_unsort = datos
    
    
    datos=datos[np.argsort(datos[:,0])]
    fig3, bx = plt.subplots(2,1, figsize=(8,8))
    
    xx2 = np.linspace(0,len(datos)-1,len(datos))
    
    bx[0].plot(xx2,datos[:,2],color="blue", label="NW",fillstyle='none',linewidth=0, marker='.', markersize=15)
    bx[0].plot(xx2,datos[:,1], color="red", label="DFT",linewidth=0, marker='.', markersize=10)  
      
    txt = " text1 "
    bx[0].set_xlabel(txt)    
    bx[0].set_ylabel("Energy (a.u.)")
    bx[0].legend() 
    
    bx[1].plot(xx2,datos[:,3],color="black", label="NW",linewidth=0, marker='.', markersize=6)
    txt = "text2"
    bx[1].set_xlabel(txt)    
    bx[1].set_ylabel("% error")
    
    filename=str(path_exper)+"/"+"graferr_tot"+str(len(datos))+"im_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".png"  
    fig3.savefig(filename, dpi=400, facecolor="#f1f1f1")
    

       
    return datos

def deriv(datos_ajuste):
    
    datos_ajuste=np.array(datos_ajuste,dtype='float')
    
    
        
    datos_pend = []
    for j in range (0, len(datos_ajuste)-2):
        x = datos_ajuste[j+1,0]
        px_ant = datos_ajuste[j,0]
        px_des = datos_ajuste[j+2,0]
        #print(px_ant, x, px_des)
        
        y = datos_ajuste[j+1,1]
        py_ant = datos_ajuste[j,1]
        py_des =datos_ajuste[j-1,1]
        #print(py_ant, y, py_des)
    
        prom_x_ant = (px_ant+x)/2
        prom_x_des = (px_des+x)/2
        prom_ant = (py_ant+y)/2
        prom_des = (py_des+y)/2
        #print(prom_x_ant,prom_x_des)
        #print(prom_ant, prom_des)
        
        pendiente = (prom_des - prom_ant) / (prom_x_des - prom_x_ant)
        #print(pendiente)
        
        datos_pend_list = [x,pendiente]
        datos_pend.append(datos_pend_list)    
    
    
    
    
    
    return datos_pend







def module4d(merged_pos_atom,num_atom):

    M_merged4d=[]
    M_merged2d_xy = []
    M_merged2d_xz = []
    M_merged2d_yz = []
    M_xy=[]
    M_xz=[]
    M_yz=[]
    for exp in range(0,len(merged_pos_atom)):
        for i in range(0,num_atom):
            for j in range(0,num_atom):
                distX=(float(merged_pos_atom[exp][j][0])-float(merged_pos_atom[exp][i][0]))**2
                distY=(float(merged_pos_atom[exp][j][1])-float(merged_pos_atom[exp][i][1]))**2
                distZ=(float(merged_pos_atom[exp][j][2])-float(merged_pos_atom[exp][i][2]))**2
                distXY=(distX+distY)**(1/2)
                distXZ=(distX+distZ)**(1/2)
                distYZ=(distY+distZ)**(1/2)
                M_xy.append(distXY)
                M_xz.append(distXZ)
                M_yz.append(distYZ)
            M_merged2d_xy.append(M_xy)
            M_merged2d_xz.append(M_xz)
            M_merged2d_yz.append(M_xy)                                         
            M_xy=[]
            M_xz=[]
            M_yz=[]
        M_merged3d = [M_merged2d_xy,M_merged2d_xz,M_merged2d_yz]
        M_merged2d_xy = []
        M_merged2d_xz = []
        M_merged2d_yz = []
        M_merged4d.append(M_merged3d)
        M_merged3d = []
      
        
        #print("Fin del procesado del experimento nº: "+str(exp+1))
    
    
    with open('module.dat', 'w') as filehandle:
        json.dump(M_merged4d,filehandle)
        print ("module 4d matrix have been created!")
        
    
    

        
    return M_merged4d




def coulomb2by2(matrix,charge_vector):
    
    matrix_calculated = []
    matrix_coulomb_2d = []
    
    for j in range(0,matrix.shape[0]):
        for i in range(0,matrix.shape[1]):
            if i == j:
                val = 0.5 * charge_vector[i]**2.4
                matrix_calculated.append(val)
            else:
                val = 0
                if (matrix[j,i] != 0):
                    val = (charge_vector[i]*charge_vector[j])/matrix[j,i]
                matrix_calculated.append(val)
        matrix_coulomb_2d.append(matrix_calculated)
        matrix_calculated = []
    
    return matrix_coulomb_2d
        
    
    
    
def matrCoulommult(matrix,charge_vector):
    matrix = np.array(matrix)


    matriz_coulomb_layers = []
    matriz_coulomb_exp = []
    
    for exp in range(0,matrix.shape[0]):
        for layer in range(0,matrix.shape[1]):
            
            matriz_coulomb = coulomb2by2(matrix[exp,layer,:,:],charge_vector)
            matriz_coulomb_layers.append(matriz_coulomb)
        
        matriz_coulomb_exp.append(matriz_coulomb_layers)
        matriz_coulomb_layers = []
        
    with open('coulomb4d.dat', 'w') as filehandle:
        json.dump(matriz_coulomb_exp,filehandle)
        print ("matriz coulomb 4d matrix have been created!")





    return matriz_coulomb_exp
    
    
    
    
    
    
def NN_3d(model,matrix_inlet,energies,orden,y1,x1,max,min,path_exper):
    
    matrix_fullposit = matrix_inlet
    energies = energies.reshape(energies.shape[0],1,1)
    
    posit_calcula = matrix_fullposit
    datos_ener = energies
    orden_calc = np.array(orden)
    orden_calc = np.array(orden).reshape(len(datos_ener),1)
    model_out = model.predict(posit_calcula)
   
    #y_test_matrix = np.array(y_test).reshape(len(y_test),1)
      
    Energias_estimadas=toeV(model_out,y1,x1,max,min)
    Energias_reales=toeV(datos_ener,y1,x1,max,min)
    Energias_reales = np.array(Energias_reales).reshape(len(datos_ener),1)
    
    
    orden = np.array(orden_calc).reshape(len(datos_ener),1)
    
    
    Porcentaje_error=abs((Energias_estimadas-Energias_reales)/Energias_reales)*100
      
    
    datos = np.concatenate((orden_calc,Energias_reales,Energias_estimadas,Porcentaje_error), axis=1) 
    #datos_unsort = datos
    
    
    datos=datos[np.argsort(datos[:,0])]
    fig3, bx = plt.subplots(2,1, figsize=(8,8))
    
    xx2 = np.linspace(0,len(datos)-1,len(datos))
    
    bx[0].plot(xx2,datos[:,2],color="blue", label="NW",fillstyle='none',linewidth=0, marker='.', markersize=15)
    bx[0].plot(xx2,datos[:,1], color="red", label="DFT",linewidth=0, marker='.', markersize=10)  
      
    txt = " text1 "
    bx[0].set_xlabel(txt)    
    bx[0].set_ylabel("Energy (a.u.)")
    bx[0].legend() 
    
    bx[1].plot(xx2,datos[:,3],color="black", label="NW",linewidth=0, marker='.', markersize=6)
    txt = "text2"
    bx[1].set_xlabel(txt)    
    bx[1].set_ylabel("% error")
    
    filename=str(path_exper)+"/"+"graferr_tot"+str(len(datos))+"im_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".png"  
    fig3.savefig(filename, dpi=400, facecolor="#f1f1f1")
    

       
    return datos
    

def ener_norm2bulk(energies,x1,y1,max,min):

     contar_antes = 0
     contar_desp = 0
     
     energ_bulk = []
     for i in range (0,len(energies)):
         if (energies[i]>=y1):
             ener_cal = ((energies[i]/y1)*(x1-max))+max
             energ_bulk.append(ener_cal)
             contar_antes = contar_antes + 1
         if (energies[i]<y1):
             ener_cal = (((energies[i]-y1)/(1-y1))*(min-x1))+x1
             energ_bulk.append(ener_cal)
             contar_desp = contar_desp + 1
     
     print(contar_antes,contar_desp)
     energ_bulk=np.array(energ_bulk,dtype=float)
     return energ_bulk

    
def energ_norm(perc,y1,energies):
    #ener_sort = sorted(energies)
    #mediana = ener_sort[int(len(energies)/2)]
    max = np.max(energies)
    min = np.min(energies)
    
    #perc = 0.65
    x1 = max + (min - max)*perc
    #y1 = .1
    
    print(y1,x1,max,min)
    contar_antes = 0
    contar_desp = 0
    
    energ_norm = []
    for i in range (0,len(energies)):
        if (energies[i]>=x1):
            ener_cal = ((energies[i]-max)/(x1-max))*y1
            energ_norm.append(ener_cal)
            contar_antes = contar_antes + 1
        if (energies[i]<x1):
            ener_cal = (((energies[i]-x1)/(min-x1))*(1-y1))+y1
            energ_norm.append(ener_cal)
            contar_desp = contar_desp + 1
    
    print(contar_antes,contar_desp)
    energies=np.array(energ_norm,dtype=float)
        
    return energies,x1,y1,max,min
        

    
def energ_norm_vector(perc,y1,min,max,x1,vector):


    contar_antes = 0
    contar_desp = 0
    
    vector_norm = []
    for i in range (0,len(vector)):
        if (vector[i]>=x1):
            vector_cal = ((vector[i]-max)/(x1-max))*y1
            vector_norm.append(vector_cal)
            contar_antes = contar_antes + 1
        if (vector[i]<x1):
            vector_cal = (((vector[i]-x1)/(min-x1))*(1-y1))+y1
            vector_norm.append(vector_cal)
            contar_desp = contar_desp + 1
    
    print(contar_antes,contar_desp)
    vector=np.array(vector_norm,dtype=float)
        
    return vector


    
def input_vectors (frac_train,frac_val,matrix_inlet,energies,orden,title):
    
    #frac_test=1-frac_train-frac_val


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
    
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
    x_val=x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
    
    y_train=np.reshape(y_train,(len(y_train),1,1))
    y_val=np.reshape(y_val,(len(y_val),1,1))
    y_test=np.reshape(y_test,(len(y_test),1,1))
    
    
    print("Para los valores de: "+title)
    print("x_train: "+str(x_train.shape[0])+" "+str(x_train.shape[1])+" "+str(x_train.shape[2])+" "+str(x_train.shape[3]))
    print("x_val: "+str(x_val.shape[0])+" "+str(x_val.shape[1])+" "+str(x_val.shape[2])+" "+str(x_val.shape[3]))
    print("x_test: "+str(x_test.shape[0])+" "+str(x_test.shape[1])+" "+str(x_test.shape[2])+" "+str(x_test.shape[3]))
    
   
    return x_train, x_test, x_val, y_train, y_test, y_val, orden_train, orden_test, orden_val
    


def perc (matrix_coulomb_bulk):
    
      
    filas=[]
    columnas=[]

    for i in range(0,len(matrix_coulomb_bulk[1])):
        for j in range(0,len(matrix_coulomb_bulk[1])):
            #print (i,j)
            mean = np.mean(matrix_coulomb_bulk[:,i,j])
            desv = np.std(matrix_coulomb_bulk[:,i,j])
            perc = desv/mean
            filas.append(perc)
        columnas.append(filas)
        filas=[]

    columnas=np.array(columnas,dtype=float)
    
    return columnas
    
    
def norm2 (matrix_coulomb_bulk,percenta,y1_coulomb,num_atoms):
    
    maximo = np.max(matrix_coulomb_bulk)
    minimo = np.min(matrix_coulomb_bulk)
    x1_coulomb = maximo + (minimo - maximo)*percenta
    
    matrix_coulomb_norm = []
    for i in range(0,len(matrix_coulomb_bulk)):
        vector = matrix_coulomb_bulk[i].flatten()
        vector_norm = energ_norm_vector(percenta,y1_coulomb,minimo,maximo,x1_coulomb,vector)
        matrix_uni = vector_norm.reshape(num_atoms,num_atoms)
        matrix_coulomb_norm.append(matrix_uni)
    matrix_coulomb_norm=np.array(matrix_coulomb_norm,dtype=float)
    
    return matrix_coulomb_norm


def trian_flatten(matrix_coulomb_bulk):
    cont = 0
    lista = []
    X = []
    for k in range(0,len(matrix_coulomb_bulk)):
        for i in range(0,len(matrix_coulomb_bulk[0])):
            for j in range(0,i):
                lista.append(matrix_coulomb_bulk[k,i,j])
                X.append(cont)
                cont = cont + 1    
    lista.sort()

    lista=np.array(lista,dtype=float)
    lista=lista/np.max(lista)
    X=np.array(X,dtype=float)
    
    return lista,X


def norm_energ (y,y1,x1,max,min):
    
    y_ener=[]
    
    for i in range(0,len(y)):
        if (y[i]<=y1):
            energ=((y[i]*(x1-max))/y1)+max
            y_ener.append(energ)
        if (y[i]>y1):
            energ=((y[i]-y1)/(1-y1))*(min-x1)+x1
            y_ener.append(energ)
    y_ener=np.array(y_ener,dtype=float)
    
    return y_ener



def rsquared(y_pred,y):
    X = sm.add_constant(y_pred)
    mod = sm.OLS(y,X).fit()
    R2 = mod.rsquared_adj
    return R2


