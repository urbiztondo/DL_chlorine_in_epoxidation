# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 08:56:25 2023

@author: urbiz
"""


import numpy as np
import matplotlib.pyplot as plt




# Carga la matriz desde el archivo .npy
energ = np.load('EnerAgClCuO_05.npy')

print(energ.shape)  # Debería imprimir (5, 25, 25)

energ_selected = 0
repetitions = (1,1)
tiled_energ = np.tile(energ[energ_selected],repetitions)

#tiled_energ = np.flipud(tiled_energ)


row, col = np.where(energ[0] == np.max(energ[0]))
print ("fila: ",row,"columna: ",col)


# Crea una figura y un eje
fig, ax = plt.subplots()

# Define el mapa de colores, por ejemplo, 'plasma', 'viridis', 'inferno', etc.
cmap = 'viridis'

# Establece los límites del rango de colores
vmin = np.min(tiled_energ)
vmax = np.max(tiled_energ)

# Muestra la matriz como un mapa de colores en 2D
cax = ax.imshow(tiled_energ, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1])

# Opcional: Agrega una barra de colores
num_values = 10
cbar = fig.colorbar(cax, ticks=np.linspace(vmin,vmax,num_values))

# Establece 10 separaciones en los ejes x e y
xticks = np.linspace(0, 1, 11)
yticks = np.linspace(0, 1, 11)
ax.set_xticks(xticks)
ax.set_yticks(yticks)

# Guarda la figura en la carpeta de trabajo con alta resolución (300 dpi)
plt.savefig('heatmap_AgClCuO_05_'+str(energ_selected)+'.png', dpi=300, bbox_inches='tight')
# Muestra la gráfica
plt.show()


