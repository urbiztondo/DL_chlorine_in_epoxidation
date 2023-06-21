# DL_chlorine_in_epoxidation
Deep learning-based energy mapping of chlorine effects in epoxidation reaction catalyzed by silver-copper oxide nanocatalyst (Data)

In the folder 1_DFT_DATA_posit_energ, you will find data on various positions and energies from the DFT calculations, separated into corresponding folders for each system: AgClCuO, EPAgClCuO, and ADAgClCuO. There is also a folder with the suffix _far, which was used as training data for points somewhat further from the equilibrium point. Functions code is also included.

In the 2_NNs_CODE folder, you will find the codes used for training the optimized and concatenated networks. You need to adapt the source data path and the number of atoms for each case.

The results of the network training are in the 3_optimized_NN_results folder.

In the 4_surface_energy_map_CODE folder, you will find the codes to generate proposed positions from a real position (posit_gener_xxx), those that calculate the energy matrix (energ_xxx), and the one that draws the graphs (energies_plotting).
