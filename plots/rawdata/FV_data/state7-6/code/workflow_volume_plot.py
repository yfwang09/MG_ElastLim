import numpy as np
import matplotlib.pyplot as plt

# Set up the file path.
input_dir = 'G:/MLmat/Data/free_volume_examples/events/1-output/vol_hist_8refs_r3.8_satom/'
Fig_dir = 'G:/MLmat/Data/free_volume_examples/events/Fig-voro/'
mises_crit = 0.01

temps = [2, 5]
strains_list = [[0.26, 0.47, 0.59, 0.70, 0.90],
               [0.05, 0.44, 0.53, 0.70, 0.89]]

frames_list = [[665, 1177, 1479, 1758, 2267],
              [141, 1078, 1325, 1810, 2115]]

for i in range(len(temps)):
    temp = temps[i]
    strains = strains_list[i]
    frames = frames_list[i]
    for j in range(len(strains)):
        strain = strains[j]
        frame = frames[j]
        keyword = 'T%dK_%.2f_frame_%d' % (temp, strain, frame)
        print('file:', keyword)

        atom_feature_file_init = input_dir + 'coords_density_atoms_%s_0.npz' % keyword
        atom_feature_file_final = input_dir + 'coords_density_atoms_%s_15.npz' % keyword
        atom_feature_init = np.load(atom_feature_file_init)['arr_0']
        atom_feature_final = np.load(atom_feature_file_final)['arr_0']

        atom_Ave_FreeVol = atom_feature_init['Ave_FreeVol']
        atom_atomic_strain = atom_feature_final['atomic strain']
        id_satom = atom_atomic_strain >= mises_crit
        id_Nonsatom = atom_atomic_strain < mises_crit

        satom_Ave_FreeVol = atom_Ave_FreeVol[id_satom]
        satom_atomic_strain = atom_atomic_strain[id_satom]
        Nonsatom_Ave_FreeVol = atom_Ave_FreeVol[id_Nonsatom]
        Nonsatom_atomic_strain = atom_atomic_strain[id_Nonsatom]

        # Visualize the whole distribution
        labelsize = 20
        fontsize = 16
        fig = plt.figure(figsize=(12, 4))

        #####
        plt.subplot(1, 2, 1)
        plt.scatter(Nonsatom_Ave_FreeVol, Nonsatom_atomic_strain, color='blue')
        plt.scatter(satom_Ave_FreeVol, satom_atomic_strain, color='red')
        plt.xlim((0.1, 0.7))
        plt.ylim((0, 0.07))
        ##plt.title('Histogram of activated volume', fontsize=labelsize)
        plt.xlabel('atom_Ave_FreeVol (Angstrom^3)', fontsize=labelsize)
        plt.ylabel('atom_atomic_strain', fontsize=labelsize)
        plt.tick_params(labelsize=fontsize)
        plt.legend(['matrix', 'stz'], loc='upper left', fontsize=fontsize)

        #######
        # define the fig
        plt.subplot(1, 2, 2)
        plt.hist(Nonsatom_Ave_FreeVol, bins=60, range=(0.1, 0.7), density=True, facecolor='blue', alpha=0.8)
        plt.hist(satom_Ave_FreeVol, bins=60, range=(0.1, 0.7), density=True, facecolor='red', alpha=0.8)
        plt.xlim((0.1, 0.7))

        # plt.title('Histogram of activated volume', fontsize=labelsize)
        plt.xlabel('ave activated volume (Angstrom^3)', fontsize=labelsize)
        plt.ylabel('Probability', fontsize=labelsize)
        plt.tick_params(labelsize=fontsize)
        plt.legend(['matrix', 'stz'], loc='upper left', fontsize=fontsize)
        plt.tight_layout()
    plt.show()