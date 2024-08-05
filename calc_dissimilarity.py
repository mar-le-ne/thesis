"""
In this script, the rows of the dissimilarity matrix are calculated. Each row represents the
distance between one sulcus to all the others. The distance is calculated here using the 
Wasserstein, the Hausdorff and the Euclidean (with the average minimum distance between pairwise combos)
measures.

Each sulcus NIfTI file is accessed with the 'paths_all_criteria_star.txt' that contains the location
to the segmentation used.

This script is called for each individual sulcus, via the terminal with
instructions 'sbatch --array=0-606 <bash_filename>.sh'. The BASH file contains:

a=$(($SLURM_ARRAY_TASK_ID*1))
b=$(($a+1))
python3 -u get_dissimilarity_new_icp.py $a $b
"""

# Imports
import numpy as np
import pandas as pd
import nibabel as nib
import ot
from scipy.spatial.distance import cdist
import utils
import sys
from scipy.spatial.distance import directed_hausdorff
import icp


def pairwise_regist_and_dissim(path, init, end):

    # Number of sulci
    N_sulci = path.size
                       
    dissimilarity_matrix_flat_wass = []
    dissimilarity_matrix_flat_hausd = []
    dissimilarity_matrix_flat_euc = []

    # One sulcus given by the argvs
    for i in range(init, end):
        
        path_cs_1 = path[i][0] + '.nii.gz'
        print(path_cs_1)
        tmp = nib.load(path_cs_1)
        cs_1 = tmp.get_fdata()

        indices_1 = np.argwhere(cs_1)
        
        # Loop through all sulci
        for j in range(N_sulci):

            print(i,',',j)

            # See target_binary
            path_cs_2 = path[j][0] + '.nii.gz'
            print(path_cs_2)

            tmp_2 = nib.load(path_cs_2)
            cs_2 = tmp_2.get_fdata()

            indices_2 = np.argwhere(cs_2)

            utils.visualize_two_binary_matrices(cs_1, cs_2)

            # Set odd shapes to 100, to flag outliers
            if np.shape(cs_1) != np.shape(cs_2):
                # Most frequent shape (288, 288, 192)
                print('np.shape(cs_1) != np.shape(cs_2)')

                dissimilarity_matrix_flat_wass.append(100)
                dissimilarity_matrix_flat_hausd.append(100)
                dissimilarity_matrix_flat_euc.append(100)
                continue

            ############
            # Flip sulci

            if ((("RSulci" in path_cs_1) and ("LSulci" in path_cs_2)) or (("LSulci" in path_cs_1) and ("RSulci" in path_cs_2))):
                # mirror right sulcus to match the left
                cs_2_flip = np.flip(cs_2, 2)

                cs_2 = cs_2_flip

                indices_2 = np.argwhere(cs_2)

            ################################################
            # Pairwise Registration Iterative Closest Point

            indices_2_align, rot, tra, dist = icp.align_pc_pair(indices_2, indices_1)

            # visualize
            indices_1_vis = indices_1
            indices_2_vis = indices_2_align
            utils.visualize_two_3d_indices(indices_1_vis.T, indices_2_vis.T)

            """ Wasserstein """
            # Compute pairwise distances between coordinates
            distance_matrix = cdist(indices_1, indices_2_align, 'sqeuclidean')
            # Compute discrete distributions based on the distance matrix
            transport_matrix = ot.emd([], [], distance_matrix)
            # all to all, maybe to all to closest, kd3 to find the closest
            wasserstein = np.sum(distance_matrix*transport_matrix)
            print(wasserstein)
            #save
            dissimilarity_matrix_flat_wass.append(wasserstein)

            """ Hausdorff """
            hausd = directed_hausdorff(indices_1, indices_2_align)[0]
            print(hausd)
            #save
            dissimilarity_matrix_flat_hausd.append(hausd)

            """ Euclidean (avg min distances)"""
            dissimilarity_matrix_flat_euc.append(dist)
            print(dist)
            

    with open(f"/home/marlenesga/Documents/output/dissimilarity_flat_single_loops_diff_shape100_wasssqeu_flip/loop_{i}.npy", 'wb') as f:
        np.save(f, dissimilarity_matrix_flat_wass)

    with open(f"/home/marlenesga/Documents/output/dissimilarity_flat_single_loops_diff_shape100_hausd_flip/loop_{i}.npy", 'wb') as f:
        np.save(f, dissimilarity_matrix_flat_hausd)

    with open(f"/home/marlenesga/Documents/output/dissimilarity_flat_single_loops_diff_shape100_eucl_flip/loop_{i}.npy", 'wb') as f:
        np.save(f, dissimilarity_matrix_flat_euc)

''' open data '''
#open sanitized data DRCMR + CFIN from selected raters
paths =  pd.read_csv("./output/complete_dataset/paths_all_criteria_star.txt", sep=',', header=None)
paths = np.array(paths)

init = int(sys.argv[1])
end  = int(sys.argv[2])

""" pairwise registration and dissimilarity matrix """
pairwise_regist_and_dissim(paths,init, end)