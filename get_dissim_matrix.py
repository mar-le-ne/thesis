"""
In this script, the dissimilarity matrix is computed from the distance rows obtained in 
'calc_dissimilarity.py'; outliers are removed; and the minimum pairwise distance 
between sulci is computed to obtain the final symmetric dissmilarity matrix.

"""

import numpy as np
import pandas as pd
import utils
import os

# parameters
N_sulci = 606
tag = "eucl_23_new_reconst" # options: hausd / eucl / wasssqeu
tag_loop = 'eucl_flip' # tag
 
# isomap parameters
dim = 10
neig = 50 # temporary when run for the first time
parameters_name = f"dim{dim}_k{neig}"

# Base directory
dir_base_local = f"/home/marlenesga/Documents/output/dim{dim}_k{neig}_{tag}"
if not os.path.exists(dir_base_local):
    os.makedirs(dir_base_local)
    print(f"Directory '{dir_base_local}' created.")

# Output filenames
diss_filename = f"{dir_base_local}/dissim_matrix_{tag}.npy"
diss_min_filename = f"{dir_base_local}/dissim_matrix_min_{tag}.npy"

# Load paths to sulci (left and right segmentations)
paths_filename_in = "./output/complete_dataset/paths_all_criteria_star.txt"
paths =  pd.read_csv(paths_filename_in, sep=',', header=None)
path = np.array(paths)

# Load data with QA tags
data_parameteriz = pd.read_pickle("./output/data_criteria123_param.pkl")


#############################################################
################# GET DISSIMILARITY MATRIX ##################

dissimilarity_matrix = np.empty((N_sulci, N_sulci ))

for sulcus in range(N_sulci):
    with open(f"/home/marlenesga/Documents/output/dissimilarity_flat_single_loops_diff_shape100_{tag_loop}/loop_{sulcus}.npy", 'rb') as f:
        dissimilarity_matrix[:,sulcus] = np.load(f)

utils.visualize_dissimilarity(dissimilarity_matrix, 'dissimilarity matrix')

with open(diss_filename, 'wb') as f:
    np.save(f, dissimilarity_matrix)


##############################################################
############ remove problematic segmentations ################

# remove sulci with different shapes
indices_out = [index for index in range(len(dissimilarity_matrix)) if (data_parameteriz.at[index, "outlier_diff_shape"] == 1.0)]

# remove sulci with QA = 1
indices_vis = [index for index in range(len(dissimilarity_matrix)) if data_parameteriz.at[index, "vis_QA"] == 1.0]
indices = [ indices_out + indices_vis ][0]

print(indices, 'length:', print(len(indices)))

diss_mat_diff_shape_row = np.delete(dissimilarity_matrix,  indices, 0)
diss_mat_diff_shape_col = np.delete(diss_mat_diff_shape_row,  indices, 1)

print("diss_mat_diff_shape_col: ",np.shape(diss_mat_diff_shape_col))

dissimilarity_matrix = diss_mat_diff_shape_col

utils.visualize_dissimilarity(dissimilarity_matrix, 'dissimilarity matrix')


#############################################################
################## MIN PAIRWISE DISTANCE ####################

# update N_sulci
N_sulci = diss_mat_diff_shape_col.shape[0] #N_sulci - len(indices)

# choose the smallest pairwise distance
# such as d (A,B) = min d (d A->B, d B->A)
dissim_matrix_min = dissimilarity_matrix.copy()

for i in range(N_sulci):
    for j in range(N_sulci):
        if dissim_matrix_min[i][j] < dissim_matrix_min[j][i]:
            dissim_matrix_min[j][i] = dissim_matrix_min[i][j]
        else:
            dissim_matrix_min[i][j] = dissim_matrix_min[j][i]

print("dissim_matrix_min: ",np.shape(dissim_matrix_min))

utils.visualize_dissimilarity(dissim_matrix_min, 'dissim_matrix_min')

# Save
with open(diss_min_filename, 'wb') as f:
    np.save(f, dissim_matrix_min)
