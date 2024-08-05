"""
In this script, the Isomap embedding is calculated from the dissimilarity matrix gotten in
'get_dissim_matrix.py'. The number of neigbors is set after analysis in the 'isomap_choose_k.ipynb'.
A dictionary cantaining the sulci points is retrieved with each sulcus aligned to the sulcus 
whose distance to all others was minimum. 

"""

import numpy as np
import pandas as pd
from sklearn import manifold
import nibabel as nib
import utils
import icp
import os

# parameters
N_sulci = 606
tag = "eucl_23_new_reconst" # options: hausd / eucl / wasssqeu
tag_loop = 'eucl_flip' # tag
 
# Isomap parameters
dim = 10
neig = 50
parameters_name = f"dim{dim}_k{neig}"

# Base directory
dir_base_local = f"/home/marlenesga/Documents/output/dim{dim}_k{neig}_{tag}"

# Output filenames
isomap_filename = f"{dir_base_local}/isomap_matrix_{parameters_name}_{tag}.npy"
dir_compon_results = f"{dir_base_local}/components_isomap_{parameters_name}_{tag}"
dir_neutral_dic_filename = f"{dir_base_local}/sulci_aligned.npy"

# Load paths to sulci (left and right segmentations)
paths_filename_in = "./output/complete_dataset/paths_all_criteria_star.txt"
paths =  pd.read_csv(paths_filename_in, sep=',', header=None)
path = np.array(paths)

# Load data with QA tags
data_parameteriz = pd.read_pickle("./output/data_criteria123_param.pkl")

# Load dissimilarity matrix and visualize
diss_min_filename = f"{dir_base_local}/dissim_matrix_min_{tag}.npy"
dissim_matrix_min = np.load(diss_min_filename)
dissimilarity_matrix = dissim_matrix_min

##############################################################
############ remove problematic segmentations ################

# remove sulci with different shapes
indices_out = [index for index in range(len(dissimilarity_matrix)) if (data_parameteriz.at[index, "outlier_diff_shape"] == 1.0)]

# remove sulci with QA = 1
indices_vis = [index for index in range(len(dissimilarity_matrix)) if data_parameteriz.at[index, "vis_QA"] == 1.0]
indices = [ indices_out + indices_vis ][0]

#############################################################
############# ISOMAP EMBEDDING CALCULATION ##################

input_isomap = dissimilarity_matrix

# ISOMAP w precomputed ######
iso=manifold.Isomap(n_neighbors=neig, n_components=dim, metric='precomputed')
isomap_matrix_precomp =iso.fit_transform(input_isomap)

utils.visualize_dissimilarity(isomap_matrix_precomp, 'isomap_matrix')

""" SAVE """

if not os.path.exists(dir_compon_results):
    os.makedirs(dir_compon_results)
    print(f"Directory '{dir_compon_results}' created.")


with open(isomap_filename, 'wb') as f:
    np.save(f, isomap_matrix_precomp)

#############################################################
############# GET NEUTRAL POSITIONED SULCI ##################

# open data
path = np.delete(path, indices, 0)

# Sulcus with less distance from rest
sum_row = []
for row in dissimilarity_matrix:
    sum_row.append(np.sum(row))

min_distance_row = np.min(sum_row)
sulcus_neutral_pos = sum_row.index(min_distance_row)
print('most neutral position:', sulcus_neutral_pos)


path_cs_1 = path[sulcus_neutral_pos] + '.nii.gz'
tmp = nib.load(path_cs_1[0])
cs_1 = tmp.get_fdata()
indices_1 = np.argwhere(cs_1)

cs_reg_to_neutral = dict()

for i in range(N_sulci):

    print('aligning sulci to most neutral sulcus')

    print(sulcus_neutral_pos,',',i)

    # See target_binary
    path_cs_2 = path[i][0] + '.nii.gz'
    tmp_2 = nib.load(path_cs_2)
    cs_2 = tmp_2.get_fdata()

    if ((("RSulci" in path_cs_1) and ("LSulci" in path_cs_2)) or (("LSulci" in path_cs_1) and ("RSulci" in path_cs_2))):
        # mirror right sulcus to match the left
        cs_2_flip = np.flip(cs_2, 2)

        cs_2 = cs_2_flip

    #Pairwise Registration Iterative Closest Point
    indices_2 = np.argwhere(cs_2)

    indices_2_align, rot, tra, dist = icp.align_pc_pair(indices_2, indices_1)

    cs_reg_to_neutral[i] = indices_2_align


with open(dir_neutral_dic_filename, 'wb') as f:
    np.save(f, cs_reg_to_neutral)

print("script complete")