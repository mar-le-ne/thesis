"""
In this script, one Isomap component is visualized via several weighted averaged windows of
sulci or via a sample pick.
The Isomap embedding matrix and aligned sulci is loaded from the results of 
'calc_isomap_and_neutral_pos.py'.

This script is called for individual component in the Isomap embedding matrix via the terminal 
with instructions 'sbatch --array=1-10 <bash_filename>.sh'. The BASH file contains:
a=$(($SLURM_ARRAY_TASK_ID))
python3 -u isomap_wind_vis_newicp.py $a

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import sys

# Choose Isomap component to analyse
compon = int(sys.argv[1]) # include [1,dim]
print(f"running component {compon}")

# Parameters
tag = "eucl_23" #options: hausd / eucl / wasssqeu

# Number of dimensions/components in the Isomap embedding
dim = 10

# number of neighbors
neig = 11

# Shape of each sulci matrix
cs_shape = np.zeros(((288, 288, 192)))


parameters_name = f"dim{dim}_k{neig}"

# Base directories
dir_base_local = f"/home/marlenesga/Documents/output/dim{dim}_k{neig}_{tag}"
dir_neutral_dic_filename = f"{dir_base_local}/sulci_aligned.npy"
dir_compon_results = f"{dir_base_local}/components_isomap_{parameters_name}_{tag}"

# Dissimilarity matrix source
diss_min_filename = f"{dir_base_local}/dissim_matrix_min_{tag}.npy"

# Isomap matrix source
isomap_filename = f"{dir_base_local}/isomap_matrix_{parameters_name}_{tag}.npy"

# Paths source
paths_filename_in = "./output/complete_dataset/paths_all_criteria_star.txt"

# Output filenames
diss_png_filename = f"{dir_base_local}/dissim_matrix_min_{tag}.png"
isomap_png_filename = f"{dir_compon_results}/isomap_matrix_{parameters_name}_{tag}.png"
distribution_comp_png_filename = f"{dir_compon_results}/component{str(compon)}_distribution_sulci_{tag}.png"
distribution_comp_filename = f"{dir_compon_results}/component{str(compon)}_distribution_sulci_{tag}.npy"
window_comp_png_filename = f"{dir_compon_results}/component{str(compon)}_window_sulci_{tag}.png"
sample_comp_png_filename = f"{dir_compon_results}/component{str(compon)}_sample_sulci_{tag}.png"
w_sort_val_filename = f"{dir_compon_results}/component{str(compon)}_w_sort_{tag}.npy"
w_idx_list_filename = f"{dir_compon_results}/component{str(compon)}_w_idx_{tag}.npy"


# Load isomap matrix and visualize
isomap_matrix = np.load(isomap_filename)
plt.figure()
plt.imshow(isomap_matrix, interpolation='nearest', aspect=0.05)
plt.colorbar(); plt.ylabel('sulci'); plt.xlabel('dimensions')
plt.savefig(isomap_png_filename)

# Load dissimilarity matrix and visualize
dissim_matrix_min = np.load(diss_min_filename)
dissimilarity_matrix = dissim_matrix_min
plt.figure()
plt.imshow(dissimilarity_matrix, interpolation='nearest')
plt.colorbar(); plt.ylabel('sulci')
plt.savefig(diss_png_filename)
plt.show()


# Load all sulci aligned to most neutral position sulcus
cs_reg_to_neutral = np.load(dir_neutral_dic_filename, allow_pickle=True).item()

# Number of sulci count
N_sulci = np.size(dissimilarity_matrix,0)

# Assign component
isomap_comp = isomap_matrix[:,compon-1]


###########################
#        Sort Component 
###########################

# Use enumerate to assign the order to each element
assigned_orders = list(enumerate(isomap_comp, start=0))  # starting index from 0

# Sort the assigned orders based on the original numbers
sorted_orders = sorted(assigned_orders, key=lambda x: x[1])

sort_index = []
for values, keys in sorted_orders:
    sort_index.append(values)

sort_values = sorted(isomap_comp)

#######################################################
#        Compute averaged sulcus in each window 
#######################################################

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))


n_windows_wanted = 14
n_divisions = n_windows_wanted+1

windows_division = np.linspace(sort_values[0], np.ceil(sort_values[-1]), n_divisions)

start = 0
sulcus_window_all = dict()
w_avg_comp = []
num = 0

for index, value in enumerate(windows_division):

    print('New window')
    if index == (n_windows_wanted):
        break

    # Which idx belong to this window
    w_idx = [i for i, val in enumerate(sort_values) if ( windows_division[index] <= sort_values[i] < windows_division[index+1]  )]

    # Idx of the sulci belonging to this window
    w_sort_idx = [sort_index[i] for i in w_idx] 
    # Component values of the sulci belonging to this window
    w_sort_val = [sort_values[i] for i in w_idx]

    x_values = np.linspace(windows_division[index], windows_division[index+1], 1000)
    
    # Parameters for the Gaussian function
    mu = np.mean(x_values)
    fwhm = 0.25 
    sigma = fwhm/ (2*np.sqrt(2*np.log(2))) #0.1

    if w_idx == []:
        sulcus_window_all[num] = np.zeros_like(cs_shape)
        num += 1
        w_avg_comp.append(mu)
        continue
    
    # Calculate Gaussian values
    gaussian_values = gaussian(x_values, mu, sigma)

    sulcus_window = np.zeros_like(cs_shape)

    for i, val in enumerate(w_sort_idx):

        x_values_round = list(np.int16(x_values*100))
       
        compon_val_round = np.int16(w_sort_val[i]*100)

        gau_idx = x_values_round.index( compon_val_round )

        cs_neut = np.zeros(((288, 288, 192)))
        x = cs_reg_to_neutral[val][:,0]
        x = x.astype(int)
        y = cs_reg_to_neutral[val][:,1]
        y = y.astype(int)
        z = cs_reg_to_neutral[val][:,2]
        z = z.astype(int)
        cs_neut[x, y, z] = 1

        sulcus_window += (cs_neut * gaussian_values[gau_idx] )

    # normalize
    sulcus_window_norm = (sulcus_window - np.min(sulcus_window)) / (np.max(sulcus_window) - np.min(sulcus_window))

    # add windowed sulcus to dict
    sulcus_window_all[num] = sulcus_window_norm

    # add average valued of the component within the window
    w_avg_comp.append(mu)

    num += 1


########################################################
#    SMOOTH - Convolve summed sulcus with a Gaussian 
########################################################

sulcus_window_gau = dict()

for i in range( n_windows_wanted ):
    
    # Convolve with a 3D Gaussian -  performs convolution of summed sulcus with a Gaussian kernel
    sulcus_conv  = gaussian_filter(sulcus_window_all[i], sigma=2)

    threshold = 0.07
    # Thresholding
    sulcus_conv[sulcus_conv < threshold] = 0

    indices = np.where(sulcus_conv)
    sulcus_window_gau[i] = indices


###########################################################
#    Visualize sulci in convolved window - ONLY ONE PLOT
###########################################################

# Create a figure
fig = plt.figure(figsize=(15, 4))
# Define a GridSpec with tight layout
gs = fig.add_gridspec(3, n_windows_wanted, wspace=0.0001, hspace=0.0001)

norm = matplotlib.colors.Normalize(vmin=0, vmax=1000)
for j in range(2):
    for i in range(n_windows_wanted):
        ax = fig.add_subplot(gs[j, i], projection='3d')

        indices = sulcus_window_gau[i]

        ax.scatter(indices[0], indices[1], indices[2], s=1/200, c= indices[2], cmap='bwr', norm = norm, marker='o') #z,

        # Set view to look at
        if j==0:
            ax.view_init(elev=61, azim=-130, roll=0)
        else:
            ax.view_init(elev=25, azim=-75, roll=50)

        ax.grid(visible=False)
        ax.set_xticks([])
        ax.axis('off')

col = ['#2a1fd0', '#1FD082', '#030F4F', '#F6D04D', '#FC7634', '#F7BBB1', '#a6a6a6', '#E83F48', '#008835', '#79238E', '#8e7923', '#1fc6d0', '#e12378', '#d7e123']

ax = fig.add_subplot(gs[2, 0:n_windows_wanted])
w_sort_val_list = []; w_idx_list = []

for index, value in enumerate(windows_division):
    if index == (n_windows_wanted):
        break
    w_idx = [i for i, val in enumerate(sort_values) if ( windows_division[index] <= sort_values[i] < windows_division[index+1]  )]
    w_idx_list.append(w_idx)
    w_sort_val = [sort_values[i] for i in w_idx]
    w_sort_val_list.append(w_sort_val)

    yaxis = np.zeros([1,len(w_sort_val)])
    label_name = "w" + str(index+1)
    ax.scatter(w_sort_val, yaxis, label=label_name, c= col[index])
    ax.grid(visible=False)
    ax.set_yticks([])
    ax.spines[['right', 'top', 'left']].set_visible(False)

plt.legend(ncols=n_windows_wanted, bbox_to_anchor=(1, -0.5), columnspacing=0.1)
plt.xlabel(f"Component {compon}")
plt.savefig(window_comp_png_filename, bbox_inches='tight')
plt.show()


#########################################################
#               Visualize sampled sulci 
#########################################################

# Create a figure
fig = plt.figure(figsize=(15, 3))
sample_sulci = sort_index[0::25]
num_masks = len(sample_sulci)
# Define a GridSpec with tight layout
gs = fig.add_gridspec(2, num_masks, wspace=0.0001, hspace=0.0001)

for j in range(2):
    # Plot each group with a specified shift
    for i,val in enumerate(sample_sulci):
        ax = fig.add_subplot(gs[j, i], projection='3d')

        x = cs_reg_to_neutral[val][:,0]; x = x.astype(int)
        y = cs_reg_to_neutral[val][:,1]; y = y.astype(int)
        z = cs_reg_to_neutral[val][:,2]; z = z.astype(int)

        sc = ax.scatter(x, y, z, s=1/50, c='#79238E', marker='o')

        # Set view to look at
        if j==0:
            ax.view_init(elev=61, azim=-130, roll=0)
        else:
            ax.view_init(elev=25, azim=-75, roll=50)

        ax.grid(visible=False)
        ax.set_xticks([])
        ax.axis('off')
        
plt.savefig(sample_comp_png_filename)
plt.show()