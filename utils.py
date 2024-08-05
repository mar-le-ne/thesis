"""
In this script, support functions are written.

"""

## Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import open3d as o3d
from scipy.ndimage import affine_transform

def visualize_binary_matrix(matrix):
    # Get the indices where the mask is 1
    indices = np.where(matrix)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(indices[0], indices[1], indices[2], c=matrix[indices], cmap='viridis')

    plt.show()

def visualize_two_binary_matrices(matrix1, matrix2):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the voxels for the first binary mask
    indices_1 = np.where(matrix1)
    ax.scatter(indices_1[0], indices_1[1], indices_1[2], c='blue', marker='o', alpha=0.5, label='Binary 1')

    # Plot the voxels for the second binary mask (transformed)
    indices_2 = np.where(matrix2)
    ax.scatter(indices_2[0], indices_2[1], indices_2[2], c='red', marker='o', alpha=0.5, label='Binary 2')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Two 3D Binary Masks')
    ax.legend()

    plt.show()

def visualize_two_3d_indices(indices_1, indices_2):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(aspect = (1,1,1))

    # Plot the voxels for the first binary mask
    ax.scatter(indices_1[0], indices_1[1], indices_1[2], s=1/2,marker='o', alpha=0.5, c = '#2F3EEA',label='Binary 1')

    # Plot the voxels for the second binary mask (transformed)
    ax.scatter(indices_2[0], indices_2[1], indices_2[2], s=1/2, marker='o', alpha=0.5, c= '#E83F48',label='Binary 2')

    # Set view to look at
    ax.view_init(elev=0, azim=120, roll=270)
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    #ax.set_title('Two 3D Binary Masks')
    ax.legend(['CS 1', 'CS 2'])

    plt.show()

def visualize_dissimilarity(matrix, tit='dissimilarity matrix', filename = None):
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    plt.title(tit)
    plt.ylabel('sulci')
    if filename != None:
        plt.savefig(filename)
    plt.show()

def binary_mask_to_point_cloud(binary_mask):
    # Get the coordinates of non-zero elements in the binary mask
    points = np.argwhere(binary_mask)

    # Convert the coordinates to float32 and create a point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))

    return point_cloud

def registration_icp_OLD(binary_mask_1, binary_mask_2):
    # Convert to point cloud to perform ICP and get the correspondent transformation matrix
    # Convert binary masks to point clouds
    point_cloud_1 = binary_mask_to_point_cloud(binary_mask_1)
    point_cloud_2 = binary_mask_to_point_cloud(binary_mask_2)

    print(type(point_cloud_1))

    # Visualize 1 and 2 clouds
    #point_cloud_1.paint_uniform_color([0, 0.7, 1]); point_cloud_2.paint_uniform_color([0.7, 0, 0.7])
    #o3d.visualization.draw_geometries([point_cloud_1, point_cloud_2])

    # Perform ICP registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        point_cloud_1, point_cloud_2, max_correspondence_distance=20,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)) 

    # Get the transformation matrix as a NumPy ndarray
    transformation_matrix_np = np.array(reg_p2p.transformation)

    # Apply the affine transformation to the 3D binary mask
    binary_mask_2_aligned = affine_transform(binary_mask_2, transformation_matrix_np, order=0, mode='nearest', cval=0, prefilter=False)

    return binary_mask_2_aligned