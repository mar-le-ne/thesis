"""
The iterative closest point algorithm in this script is sourced from
 https://github.com/neurospin/point-cloud-pattern-mining/blob/main/pcpm/distance/core.py
"""

## Imports
import numpy as np

def _calc_closest_point(a1: np.ndarray, a2: np.ndarray):
    
    """Calculate the closest points of a2 relative to the points of a1.

    The shape of the array is interpreted as DxN where D is the
    number of the point's coordinates and N the number of points.

    :param a1: array against which the distance are calculated
    :type a1: DxN np.ndarray
    :param a2: second array
    :type a2: DxM np.ndarray
    :return: closest points coodrinates (DxN),
        closest points indices in a2 (1xN),
        average distance (scalar)
    :rtype: tuple
    """

    # Calculate the square distance of each point of a2 with all points of a1
    # NOTE: this is faster than scipy.spatial's distances
    # 1) calculate the square-difference of the coordinates
    dist = ((a2.T[..., None] - a1)**2)  # shape = (a2-n-points, 3, a1-n-points)
    # 2) sum the square differences of the corrdinate along the coordinates axis
    #    get the square-distance matrix
    dist = dist.sum(axis=1)  # shape = (a2-n-points, a1-n-points)

    # the distance of the closest points of a2 for each point in a1
    # ! OLD : self.minDist[:, 1] = dist.min(axis=0)
    closest_points_distances = dist.min(axis=0)  # shape (1, a1-n-points)

    # get the index of the closest point of a2 to each point of a1
    # ! OLD : self.minDist[:, 0] = indices
    closest_points_idx = dist.argmin(axis=0)  # shape = (1, a1-n-points)

    # TODO: the min is calculated twice in the previous expressions, it could be done only once

    # the coordinates of the points of a2 which are closest to a1
    # ! OLD : self.curClose = self.model[:, indices]
    closest_points = a2[:, closest_points_idx]

    # average distance
    # ! OLD : self.curDist = self.minDist[:,1].sum()/self.numPtr
    av_dist = closest_points_distances.mean()

    return closest_points, av_dist

def transform_datapoints(data_points: np.ndarray, dxyz=None, rotation_matrix=None, translation_vector=None, flip: bool = False) -> np.ndarray:
    """Transform the data_points.

    The datapoint are scaled according to dxyz, then rotated with rotation_matrix and
    translated by translation_vector.

    If flip is True, the x coordinates are inverted (x --> -x)
    """

    dp = data_points.copy()

    if dxyz is not None:
        # rescale
        dp *= dxyz

    if (rotation_matrix is not None) and not np.array_equal(rotation_matrix, np.eye(3)):
        # rotate
        # DO NOT USE SCIPY ROTATION HERE, It does not work as expected.
        # found a problem with coarse_PCA rotation

        dp = (rotation_matrix@dp.T).T

    if (translation_vector is not None) and not np.array_equal(translation_vector, np.zeros(3)):
        # translate
        dp += translation_vector

    if flip:
        # invert x coordinates
        dp[:,0]=-dp[:,0]
    
    return dp

def _calc_transform(a1: np.ndarray, a2: np.ndarray, closest_points: np.ndarray):
    """Calculate SVD-based rotation and translation that
    transform the points in a1 into those of a2"""

    # TODO this could be taken as fucntion argument, not calcuated every time
    meanModel = a2.mean(axis=1)[:, None]
    meanData = a1.mean(axis=1)[:, None]

    A = a1 - meanData
    B = closest_points - meanModel

    # SVD decomposition
    (U, S, V) = np.linalg.svd(np.dot(B, A.T))
    U[:, -1] *= np.linalg.det(np.dot(U, V))

    rotation_M = np.dot(U, V)
    translation_v = meanModel - np.dot(rotation_M, meanData)

    return rotation_M, translation_v

def icp_python(moving: np.ndarray, model: np.ndarray, max_iter: int = 10, epsilon: float = 0.1):
    """Calculate Iterative Closest Point (ICP) distance.

    Array a1 is iteratively rotated and translated to make it closer
    to array a2.

    Args:
        moving ndarray (N,3): point_cloud
        model ndarray (N,3): reference point_cloud
        max_iter (int, optional): man number of iterations, defaults to 10
        epsilon, float: min distance improvement on one iteration, defaults to 0.1

    Returns:
    (dist,rot,trans) tuple containing
    - the final average distance between closest points after transformation of a1
    - the rotation matrix and the translation vector of the ICP transformation of a1
    """

    a1 = moving.T
    a2 = model.T

    dim_n = a1.shape[0]  # single point dimensionality
    # cumulative rotation matrix and translation vector
    cum_rot = np.eye(dim_n, dim_n, dtype=float)
    cum_tra = np.zeros((dim_n, 1), dtype=float)

    old_dist = 0

    # check that max_iter is valid
    #assert max_iter < np.iinfo(np.int).max
    max_iter = int(max_iter)

    for i in range(max_iter):
        # calculate distance
        pts, dist = _calc_closest_point(a1, a2)
        improvement = abs(old_dist - dist)

        if (improvement <= epsilon):
            break

        # calculate rotation and transformation
        rot, tra = _calc_transform(a1, a2, pts)

        # update the rotation and translation
        a1 = np.dot(rot, a1) + tra
        cum_rot = np.dot(rot, cum_rot)
        cum_tra = np.dot(rot, cum_tra) + tra

        # update the loop variables
        old_dist = dist
        i += 1

    return dist, cum_rot, cum_tra.flatten()

_DEFAULT_DIST_FUNCTION_NAME = "icp_python"


def calc_distance(a1: np.ndarray, a2: np.ndarray,
                  distance_f: str = _DEFAULT_DIST_FUNCTION_NAME, **kwargs):

    distance_f, rot, tran = icp_python(a1, a2, max_iter = 1000, epsilon = 0.01)

    #distance_f = _get_distance_f(distance_f)

    return distance_f, rot, tran 

def align_pc_pair(pc_to_align: np.ndarray, reference_pc: np.ndarray,
                  max_iter=100, epsilon=0.01):
    """Align two point-clouds (uses the default distance function)."""
    # calculate distance
    dist, rot, tra = calc_distance(
        pc_to_align, reference_pc, distance_f=None,
        max_iter=max_iter, epsilon=epsilon)
    # transform the bucket with the ICP rotation matrix and translation vector
    data_points = transform_datapoints(pc_to_align, (1, 1, 1), rot, tra)

    return data_points, rot, tra, dist
