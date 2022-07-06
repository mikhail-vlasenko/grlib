import numpy as np


def best_fit_transform(cloud1: np.array, cloud2: np.array):
    """
    Calculates the least-squares best-fit transform that maps
    corresponding points from cloud1 to cloud2 in M spatial dimensions
    :param cloud1: NxM numpy array of points
    :param cloud2: NxM numpy array of points
    :return: a tuple of
        (M+1)x(M+1) homogeneous transformation matrix
        MxM rotation matrix
        Mx1 translation vector
    """

    if cloud1.shape != cloud2.shape:
        raise ValueError

    # get number of dimensions
    m = cloud1.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(cloud1, axis=0)
    centroid_B = np.mean(cloud2, axis=0)
    AA = cloud1 - centroid_A
    BB = cloud2 - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))


def corresponding_landmark(src, dst):
    """
    Distances to corresponding points.
    :param src: points to be
    :param dst:
    :return:
    """
    distances = np.empty(src.shape[0])
    for i in range(src.shape[0]):
        distances[i] = distance(src[i], dst[i])
    return distances, np.arange(0, src.shape[0])


def point_correspondence_icp(
        to_transform,
        anchor,
        init_pose=None,
        max_iterations=20,
        tolerance=0.001
):
    """
    The Iterative Closest Point method: finds best-fit transform that maps one cloud of point
    onto another.
    :param to_transform: NxM numpy array of source M-dimensional points
    :param anchor: NxM numpy array of destination M-dimensional points
    :param init_pose: (M+1)x(M+1) homogeneous transformation
    :param max_iterations: exit algorithm after max_iterations
    :param tolerance: convergence criteria
    :return: a tuple of:
        the final homogeneous transformation that maps first cloud onto the anchor
        distances between points after the last iteration
    """

    if to_transform.shape != anchor.shape or max_iterations < 1:
        raise ValueError

    # get number of dimensions
    m = to_transform.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, to_transform.shape[0]))
    dst = np.ones((m + 1, anchor.shape[0]))
    src[:m, :] = np.copy(to_transform.T)
    dst[:m, :] = np.copy(anchor.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    distances = np.zeros(m)

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = corresponding_landmark(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        transform_matrix, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(transform_matrix, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    transform_matrix, _, _ = best_fit_transform(to_transform, src[:m, :].T)

    return transform_matrix, distances
