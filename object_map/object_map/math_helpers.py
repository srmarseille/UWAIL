import math
import numpy as np



def normalize_angle(angle):
    """
    normalize an angle to [-pi, pi).
    This keeps yaw values bounded after subtraction or Kalman updates.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a, b):
    """smallers (singed) angle between a and b"""
    return normalize_angle(a - b)


def yaw_from_quat(x, y, z, w):
    # gets yaw from quaternion
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)



def quat_from_yaw(yaw):
    # makes quaternion from yaw
    half = 0.5 * yaw
    qz = math.sin(half)
    qw = math.cos(half)
    return 0.0, 0.0, qz, qw



def quat_to_rot(x, y, z, w):
    # convert quaternion to rotation matrix [3x3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array([[1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                    [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                    [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],], dtype=float)


def transform_point(transform_matrix, point):
    # transforms a point from source frame to target frame using 4x4 transformation matrix
    point_h = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    point_camera_h = transform_matrix @ point_h
    return point_camera_h[:3]

def transform_msg_to_matrix(transform):
    # make homogenous transformation matrix from ros tf message
    tx = transform.translation.x
    ty = transform.translation.y
    tz = transform.translation.z

    qx = transform.rotation.x
    qy = transform.rotation.y
    qz = transform.rotation.z
    qw = transform.rotation.w

    rot = quat_to_rot(qx, qy, qz, qw)

    T = np.eye(4)
    T[:3, :3] = rot # rotation part of T
    T[:3, 3] = [tx, ty, tz] # translation part of T
    return T


def get_fov_from_k(k, image_width, image_height):
    """
    Get camera frustum angles, for visibiltiy only
    - pinhole model equations
    """
    k = np.array(k, dtype=float).reshape(3, 3)

    fx = k[0, 0]
    fy = k[1, 1]

    horizontal_fov = 2.0 * math.atan(image_width / (2.0 * fx))
    vertical_fov = 2.0 * math.atan(image_height / (2.0 * fy))

    return horizontal_fov, vertical_fov

### Kalman filter functions
def kalman_predict(x, P, Q):
    # prediction step, has no effect because objects do no move (Q = I)
    x_pred = x.copy()
    P_pred = P + Q
    x_pred[3] = normalize_angle(x_pred[3])
    return x_pred, P_pred

def kalman_predict_extent(x, P, Q):
    # prediction step, has no effect because objects do not change shape (Q = I)
    return x.copy(), P + Q


def kalman_update_pose(x, P, z, R):
    """
    Kalman pose update fr [x, y, z, yaw]
    """
    H = np.eye(4) # measurement matrix. =I, assumes observations exactly match tracks state

    # innovation: diff between observed and predicted pose
    y = z - H @ x
    y[3] = normalize_angle(y[3]) # yaw normalized 

    # Innovation covariance (in practice P + R, since H=I)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # state update
    x_new = x + K @ y
    x_new[3] = normalize_angle(x_new[3])

    # Joseph form update for numerical stability
    I = np.eye(4)
    P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    return x_new, P_new

def kalman_update_extent(x, P, z, R):
    """
    Same as pose, but no angle normalizing
    - x = current estimate [x, y, z, yaw]
    - P = current covariance 3x3
    - z = observed extent [x, y, z]
    - R = observed covariance 3x3
    H = identity matrix
    """
    H = np.eye(3) # again, H=I assumes observation state matches track state

    # innovation
    y = z - H @ x

    # innovation covariance, in practice P + R
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # state update
    x_new = x + K @ y

    # Joseph form covariance update
    I = np.eye(3)
    P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    return x_new, P_new


def mahalanobis_distance_xyz(x, P, z, R):
    x_xyz = x[:3]
    z_xyz = z[:3]
    P_xyz = P[:3, :3]
    R_xyz = R[:3, :3]

    y = z_xyz - x_xyz
    S = P_xyz + R_xyz
    return y.T @ np.linalg.inv(S) @ y




def clamp_probability(value):
    # clamp value between, used in track update for p_existence
    if value < 1e-4:
        return 1e-4
    if value > 1.0 - 1e-4:
        return 1.0 - 1e-4
    return value


def get_bbox_corners_map(track):
    # for visibility check, get bbox corners in map frame
    # bbox is in local object frame. This transforms to map frame to check if corners are inside frustum for visibility check
    x = float(track.pose[0])
    y = float(track.pose[1])
    z = float(track.pose[2])
    yaw = float(track.pose[3])

    length = float(track.extent[0])
    width = float(track.extent[1])
    height = float(track.extent[2])

    half_length = 0.5 * length
    half_width = 0.5 * width

    # 8 corners in local object frame
    # first 4 are bottom footprint, last 4 are top
    corners_local = [[ half_length,  half_width, 0.0],
                    [ half_length, -half_width, 0.0],
                    [-half_length, -half_width, 0.0],
                    [-half_length,  half_width, 0.0],
                    [ half_length,  half_width, height],
                    [ half_length, -half_width, height],
                    [-half_length, -half_width, height],
                    [-half_length,  half_width, height]]

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    corners_map = []

    for corner in corners_local:
        # for each cornen, simple trig, to rotate them by yaw
        local_x = corner[0]
        local_y = corner[1]
        local_z = corner[2]

        map_x = x + cos_yaw * local_x - sin_yaw * local_y
        map_y = y + sin_yaw * local_x + cos_yaw * local_y
        map_z = z + local_z # Yaw does not affect height, so z is only shifted by the local corner height.

        corners_map.append([map_x, map_y, map_z])

    return corners_map

