import math
import numpy as np

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from .math_helpers import quat_from_yaw



def make_shape_marker(track, map_frame, stamp, marker_id):
    marker = Marker()
    marker.header.frame_id = map_frame
    marker.header.stamp = stamp
    marker.ns = "object_tracks"
    marker.id = marker_id
    marker.action = Marker.ADD

    marker.pose.position.x = float(track.pose[0])
    marker.pose.position.y = float(track.pose[1])
    marker.pose.position.z = float(track.pose[2] + 0.5 * track.extent[2])

    if track.asymmetric:
        marker.type = Marker.CUBE

        qx, qy, qz, qw = quat_from_yaw(track.pose[3])
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        marker.scale.x = float(track.extent[0])
        marker.scale.y = float(track.extent[1])
        marker.scale.z = float(track.extent[2])
    else:
        marker.type = Marker.CYLINDER
        marker.pose.orientation.w = 1.0

        diameter = float(max(track.extent[0], track.extent[1]))
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = float(track.extent[2])

    marker.color.r = 0.2
    marker.color.g = 0.8
    marker.color.b = 0.2
    marker.color.a = float(track.prob_existence)
    return marker


def make_position_gaussian_marker(track, map_frame, stamp, marker_id, sigma_scale=4.0, rings=15, segments=32, height_scale=0.2):
    # Get 2D position covariance (xy)
    pos_xy_cov = track.pose_cov[:2, :2].copy()
    pos_xy_cov = 0.5 * (pos_xy_cov + pos_xy_cov.T) # Make symmetric
    
    # check if actual numbers
    if not np.isfinite(pos_xy_cov).all():
        return None
    
    # Get eigenvectors = variances and direction of ellipsioid 
    variances, axis_directions = np.linalg.eig(pos_xy_cov)

    # get standard deviations
    sigma_minor = math.sqrt(max(1e-9, float(variances[0])))
    sigma_major = math.sqrt(max(1e-9, float(variances[1])))

    # center of track
    center_x = float(track.pose[0])
    center_y = float(track.pose[1])
    base_z = float(track.pose[2] + 0.01) # z position

    points = []
    grid = []
    for i in range(rings):
        # r is the number of standard deviations from the center
        r = (i / max(1, (rings - 1))) * sigma_scale 
        
        # Gaussian height is a function of r
        mahalanobis_sq = r**2
        gaussian_height = height_scale * math.exp(-0.5 * mahalanobis_sq)
        
        ring_points = []
        for j in range(segments):
            theta = (j / segments) * 2.0 * math.pi
            
            u = r * sigma_minor * math.cos(theta)
            v = r * sigma_major * math.sin(theta)
            
            # Rotate into world coordinates using eigenvectors
            local_xy = axis_directions @ np.array([u, v])
            
            pt = Point()
            pt.x = center_x + local_xy[0]
            pt.y = center_y + local_xy[1]
            pt.z = base_z + gaussian_height
            ring_points.append(pt)
            
        grid.append(ring_points)

    # Build the top surface triangles
    for i in range(rings - 1):
        for j in range(segments):
            j_next = (j + 1) % segments

            p1 = grid[i][j]
            p2 = grid[i][j_next]
            p3 = grid[i + 1][j]
            p4 = grid[i + 1][j_next]

            points.extend([p1, p2, p3])
            points.extend([p3, p2, p4])

    # Create the vertical skirt
    outer_ring = grid[-1]
    for j in range(segments):
        j_next = (j + 1) % segments

        top1 = outer_ring[j]
        top2 = outer_ring[j_next]

        bot1 = Point(x=top1.x, y=top1.y, z=base_z)
        bot2 = Point(x=top2.x, y=top2.y, z=base_z)

        points.extend([top1, top2, bot1])
        points.extend([bot1, top2, bot2])

    # Create the flat bottom cap
    center_bot = Point(x=center_x, y=center_y, z=base_z)
    for j in range(segments):
        j_next = (j + 1) % segments
        bot1 = Point(x=outer_ring[j].x, y=outer_ring[j].y, z=base_z)
        bot2 = Point(x=outer_ring[j_next].x, y=outer_ring[j_next].y, z=base_z)
        points.extend([bot1, bot2, center_bot])

    marker = Marker()
    marker.header.frame_id = map_frame
    marker.header.stamp = stamp
    marker.ns = "position_gaussian"
    marker.id = marker_id
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD

    marker.pose.orientation.w = 1.0
    marker.points = points

    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    marker.color.r = 0.1
    marker.color.g = 0.4
    marker.color.b = 1.0
    marker.color.a = 0.28

    return marker

def make_yaw_gaussian_marker(track, map_frame, stamp, marker_id, sigma_scale=4.0, samples=30, radius_scale=1.0, height_scale=0.20, max_display_angle=math.radians(89.0)):
    if not track.asymmetric:
        return None

    yaw_var = max(1e-9, float(track.pose_cov[3, 3]))
    yaw_sigma = math.sqrt(yaw_var)

    if yaw_sigma < 1e-6:
        return None

    yaw = float(track.pose[3])

    angle_width = sigma_scale * yaw_sigma
    angle_width = min(angle_width, max_display_angle)

    radius = radius_scale * max(float(track.extent[0]), float(track.extent[1]))
    base_z = float(track.pose[2] + track.extent[2] + 0.05)

    center_x = float(track.pose[0])
    center_y = float(track.pose[1])

    angle_min = -angle_width
    angle_max = angle_width

    points = []

    for i in range(samples - 1):
        t0 = i / max(1, samples - 1)
        t1 = (i + 1) / max(1, samples - 1)

        angle_offset_0 = angle_min + t0 * (angle_max - angle_min)
        angle_offset_1 = angle_min + t1 * (angle_max - angle_min)

        height_0 = height_scale * math.exp(-0.5 * (angle_offset_0 / yaw_sigma)**2)
        height_1 = height_scale * math.exp(-0.5 * (angle_offset_1 / yaw_sigma)**2)

        angle_0 = yaw + angle_offset_0
        angle_1 = yaw + angle_offset_1

        base_0 = Point()
        base_0.x = center_x + radius * math.cos(angle_0)
        base_0.y = center_y + radius * math.sin(angle_0)
        base_0.z = base_z

        base_1 = Point()
        base_1.x = center_x + radius * math.cos(angle_1)
        base_1.y = center_y + radius * math.sin(angle_1)
        base_1.z = base_z

        top_0 = Point()
        top_0.x = base_0.x
        top_0.y = base_0.y
        top_0.z = base_z + height_0

        top_1 = Point()
        top_1.x = base_1.x
        top_1.y = base_1.y
        top_1.z = base_z + height_1

        points.extend([base_0, base_1, top_0])
        points.extend([top_0, base_1, top_1])

    marker = Marker()
    marker.header.frame_id = map_frame
    marker.header.stamp = stamp
    marker.ns = "yaw_gaussian"
    marker.id = marker_id
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD

    marker.pose.orientation.w = 1.0
    marker.points = points

    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    marker.color.r = 1.0
    marker.color.g = 0.75
    marker.color.b = 0.1
    marker.color.a = 0.35

    return marker


def make_extent_cov_marker(track, map_frame, stamp, marker_id, sigma_scale=1.0):
    extent_cov = track.extent_cov.copy()
    extent_cov = 0.5 * (extent_cov + extent_cov.T)

    sigma_x = sigma_scale * math.sqrt(float(extent_cov[0, 0]))
    sigma_y = sigma_scale * math.sqrt(float(extent_cov[1, 1]))
    sigma_z = sigma_scale * math.sqrt(float(extent_cov[2, 2]))

    marker = Marker()
    marker.header.frame_id = map_frame
    marker.header.stamp = stamp
    marker.ns = "extent_covariance"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x = float(track.pose[0])
    marker.pose.position.y = float(track.pose[1])
    marker.pose.position.z = float(track.pose[2] + 0.5 * track.extent[2])

    if track.asymmetric:
        qx, qy, qz, qw = quat_from_yaw(track.pose[3])
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
    else:
        marker.pose.orientation.w = 1.0

    marker.scale.x = float(track.extent[0] + 2.0 * sigma_x)
    marker.scale.y = float(track.extent[1] + 2.0 * sigma_y)
    marker.scale.z = float(track.extent[2] + 2.0 * sigma_z)

    marker.color.r = 1.0
    marker.color.g = 0.3
    marker.color.b = 0.2
    marker.color.a = 0.12
    return marker


def make_text_marker(track, map_frame, stamp, marker_id):
    marker = Marker()
    marker.header.frame_id = map_frame
    marker.header.stamp = stamp
    marker.ns = "object_track_labels"
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD

    marker.pose.position.x = float(track.pose[0])
    marker.pose.position.y = float(track.pose[1])
    marker.pose.position.z = float(track.pose[2] + track.extent[2] + 0.2)

    marker.scale.z = 0.045

    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 0.55

    marker.text = (f"{track.class_name}={track.class_conf:.2f}\np_e={track.prob_existence:.2f}")

    return marker