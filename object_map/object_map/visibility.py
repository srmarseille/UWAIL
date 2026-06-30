import math

from .math_helpers import transform_point, get_bbox_corners_map, normalize_angle


def point_is_in_camera_fov(point_camera, horizontal_fov, vertical_fov, min_range, max_range):
    """
    Checks if point is in camera fov and range based on horizontal and vertical angles that define camera fov
    """

    x = float(point_camera[0])
    y = float(point_camera[1])
    z = float(point_camera[2])

    # check if point is behind camera
    if z <= 0.0:
        return False

    horizontal_angle = math.atan(x / z)
    vertical_angle = math.atan(y / z)
    distance = math.sqrt(x * x + y * y + z * z)

    # check if point is in range
    if distance < min_range or distance > max_range:
        return False
    
    # check if point is in fov, angle 0 if pointing into camera plane (image frame)
    if abs(horizontal_angle) > horizontal_fov * 0.5:
        return False

    if abs(vertical_angle) > vertical_fov * 0.5:
        return False

    return True

class VisibilityEvaluator:
    """
    Evaluates if track was visible, based on FOV (+ range threshold), and 2d scan points
    - Instantiated once per observation array/map update, since sensor location is same for all observation used in updated
    - = classifier now (with binary outcome), 
    - first checks if unmatched observation is within fov and range, if so, check if space is free (raytracing)
    """
    def __init__(self, T_map_camera, horizontal_fov, vertical_fov, min_range, max_range, T_map_laser=None, scan_msg=None, scan_distance_margin=0.05, debug_vis=True):
        # transform from map to camera, used for frustum check
        self.T_map_camera = T_map_camera 
        # angles and range for fov check
        self.hfov = horizontal_fov
        self.vfov = vertical_fov
        self.min_range = min_range
        self.max_range = max_range

        # LiDAR inputs: scan and T_map_laser for raytracing
        self.scan_msg = scan_msg
        self.scan_margin = scan_distance_margin
        self.T_map_laser = T_map_laser

        self.debug_vis = debug_vis

        self.min_valid_rays_count = 2


    
    def is_within_fov(self, track):
        """
        Returns true if all eight corners of bbox are in fov
        - Uses bbox corners in map frame (using helper in math_helpers)
        - then checks with point_is_in_camera_fov, counts how many are in fov
        """
        corners_map = get_bbox_corners_map(track)

        inside_count = 0
        for corner_map in corners_map:
            corner_cam = transform_point(self.T_map_camera, corner_map)
            if point_is_in_camera_fov(corner_cam, self.hfov, self.vfov, self.min_range, self.max_range):
                inside_count += 1

        if inside_count >= 8:
            return True

        return False


    def angle_is_in_interval(self, angle, start_angle, end_angle):
        """
        Checks if angle is between start_angle and end_angle
        - Handles intervals that cross -pi to pi boundary
        """
        angle = normalize_angle(angle)
        start_angle = normalize_angle(start_angle)
        end_angle = normalize_angle(end_angle)

        if start_angle <= end_angle:
            return start_angle <= angle <= end_angle
        return angle >= start_angle or angle <= end_angle





    def get_object_bearings_from_corners(self, bottom_corners_laser):
        """
        Find left and right angles between laser, and visible vertical edges of bbox
        - The returned interval is used to select only lidar rays that fall within the bbox
        """
        # 1. Find angles between laser, and corners (horizontal (bottom) profile)
        corner_angles = []
        for corner_laser in bottom_corners_laser:
            angle = math.atan2(corner_laser[1], corner_laser[0]) # angle from laser to that corner
            angle = normalize_angle(angle) 
            corner_angles.append(angle)

        # 2. Get left and right angles
        corner_angles.sort()


        # 3. to find smallest angle interval, get largest gap between neighbor angles.
        # then the object angle interval is everything except that largest empty gap
        gaps = []

        for i in range(len(corner_angles) - 1):
            # find angle gap between the sourced corners and next, up until last one
            gaps.append(corner_angles[i + 1] - corner_angles[i])

        # add gap between last and first, add 2pi, since crosses -pi ot pi
        gaps.append(corner_angles[0] - corner_angles[-1] + 2.0*math.pi)

        largest_gap = gaps.index(max(gaps))

        if largest_gap == len(corner_angles) - 1:
            # its between last and first
            start_angle = corner_angles[0]
            end_angle = corner_angles[-1]
        else:
            start_angle = corner_angles[largest_gap + 1]
            end_angle = corner_angles[largest_gap]
        
        return start_angle, end_angle




    def ray_segment_intersection_distance(self, ray_angle, p1, p2):
        """
        Compute where laser ray intersects line segment
        The ray starts at (0, 0) and points in direction of ray_angle
        The segment is one edge of bbox footprint (p1 -> p2)
        """

        # 1. unit vector in direction of ray
        ray_dx = math.cos(ray_angle)
        ray_dy = math.sin(ray_angle)

        # 2. line segment vector
        segment_dx = p2[0] - p1[0]
        segment_dy = p2[1] - p1[1]

        # 3. intersection point
        # If denom is near zero, they are parallel (or nearly parallel)
        denom = ray_dx * segment_dy - ray_dy * segment_dx

        if abs(denom) < 1e-9:
            return None

        # Vector from ray origin to segment start
        diff_x = p1[0]
        diff_y = p1[1]

        # ray_t = distance along the ray
        # seg_u = interpolation factor along the segment [0, 1]
        ray_t = (diff_x * segment_dy - diff_y * segment_dx) / denom
        seg_u = (diff_x * ray_dy - diff_y * ray_dx) / denom

        # Intersection must be in front of the laser origin
        if ray_t < 0.0:
            return None

        # Intersection must lie on the finite segment, not on its infinite line
        if seg_u < 0.0 or seg_u > 1.0:
            return None

        return ray_t

    def ray_box_distance(self, ray_angle, bottom_corners_laser):
        """
        Compute where laser ray would first hit objects bbox footprint
        - Bbox footprint: 2D rectangle on ground
        - Checks intersection of ray with all four angles and returns nearest valid hit distance
        - uses scan ray angle and the 4 bbox footprint corners
        """

        edges = [[bottom_corners_laser[0], bottom_corners_laser[1]],
                    [bottom_corners_laser[1], bottom_corners_laser[2]],
                    [bottom_corners_laser[2], bottom_corners_laser[3]],
                    [bottom_corners_laser[3], bottom_corners_laser[0]]]

        hit_distances = []
        for p1, p2 in edges: # check the four edges
            hit_distance = self.ray_segment_intersection_distance(ray_angle, p1, p2)
            if hit_distance is not None:
                hit_distances.append(hit_distance)
        
        if len(hit_distances) == 0:
            return None # no hits
        
        return min(hit_distances)
    
    def get_scan_distance_margin(self, track):
        # Scan margin (for hit, free, blocked) depends on track pose uncertainty
        
        base_margin = self.scan_margin
        pose_xy_var =  float(track.pose_cov[0, 0]) + float(track.pose_cov[1, 1])

        return base_margin + math.sqrt(pose_xy_var)


    def check_occlusion(self, track):
        """
        Classify scan rays within bbox angular (horizontal) interfal into three categories:
        - blocked: ray hits before (expected) object
        - hit: ray hits object (expected)
        - free: ray hits after (expected) object
        """

        if self.scan_msg is None or self.T_map_laser is None: # if no scan msg or transform, return None
            return None
        

        # 1. get  bbox corners, select bottom 4 (horizontal profile)      
        corners_map = get_bbox_corners_map(track)
        bottom_corners_map = corners_map[:4]

        # 1.2 transform points map-> laser (iteratively)
        bottom_corners_laser = []
        for corner_map in bottom_corners_map:
            corner_laser = transform_point(self.T_map_laser, corner_map)
            bottom_corners_laser.append([corner_laser[0], corner_laser[1]])

        # 1.3 get start and end angles (horizontal mask)
        start_angle, end_angle = self.get_object_bearings_from_corners(bottom_corners_laser)

        # 2. check scan rays
        blocked = 0
        hit = 0
        free = 0
        valid = 0

        ray_angle = self.scan_msg.angle_min
        for r_scan in self.scan_msg.ranges:
            if self.angle_is_in_interval(ray_angle, start_angle, end_angle):
                expected_distance = self.ray_box_distance(ray_angle, bottom_corners_laser)
                if expected_distance is not None:
                    valid += 1

                    scan_margin = self.get_scan_distance_margin(track)
                    
                    if not (math.isfinite(r_scan) and self.scan_msg.range_min <= r_scan <= self.scan_msg.range_max):
                        # no return = free space ray
                        free += 1

                    elif r_scan + scan_margin < expected_distance:
                        # hit before is blocked
                        blocked += 1

                    elif r_scan - scan_margin > expected_distance:
                        # hit after is free
                        free += 1
                    else:
                        # hit at box is hit
                        hit += 1

            ray_angle += self.scan_msg.angle_increment

        if valid >= self.min_valid_rays_count:
            result = {"occluded_fraction": blocked / valid, "confirmed_fraction": hit / valid, "free_fraction": free / valid, "valid_rays": valid}
            if self.debug_vis:
                print(f"VIS-ID {track.track_id}: valid {result['valid_rays']}, blckd {result['occluded_fraction']}, hit {result['confirmed_fraction']}, free {result['free_fraction']}")
            
            return result
        else:
            if self.debug_vis:
                print(f"VIS-not enough valid rays: {valid}")
            return None