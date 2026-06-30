import numpy as np
from scipy.spatial.transform import Rotation

from mirte_mapping_interfaces.msg import Observation3D, Observation3DArray
from .measurement_model2 import MeasurementModel


class PoseEstimationCore:
    def __init__(self, max_detection_depth_m, max_object_z, debug_perception, min_points_per_object=10, mask_edge_margin_px=1, asymmetry_by_class=None, sensor_bias_correction=True):
        self.max_detection_depth_m = max_detection_depth_m
        self.max_object_z = max_object_z
        self.min_points_per_object = min_points_per_object
        self.mask_edge_margin_px = mask_edge_margin_px
        self.asymmetry_by_class = asymmetry_by_class or {}


        self.measurement_model = MeasurementModel()
        self.sensor_bias_correction = True # corrects estimates using estimated sensor bias (from measurement model)
        self.debug_perception = debug_perception


    def is_asymmetric(self, class_id):
        """returns if class is asymetric"""
        return self.asymmetry_by_class.get(str(class_id), True)
    
    def normalize_axis_yaw(self, yaw):
        """ 
        Normalizes yaw to [0, pi]
        - asymmetric objects are not distinguished between front and rear
        - so yaw is axis, not direction
        """
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw)) # [-pi, pi]

        if yaw < 0.0:
            yaw += np.pi # [0, pi)

        return yaw
    

    def get_R_rtz_world(self, object_pos_world, camera_pos):
        """
        Finds rotation matrix from rtz (cylinder camera frame) to world.
        """
        # 1. Radial: cector from camera to object, flattened to ground
        radial = object_pos_world - camera_pos
        radial[2] = 0.0
        radial_norm = np.linalg.norm(radial)

        radial = radial / radial_norm # unit vector

        # 2. z points up
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        
        # 3. y perpendicular to z and radial
        tangential = np.cross(z_axis, radial)
        tangential_norm = np.linalg.norm(tangential)


        tangential = tangential / tangential_norm
        
        # create 3x3 rot mat 
        R_rtz_to_world = np.column_stack((radial, tangential, z_axis))
        return R_rtz_to_world
    

    def get_relative_view_angle(self, object_pos_world, camera_pos, object_rotation_world):
        # 1. vector object -> camera
        view_dir = camera_pos - object_pos_world
        view_dir[2] = 0.0
        view_norm = np.linalg.norm(view_dir)

        if view_norm < 1e-9:
            return 0.0

        view_dir = view_dir / view_norm #normalize vector

        # 2. object heading 
        object_x = object_rotation_world[:, 0].copy()
        object_x[2] = 0.0
        object_x_norm = np.linalg.norm(object_x)

        if object_x_norm < 1e-9:
            return 0.0

        object_x = object_x / object_x_norm

        # 3. Angle Calculation: absolute value restricts output to 0 -> 90 degrees
        dot_value = np.clip(np.abs(np.dot(view_dir, object_x)), 0.0, 1.0)
        
        return float(np.arccos(dot_value))
            
    def get_range_to_object(self, object_pos_world, camera_pos):
        # distance from camera to object in xy (planar)
        return np.linalg.norm(object_pos_world[:2] - camera_pos[:2])


    def filter_instance_points(self, depth_img, mask_img, instance_id, intrinsics):
        """"
        Filters depth points in three step process
        1. reject if < 20 pixels
        2. Trim mask edge by 1 pixel
        3. MAD filtering
        """

        # 0. extract depth points whose value == instance_id
        v_raw, u_raw = np.where(mask_img == instance_id)
        n_pixels_raw = len(v_raw) # number of depth points in instance

        # 1. reject if < 20
        if n_pixels_raw < 20:
            return None

        u = u_raw.copy()
        v = v_raw.copy()

        # 2. Trim mask edge by 1 pixel
        if self.mask_edge_margin_px > 0:
            h, w = mask_img.shape
            keep = np.ones(len(v), dtype=bool)

            for step in range(1, self.mask_edge_margin_px + 1):
                # Keep only pixels for which left righ up and down neighbors are also in the mask
                # also prevents indexing out of image
                inside = (u - step >= 0) & (u + step < w) & (v - step >= 0) & (v + step < h)
                keep &= inside

                if not np.any(keep):
                    return None

                # for each remaining pixel, check if neighbors are also in the mask
                # if not, remove
                valid_idx = np.where(keep)[0]
                same_left = np.zeros(len(v), dtype=bool)
                same_right = np.zeros(len(v), dtype=bool)
                same_up = np.zeros(len(v), dtype=bool)
                same_down = np.zeros(len(v), dtype=bool)

                same_left[valid_idx] = mask_img[v[valid_idx], u[valid_idx] - step] == instance_id
                same_right[valid_idx] = mask_img[v[valid_idx], u[valid_idx] + step] == instance_id
                same_up[valid_idx] = mask_img[v[valid_idx] - step, u[valid_idx]] == instance_id
                same_down[valid_idx] = mask_img[v[valid_idx] + step, u[valid_idx]] == instance_id

                keep &= same_left & same_right & same_up & same_down

            if not np.any(keep):
                return None

            u = u[keep]
            v = v[keep]

        # check if still enough points
        if len(v) < self.min_points_per_object:
            return None

        # convert from mm, to meters
        depth_values = depth_img[v, u].astype(np.float32) / 1000.0

        # remove outside of bounds (default 0.3 - 2.0m)
        valid_depth = np.isfinite(depth_values) & (depth_values > 0.0)
        valid_depth &= depth_values <= self.max_detection_depth_m

        if not np.any(valid_depth):
            return None

        # 3. MAD filtering
        depth_valid = depth_values[valid_depth]
        depth_median = np.median(depth_valid)
        depth_mad = np.median(np.abs(depth_valid - depth_median))

        keep_depth = np.zeros_like(depth_values, dtype=bool)
        valid_indices = np.where(valid_depth)[0]

        # If the MAD is almost zero, use a small absolute threshold instead.
        # Otherwise use 1.4826*MAD   
        if depth_mad < 1e-6:
            keep_valid = np.abs(depth_valid - depth_median) < 0.03
        else:
            depth_sigma = 1.4826 * depth_mad
            keep_valid = np.abs(depth_valid - depth_median) < 1.5 * depth_sigma

        keep_depth[valid_indices] = keep_valid

        if not np.any(keep_depth):
            return None

        u = u[keep_depth]
        v = v[keep_depth]
        depth_values = depth_values[keep_depth]

        if len(depth_values) < self.min_points_per_object:
            return None
        
        # Back-project image pixels to 3D camera-frame points using the pinhole camera model:
        # x = (u - cx) * z / fx, y = (v - cy) * z / fy, z = depth.
        x = (u - intrinsics["cx"]) * depth_values / intrinsics["fx"]
        y = (v - intrinsics["cy"]) * depth_values / intrinsics["fy"]

        points_cam = np.column_stack((x, y, depth_values))
        valid_points = np.all(np.isfinite(points_cam), axis=1)
        points_cam = points_cam[valid_points]

        if len(points_cam) < self.min_points_per_object:
            return None

        n_points_final = len(points_cam)

        # extra information, used for measurement model fitting
        extra_information = {"n_pixels_raw": int(n_pixels_raw),
                            "n_points_final": int(n_points_final),
                            "surviving_fraction": float(n_points_final / max(n_pixels_raw, 1)),
                            "depth_mad_m": float(depth_mad),
                            "depth_median_m": float(depth_median),
                            "u_mean": float(np.mean(u)),
                            "v_mean": float(np.mean(v)),
                            "u_offset_norm": float((np.mean(u) - intrinsics["cx"]) / intrinsics["fx"]),
                            "v_offset_norm": float((np.mean(v) - intrinsics["cy"]) / intrinsics["fy"]),
                            "object_y_cam": float(np.median(points_cam[:, 1]))}

        return points_cam, extra_information

    def estimate_yaw_rotation(self, points_world, camera_pos):
        """
        Estimates yaw rotation using PCA
        """
        xy = points_world[:, :2] #project to xy
        xy_center = np.mean(xy, axis=0) # find mean of points
        xy_zero = xy - xy_center # center points around mean

        if len(xy_zero) < 2:
            return None, None
        
        # PCA on 2d (xy projected pointcloud)
        cov = np.cov(xy_zero.T) # first variance around mean of points
        eig_vals, eig_vecs = np.linalg.eigh(cov) # get eigenvalues and eigenvectors

        # principal (largest-eig) axis in xy
        principal_xy = eig_vecs[:, np.argmax(eig_vals)]
        principal_xy = principal_xy / np.linalg.norm(principal_xy)

        # disambiguate sign: principal axis should have non-negative
        # component along the "object -> away-from-camera" direction
        away = xy_center - camera_pos[:2]
        away_norm = np.linalg.norm(away)
        if away_norm > 1e-9:
            away = away / away_norm
            if np.dot(principal_xy, away) < 0.0:
                principal_xy = -principal_xy

        # x_axis = perpendicular to principal in xy (object's "front" direction
        # if you treat the long side as the side of the object)
        x_axis = np.array([-principal_xy[1], principal_xy[0], 0.0], dtype=float)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # also disambiguate x_axis sign the same way
        if np.dot(x_axis[:2], away) < 0.0:
            x_axis = -x_axis
        
        # z is fixed upward, so the remaining horizontal axis follows from the cross product.
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Rotation matrix whose columns are the object-frame axes expressed in the world frame.
        rotation_world = np.column_stack((x_axis, y_axis, z_axis))
        # Sort eigenvalues from large to small so eig_vals[0] axis corresponding to direction with largest spread
        eig_vals = np.sort(np.asarray(eig_vals, dtype=float))[::-1]

        return rotation_world, eig_vals

    def estimate_view_rotation(self, points_world, camera_pos):
        """
        For symmetric objects, yaw is not estimated from PCA.
        Instead, use a view-aligned frame so the extent is measured consistently from the camera view.
        - frame is defined as [radial, tangential, z] (normalized)
        - radial axis is direction from camera to object
        - z axis is up
        - tangential axis is the cross product of radial and z (horizontal along image plane)

        """
        visible_center = np.mean(points_world, axis=0)

        radial_axis = visible_center - camera_pos # direction from camera to object
        radial_axis[2] = 0.0
        radial_norm = np.linalg.norm(radial_axis)

        if radial_norm < 1e-9:
            return None

        radial_axis = radial_axis / radial_norm # norm vector camera -> object

        # z up,
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        tangential_axis = np.cross(z_axis, radial_axis)
        tangential_norm = np.linalg.norm(tangential_axis)

        if tangential_norm < 1e-9:
            return None

        tangential_axis = tangential_axis / tangential_norm

        # Return rotation matrix [r, t, z] (normalized)
        return np.column_stack((radial_axis, tangential_axis, z_axis))

    def get_local_box(self, points_world, center_world, rotation_world):
        """
        Returns local bounding box, with center bottom convention
        - find min max of point in local frame
    
        """

        local_points = (points_world - center_world) @ rotation_world

        x_min = float(np.min(local_points[:, 0]))
        x_max = float(np.max(local_points[:, 0]))
        y_min = float(np.min(local_points[:, 1]))
        y_max = float(np.max(local_points[:, 1]))
        z_min = float(np.min(local_points[:, 2]))
        z_max = float(np.max(local_points[:, 2]))

        size_x = x_max - x_min
        size_y = y_max - y_min
        size_z = z_max - z_min

        # The observation position is not the visible point-cloud center.
         # It is the bottom-center of the local bounding box, matching the object-map convention.
        object_pos_local = np.array([0.5 * (x_min + x_max), 0.5 * (y_min + y_max), z_min], dtype=float)
        object_pos_world = center_world + rotation_world @ object_pos_local

        return size_x, size_y, size_z, object_pos_world

    def build_observation(self, points_world, camera_pos, sensor_yaw, det_2d, header, diag):
        
        # 1. first some checks
        # skip if too few points survived filter (redundant for when filtering is disabled, already done after filtering)
        if len(points_world) < self.min_points_per_object:
            return None

        # get first detection hypothesis as class label and confidence
        if det_2d.results:
            class_id = str(det_2d.results[0].hypothesis.class_id)
            class_id_int = int(class_id)
            confidence = float(det_2d.results[0].hypothesis.score)
        else: 
            # if there is none, keep object but mark class as unknown
            class_id = "unknown"
            class_id_int = -1
            confidence = 0.0

        # 2. get asymmetry flag
        asymmetric = self.is_asymmetric(class_id)
        visible_center = np.mean(points_world, axis=0) # visible center, used for box fitting


        # 3.1 for asymmetric objects, estimate yaw and oriented extent
        if asymmetric:
            #1. call PCA method to estimate yaw
            pca_rotation, eig_vals = self.estimate_yaw_rotation(points_world, camera_pos) # estimate
            if pca_rotation is None: #gaurd
                return None
            
            yaw_world = Rotation.from_matrix(pca_rotation).as_euler("xyz")[2] # matrix to angle around z
            yaw_axis_world = self.normalize_axis_yaw(yaw_world) # direction=meaninless; normalize 0-180
            
            # 2. get local box
            size_x, size_y, size_z, object_pos_world = self.get_local_box(points_world, visible_center, pca_rotation)



            range_to_object = self.get_range_to_object(object_pos_world, camera_pos)
            relative_view_angle = self.get_relative_view_angle(object_pos_world, camera_pos, pca_rotation)
            

            covariances = self.measurement_model.get_bias_and_covariance_v2(class_id_int, 
                                                                             object_z_elevation=object_pos_world[2])


                                                                            
            pos_bias_rtz, pos_cov_rtz, extent_bias, extent_cov, _, yaw_var = covariances
            R_rtz_world = self.get_R_rtz_world(object_pos_world, camera_pos)
            if R_rtz_world is None:
                return None
            
            # convert position bias and cov from RTZ to world frame
            pos_bias_world = R_rtz_world @ pos_bias_rtz
            pos_cov_world = R_rtz_world @ pos_cov_rtz @ R_rtz_world.T
            
            # if to correct for bias, apply... (not for yaw)
            if self.sensor_bias_correction:
                object_pos_world = object_pos_world - pos_bias_world

                size_x = max(size_x - extent_bias[0], 1e-4)
                size_y = max(size_y - extent_bias[1], 1e-4)
                size_z = max(size_z - extent_bias[2], 1e-4)
            
            quat = Rotation.from_euler("z", yaw_axis_world).as_quat()

        # 3.2 for symmetric objects
        else: # only do pose
            view_rotation = self.estimate_view_rotation(points_world, camera_pos)
            if view_rotation is None:
                return None
            
            # get extent: width = depth -> size t = both x and y
            _, size_t, size_z, object_pos_world = self.get_local_box(points_world, visible_center, view_rotation)

            
            size_x = size_t
            size_y = size_t
            
            # for unnormalizing cov and bias, make arrays for measurement model 
            range_to_object = self.get_range_to_object(object_pos_world, camera_pos)

            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float) # identity quaternion, yaw set to 0 deg in world frame
            eig_vals = np.zeros(2, dtype=float)
            yaw_var = 1e6 # set yaw variance high, cause no reliable yaw

            relative_view_angle = float('nan')

            covariances = self.measurement_model.get_bias_and_covariance_v2(class_id_int, 
                                                                             object_z_elevation=object_pos_world[2])
            

            pos_bias_rtz, pos_cov_rtz, extent_bias, extent_cov, _, _ = covariances

            # convert position bias and cov from RTZ to world frame
            R_rtz_world = self.get_R_rtz_world(object_pos_world, camera_pos)
            pos_bias_world = R_rtz_world @ pos_bias_rtz
            pos_cov_world = R_rtz_world @ pos_cov_rtz @ R_rtz_world.T

            # if to correct for bias, apply...
            if self.sensor_bias_correction:
                object_pos_world = object_pos_world - pos_bias_world

                size_x = max(size_x - extent_bias[0], 1e-4)
                size_y = max(size_y - extent_bias[1], 1e-4)
                size_z = max(size_z - extent_bias[2], 1e-4)


        if size_x <= 0.0 or size_y <= 0.0 or size_z <= 0.0:
            return None
        
        # prepare covariance
        pose_cov = np.zeros((6, 6), dtype=float)
        pose_cov[0:3, 0:3] = pos_cov_world
        pose_cov[5, 5] = yaw_var

        # sigmas for debugging
        sigma_r = float(np.sqrt(pos_cov_rtz[0, 0]))
        sigma_t = float(np.sqrt(pos_cov_rtz[1, 1]))
        sigma_z = float(np.sqrt(pos_cov_rtz[2, 2]))
        sigma_yaw = float(np.sqrt(yaw_var))

        sigma_size_x = float(np.sqrt(extent_cov[0, 0]))
        sigma_size_y = float(np.sqrt(extent_cov[1, 1]))
        sigma_size_z = float(np.sqrt(extent_cov[2, 2]))
                                                                                                              

        obs = Observation3D()
        obs.header = header
        obs.class_id = class_id
        obs.confidence = confidence
        obs.asymmetric = asymmetric

        obs.object_pose.pose.position.x = float(object_pos_world[0])
        obs.object_pose.pose.position.y = float(object_pos_world[1])
        obs.object_pose.pose.position.z = float(object_pos_world[2])
        
        
        obs.object_pose.pose.orientation.x = float(quat[0])
        obs.object_pose.pose.orientation.y = float(quat[1])
        obs.object_pose.pose.orientation.z = float(quat[2])
        obs.object_pose.pose.orientation.w = float(quat[3])
        obs.object_pose.covariance = pose_cov.flatten().tolist()


        obs.sensor_position.x = float(camera_pos[0])
        obs.sensor_position.y = float(camera_pos[1])
        obs.sensor_position.z = float(camera_pos[2])
        obs.sensor_yaw = float(sensor_yaw)

        obs.pca_eig[0] = float(eig_vals[0])
        obs.pca_eig[1] = float(eig_vals[1])

        obs.size_x = float(size_x)
        obs.size_y = float(size_y)
        obs.size_z = float(size_z)
        obs.extent_covariance = extent_cov.flatten().tolist()

        obs.range_to_object = float(range_to_object)
        obs.relative_view_angle = float(relative_view_angle)

        obs.sigma_r = sigma_r
        obs.sigma_t = sigma_t
        obs.sigma_z = sigma_z
        obs.sigma_yaw = sigma_yaw

        obs.sigma_size_x = sigma_size_x
        obs.sigma_size_y = sigma_size_y
        obs.sigma_size_z = sigma_size_z

        # diagnosis data
        obs.n_pixels_raw = int(diag["n_pixels_raw"])
        obs.n_points_final = int(diag["n_points_final"])
        obs.surviving_fraction = float(diag["surviving_fraction"])
        obs.depth_mad_m = float(diag["depth_mad_m"])
        obs.depth_median_m = float(diag["depth_median_m"])
        obs.u_mean = float(diag["u_mean"])
        obs.v_mean = float(diag["v_mean"])
        obs.u_offset_norm = float(diag["u_offset_norm"])
        obs.v_offset_norm = float(diag["v_offset_norm"])
        obs.object_y_cam = float(diag["object_y_cam"])

        return obs

    def process_frame(self, depth_img, mask_img, detections, intrinsics, transform, camera_pos, sensor_yaw, header):
        out_msg_array = Observation3DArray()
        out_msg_array.header = header
        all_object_points = []

        for i, det_2d in enumerate(detections):
            instance_id = i + 1

            result = self.filter_instance_points(depth_img, mask_img, instance_id, intrinsics)
            if result is None:
                continue

            points_cam, diag = result

            # transform object points from depth camera frame to world frame
            points_cam_h = np.column_stack((points_cam, np.ones(len(points_cam))))
            points_world = (transform @ points_cam_h.T).T[:, :3]

            if len(points_world) < self.min_points_per_object:
                continue

            all_object_points.extend(points_world.tolist())

            obs = self.build_observation(points_world, camera_pos, sensor_yaw, det_2d, header, diag)
            if obs is not None:
                out_msg_array.observations.append(obs)

        return out_msg_array, all_object_points

