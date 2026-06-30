from .math_helpers import angle_diff,kalman_update_pose, clamp_probability, kalman_update_extent, kalman_predict, kalman_predict_extent
from .detector_model import DetectorModel
import numpy as np
import math
from collections import deque


class Observation:
    def __init__(self, header, pose, pose_cov, extent, extent_cov, class_id, confidence, asymmetric, sensor_xyz):
        self.time = header.stamp

        self.pose = pose  # pose = [x, y, z, yaw]; x,y,z in backproj, frame (currently world); yaw around z
        self.pose_cov = pose_cov

        self.extent = extent # extent = [size_x, size_y, size_z]
        self.extent_cov = extent_cov

        self.class_id = class_id
        self.confidence = confidence

        self.asymmetric = asymmetric
        self.sensor_xyz = sensor_xyz




class TrackedObject:
    def __init__(self, track_id, obs, prior_prob_existence=0.5):
        self.detector_model = DetectorModel()
        self.track_id = track_id

        # The first (intializing) observation provides pose, extent etc directly
        self.pose = obs.pose.copy()
        self.pose_cov = obs.pose_cov.copy()
        self.extent = obs.extent.copy()
        self.extent_cov = obs.extent_cov.copy()

        # Dirichlet categorical class belief
        self.class_alpha = np.full(self.detector_model.number_of_classes, self.detector_model.dirichlet_prior, dtype=float)
        c0 = int(obs.class_id)
        if 0 <= c0 < self.detector_model.number_of_classes:
            self.class_alpha[c0] += float(obs.confidence)
    
        self.prob_existence = prior_prob_existence

        self.first_seen_time = obs.time
        self.last_seen_time = obs.time
        self.last_update_time = obs.time


        ## Keyframe logic - (for pose/extent(geometry) updates only)
        # logic for keyframes queu, and storing initial keyframe
        self.max_stored_keyframes = 1
        self.recent_keyframes = deque(maxlen=self.max_stored_keyframes) # make a queu of lenght of max stored keyframes
        
        # append first keyframe
        self.recent_keyframes.append((self.compute_viewpoint(obs.sensor_xyz))) # use self.pose obs.pose, current estimate

        # keyframe novelty thresholds
        self.min_view_angle_change = math.radians(5) # 5 deg in radians
        self.min_range_change = 0.10 # meters


    @property
    def class_id(self):
        return int(np.argmax(self.class_alpha))
    
    @property
    def class_conf(self):
        return self.class_alpha.max() / self.class_alpha.sum()

    @property
    def asymmetric(self):
        return self.detector_model.get_asymmetry_by_class_id(self.class_id)
    
    @property
    def class_name(self):
        return self.detector_model.get_class_name_by_class_id(self.class_id)


    def compute_viewpoint(self, sensor_xyz):
        """
        Viewpoint is defined as the angle and range around a track center
        """

        dx = sensor_xyz[0] - self.pose[0] # distance in x
        dy = sensor_xyz[1] - self.pose[1] # distance in y


        view_angle_object_to_sensor = math.atan2(dy, dx)
        range_object_sensor = math.sqrt(dx**2 + dy**2)

        return view_angle_object_to_sensor, range_object_sensor
    
    def is_viewpoint_novel(self, sensor_xyz):

        # if no keyframes, novel
        if len(self.recent_keyframes) == 0:
            return True

        new_angle, new_range = self.compute_viewpoint(sensor_xyz)

        # compare against recent keyframes (one is stored in this case)
        for keyframe_angle, keyframe_range in self.recent_keyframes:
            angle_change = abs(angle_diff(new_angle, keyframe_angle))
            range_change = abs(new_range - keyframe_range)

            close_in_angle = angle_change < self.min_view_angle_change
            close_in_range = range_change < self.min_range_change

            # if close in angle and range, not novel
            if close_in_angle and close_in_range:
                return False
        return True
    
    def localization_confidence(self, localization_cov=None, sigma_ref=0.1):
        """
        Confidence that localization is correct enough so that the visbility check can be trusted
        - alpha = sigma_ref^2 / (sigma_ref^2 + sigma_pose^2)
        - a larger covarinace reduces strenght of negative free-space evidence
        - reference sigma (in meters) provides baseline to compare too
        """
        if localization_cov is None:
            return 1.0
        
        # only use x and y covariance 
        var_pose = localization_cov[0, 0] + localization_cov[1, 1]
        var_ref = sigma_ref ** 2
        
        return var_ref / (var_ref + var_pose)


    
    def update_existence_from_llr(self, llr, stamp):
        prior = self.prob_existence

        # additive update in log odds form
        log_odds = math.log(prior / (1.0 - prior)) + llr

        # map to probability
        self.prob_existence = clamp_probability(1.0 / (1.0 + math.exp(-log_odds)))

        # update last update time
        self.last_update_time = stamp


    def update_classes_from_detection(self, obs):
        # prevent invalid num of class ids from updating the wrong class index
        if int(obs.class_id) < 0 or int(obs.class_id) >= self.detector_model.number_of_classes:
            return
        
        # add confidence to correspoding class as pseudocount
        self.class_alpha[int(obs.class_id)] += obs.confidence

    def update_existence_from_detection(self, obs):
        # gaurd for invalid class id
        if int(obs.class_id) < 0 or int(obs.class_id) >= self.detector_model.number_of_classes:
            return

        # get recall and fp rate as P(observed | existes) and P(observed | not existes)
        recall = self.detector_model.get_recall(int(obs.class_id))
        fp_rate = self.detector_model.get_fp_rate(int(obs.class_id))

        # 1. compute log-likelihood ratio as log(P(observed | existes) / P(observed | not existes))
        llr = math.log(recall / fp_rate)

        ### Existence update core ###
        self.update_existence_from_llr(llr, obs.time)

    
    def predict(self, Q_pose, Q_extent):
        """
        This is the prediction step of the kalman filter
        - Currently, a placeholder, no process noise and transition model defined, as objects are assumed fully static
        - This method is added for completness of kalman implementation 
        """
        self.pose, self.pose_cov = kalman_predict(self.pose, self.pose_cov, Q_pose)
        self.extent, self.extent_cov = kalman_predict_extent(self.extent, self.extent_cov, Q_extent)





    def update_detected(self, obs):
        """
        Positive evidence: observation was matched to this track:
        - updates pose, extent, class belief (dirichlet categorical), and existence (bayesian binary hypothesis)
        """
        # update last track seen time 
        self.last_seen_time = obs.time

        self.update_classes_from_detection(obs)
        self.update_existence_from_detection(obs)

        ### Pose/extent
        # 0. check if viewpoint novel, if not return 
        if not self.is_viewpoint_novel(obs.sensor_xyz): # check if VP is novel, else return
            return

        # 1. kalman pose 
        self.pose, self.pose_cov = kalman_update_pose(self.pose, self.pose_cov, obs.pose, obs.pose_cov)

        # # 2. kalman extent     
        self.extent, self.extent_cov = kalman_update_extent(self.extent, self.extent_cov, obs.extent, obs.extent_cov)

        # 3. add to recent keyframes
        self.recent_keyframes.append((self.compute_viewpoint(obs.sensor_xyz)))

    def update_lidar_freespace(self, observation_time, localization_cov=None):
        """
        Update existence given lidar rays pass through track
        """
        # compute alpha, the localization weight 
        alpha = self.localization_confidence(localization_cov)

        # get LiDAR recall and fpr (currenlty just user defined/non-calibrated)
        p_free_if_exists = self.detector_model.lidar_p_free_if_exists
        p_free_if_not_exists = self.detector_model.lidar_p_free_if_not_exists

        # llr_free = alpha log(P(free | exists) / P(free | not exists))
        llr = alpha * math.log(p_free_if_exists / p_free_if_not_exists)

        self.update_existence_from_llr(llr, observation_time)

