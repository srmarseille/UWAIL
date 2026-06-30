import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracked_object import TrackedObject

from .math_helpers import mahalanobis_distance_xyz

# Mehalanobis distance as match cost (for hung association)
def mahalanobis_match_cost(track, obs):
    return mahalanobis_distance_xyz(track.pose, track.pose_cov, obs.pose, obs.pose_cov)

def euclidean_distance(track, obs):
    """Returns euclidean distance between track and obs, = match cost"""
    return np.linalg.norm(obs.pose[:3] - track.pose[:3])


def match_cost(track, obs, association_method):
    if association_method == "mahalanobis":
        return mahalanobis_match_cost(track, obs)
    elif association_method == "euclidean":
        return euclidean_distance(track, obs)
    raise ValueError(f"Unknown association method: {association_method}")
    

# Function for hungarian association, room for other association methods later
def hungarian_association(observations, tracks, max_match_cost, association_method="mahalanobis"):
    # 1. handle edge cases
    if len(tracks) == 0: # no existing tracks, all obs are unmatched
        return [], [], list(range(len(observations)))

    if len(observations) == 0: # no new obs, all tracks are unmatched
        return [], list(range(len(tracks))), []

    # 2. construct cost matrix
    initialization_cost = 1e9 # big starting cost, to fill mat
    cost_matrix = np.full((len(tracks), len(observations)), initialization_cost, dtype=float)

    # 3. fill cost matrix
    for track_idx, track in enumerate(tracks):
        for obs_idx, obs in enumerate(observations):
            cost = match_cost(track, obs, association_method=association_method)
            if cost < max_match_cost:

                cost_matrix[track_idx, obs_idx] = cost # add cost to mat

    # 4. Hungarian assignment finds globally minimum cost match
    row_ids, col_ids = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    matched_track_ids = set()
    matched_obs_ids = set()

    # 5. Find matched pairs = associated tracks and observations.
    for track_idx, obs_idx in zip(row_ids, col_ids):
        if cost_matrix[track_idx, obs_idx] < initialization_cost: # keep only pairs that pass match cost
            matched_pairs.append((track_idx, obs_idx))
            matched_track_ids.add(track_idx)
            matched_obs_ids.add(obs_idx)

    # 6. Tracks not used in matches are unmatched
    unmatched_tracks = []
    for track_idx in range(len(tracks)):
        if track_idx not in matched_track_ids:
            unmatched_tracks.append(track_idx)

    # 7. Observations not used in matches are unmatched
    unmatched_observations = []
    for obs_idx in range(len(observations)):
        if obs_idx not in matched_obs_ids:
            unmatched_observations.append(obs_idx)

    return matched_pairs, unmatched_tracks, unmatched_observations

        


class ObjectMap:
    def __init__(self, max_match_cost, min_prob_existence=0.1, free_threshold=0.8, unreliable_threshold=0.4, max_unseen_time=None):
        self.tracks = []
        self.max_match_cost = max_match_cost
        self.min_prob_existence = min_prob_existence

        # thresholds for lidar scan occlusion checking
        self.free_threshold = free_threshold
        self.unreliable_threshold =  unreliable_threshold
        self.extra_free_required = 0.2

        # Zero process noise for both pose and extent
        self.Q_pose = np.zeros((4, 4))
        self.Q_extent = np.zeros((3, 3))

        # association method: euclidean or mahalanobis (with cov), for abblation study
        #self.association_method = "euclidean"
        self.association_method = "mahalanobis"


        self.next_track_id = 0




    def new_track(self, obs, prior_prob_existence=0.5):
        """Make new track from observation"""
        new_track_id = self.next_track_id

        track = TrackedObject(new_track_id, obs, prior_prob_existence=prior_prob_existence)
        self.next_track_id += 1 # increase counter

        self.tracks.append(track) # add to mapp

    def associate(self, observations, method="hungarian"):
        """
        Method to associate incoming observations with existing tracks,
        returns matched pairs, unmatched tracks and unmatched observations
        uses current map state (self.tracks), and requires observations as input 
        """
        if method == "hungarian":
            matched_pairs, unmatched_tracks, unmatched_observations = hungarian_association(observations, self.tracks, self.max_match_cost, association_method=self.association_method)
            return matched_pairs, unmatched_tracks, unmatched_observations
        
        raise ValueError(f"Unknown association method: {method}")

    def update_map(self, observations, visibility_evaluator, time_observed, localization_cov=None, prior_prob_existence=0.5):
        """
        Method to update map state based on incoming observations
        """

        # 0. predict step: add process noise (bounds covariance)
        for track in self.tracks:
            track.predict(self.Q_pose, self.Q_extent)
        
        # 1. Associate incoming observations with existing tracks
        matched_pairs, unmatched_tracks, unmatched_observations = self.associate(observations)

        

        # 2. Update matched tracks with positive evidence
        for track_idx, obs_idx in matched_pairs:
            track = self.tracks[track_idx]
            obs = observations[obs_idx]
            track.update_detected(obs)

        # 3. Create new tracks for unmatched observations
        for obs_idx in unmatched_observations:
            obs = observations[obs_idx]
            self.new_track(obs, prior_prob_existence=prior_prob_existence)

        # 4. Update unmatched tracks with negative evidence
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            
            # check 1: skip if no camera transform
            if visibility_evaluator.T_map_camera is None:
                continue 

            # check 2:  skip if track outside frustum (fov and range)
            if not visibility_evaluator.is_within_fov(track):
                continue
            
            # check predicted object space with lidar rays 
            occlusion = visibility_evaluator.check_occlusion(track) # returns dict: free_fraction, confirmed_fraction, occluded_fraction

            # check 3: skip if no fractions returned
            if occlusion is None:
                continue
            
            

            free_frac = occlusion['free_fraction']
            non_free_frac = occlusion['confirmed_fraction'] + occlusion['occluded_fraction']

            # apply negative evidence only when the footprint is free 
            if free_frac > self.free_threshold and free_frac > non_free_frac + self.extra_free_required:
                # classify as free and apply LLR (then likel object not there)
                track.update_lidar_freespace(observation_time=time_observed, localization_cov=localization_cov)
                
                continue



        # 5. Remove tracks with low existence probability
        self.prune_map()

    def prune_map(self):
        alive_tracks = []

        for track in self.tracks: # if below prob threshold, remove
            if track.prob_existence > self.min_prob_existence:
                alive_tracks.append(track)

        self.tracks = alive_tracks