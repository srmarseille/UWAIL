import json
import numpy as np
from .math_helpers import yaw_from_quat


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def pose_msg_to_dict(pose_msg):
    return {"x": float(pose_msg.position.x),
            "y": float(pose_msg.position.y),
            "z": float(pose_msg.position.z),
            "yaw": float(yaw_from_quat(pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w))}


def array_to_list(value):
    return np.asarray(value, dtype=float).tolist()


def detection_msg_to_log_dict(det):
    return {"stamp": stamp_to_sec(det.header.stamp),
        "frame_id": det.header.frame_id,
        "class_id": int(det.class_id),
        "confidence": float(det.confidence),
        "asymmetric": bool(det.asymmetric),
        "range_to_object": float(det.range_to_object),
        "relative_view_angle": float(det.relative_view_angle),
        "n_pixels_raw": int(det.n_pixels_raw),
        "n_points_final": int(det.n_points_final),
        "surviving_fraction": float(det.surviving_fraction),
        "depth_mad_m": float(det.depth_mad_m),
        "depth_median_m": float(det.depth_median_m),
        "u_mean": float(det.u_mean),
        "v_mean": float(det.v_mean),
        "u_offset_norm": float(det.u_offset_norm),
        "v_offset_norm": float(det.v_offset_norm),
        "object_y_cam": float(det.object_y_cam),
        "sigma_r": float(det.sigma_r),
        "sigma_t": float(det.sigma_t),
        "sigma_z": float(det.sigma_z),
        "sigma_yaw": float(det.sigma_yaw),
        "sigma_size_x": float(det.sigma_size_x),
        "sigma_size_y": float(det.sigma_size_y),
        "sigma_size_z": float(det.sigma_size_z),
        "object_pose": {"pose": pose_msg_to_dict(det.object_pose.pose),
                        "covariance": [float(x) for x in det.object_pose.covariance]},
        "sensor_position": {"x": float(det.sensor_position.x),
                            "y": float(det.sensor_position.y),
                            "z": float(det.sensor_position.z)},
        "sensor_yaw": float(det.sensor_yaw),
        "pca_eig": [float(x) for x in det.pca_eig],
        "size_x": float(det.size_x),
        "size_y": float(det.size_y),
        "size_z": float(det.size_z),
        "extent_covariance": [float(x) for x in det.extent_covariance]}


def observation_to_log_dict(obs):
    return {"stamp": stamp_to_sec(obs.time),
            "pose": {"x": obs.pose[0],
                    "y": obs.pose[1],
                    "z": obs.pose[2],
                    "yaw": obs.pose[3]},
            "pose_cov": array_to_list(obs.pose_cov),
            "extent": {"l_x": obs.extent[0],
                        "l_y": obs.extent[1],
                        "l_z": obs.extent[2]},
            "extent_cov": array_to_list(obs.extent_cov),
            "class_id": int(obs.class_id),
            "confidence": float(obs.confidence),
            "asymmetric": bool(obs.asymmetric),
            "sensor_xyz": array_to_list(obs.sensor_xyz)}



def track_to_log_dict(track):
    return {"track_id": int(track.track_id),
            "pose": {"x": track.pose[0],
                    "y": track.pose[1],
                    "z": track.pose[2],
                    "yaw": track.pose[3]},
            "pose_covariance": array_to_list(track.pose_cov),
            "extent": {"l_x": track.extent[0],
                        "l_y": track.extent[1],
                        "l_z": track.extent[2]},
            "extent_covariance": array_to_list(track.extent_cov),
            "class_id": int(track.class_id),
            "class_conf": float(track.class_conf),
            "prob_existence": float(track.prob_existence),
            "asymmetric": bool(track.asymmetric)}


def make_log_entry(detections_msg, observations, tracks, map_frame):
    stamp = detections_msg.header.stamp

    return {"stamp": stamp_to_sec(stamp),
            "map_frame": map_frame,
            "num_detections": len(detections_msg.observations),
            "num_observations_used": len(observations),
            "num_tracks": len(tracks),
            "detections": [detection_msg_to_log_dict(det) for det in detections_msg.observations],
            "observations_used_by_mapper": [observation_to_log_dict(obs) for obs in observations],
            "tracks_after_update": [track_to_log_dict(track) for track in tracks ]}


def write_map_state_jsonl(log_file, detections_msg, observations, tracks, map_frame):
    if log_file is None:
        return

    entry = make_log_entry(
        detections_msg,
        observations,
        tracks,
        map_frame)

    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()