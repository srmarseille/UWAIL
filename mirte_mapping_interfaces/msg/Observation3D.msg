std_msgs/Header header
string class_id
float64 confidence
bool asymmetric

float32 range_to_object
float32 relative_view_angle
int32 n_pixels_raw
int32 n_points_final
float32 surviving_fraction
float32 depth_mad_m
float32 depth_median_m
float32 u_mean
float32 v_mean
float32 u_offset_norm
float32 v_offset_norm
float32 object_y_cam

float32 sigma_r
float32 sigma_t
float32 sigma_z
float32 sigma_yaw

float32 sigma_size_x
float32 sigma_size_y
float32 sigma_size_z

# Bottom center of the filtered visible geometry estimate.
# The pose orientation is the estimated upright local object frame from visible geometry.
# local +x is the estimated front/depth axis in the ground plane and is flipped
# to point away from the sensor when possible.
# local +y is the estimated side axis in the ground plane.
# local +z points up.
geometry_msgs/PoseWithCovariance object_pose

# Depth camera position in header.frame_id
geometry_msgs/Point sensor_position
float64 sensor_yaw

# Useful metadata for mapping and later measurement-model fitting
float64[2] pca_eig 

# Extent in the local estimated object frame
float64 size_x
float64 size_y
float64 size_z

# Row-major 3x3 covariance for [size_x, size_y, size_z]
float64[9] extent_covariance
