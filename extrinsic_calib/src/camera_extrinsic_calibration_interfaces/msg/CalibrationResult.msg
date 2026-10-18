# Hand-eye result. transform = T_parent_from_child, SI units (m), quaternion xyzw.
# eye_in_hand: parent=eef, child=camera optical frame (T_eef_from_camera)
# eye_to_hand: parent=base, child=camera optical frame (T_base_from_camera)
string mode
string parent_frame
string child_frame
geometry_msgs/Transform transform
string method
int64 config_revision
int64 dataset_revision
string[] sample_ids
float64 reproj_rms_px
float64 consistency_trans_rms_mm
float64 consistency_trans_max_mm
float64 consistency_rot_rms_deg
float64 consistency_rot_max_deg
string report
