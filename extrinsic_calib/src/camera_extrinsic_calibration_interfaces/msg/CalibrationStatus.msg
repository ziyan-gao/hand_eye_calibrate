uint8 UNCONFIGURED=0
uint8 WAITING_FOR_DATA=1
uint8 READY=2
uint8 CAPTURING=3
uint8 SOLVING=4
uint8 ERROR=5
uint8 RESULT_NONE=0
uint8 RESULT_VALID=1
uint8 RESULT_STALE=2
uint8 RESULT_FAILED=3

builtin_interfaces/Time stamp
uint8 state
uint8 result_state
bool busy
string message

string session_id
string session_dir
string mode
int32 columns
int32 rows
float64 square_size_mm
int64 config_revision
int64 dataset_revision
int32 num_valid_samples
int32 num_total_samples

# Live robot pose: T_base_from_eef (m). Valid only when eef_valid.
bool eef_valid
string eef_reason
builtin_interfaces/Time eef_stamp
geometry_msgs/Transform eef
string base_frame
# xArm RobotMsg mode (0 position, 2 manual/teaching) and state; -1 if unknown
int16 robot_mode
int16 robot_state
string eef_frame

# Live board pose: T_camera_from_board (m). Valid only when board_valid.
bool board_valid
string board_reason
builtin_interfaces/Time board_stamp
geometry_msgs/Transform board
string camera_frame
bool intrinsics_valid

bool can_capture
bool can_solve
bool can_save
CalibrationResult result
