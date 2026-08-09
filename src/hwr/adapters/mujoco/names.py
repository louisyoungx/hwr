"""Frozen entity names shared by the model compiler and runtime adapter."""

WHEEL_JOINTS = (
    "wheel_front_left_joint",
    "wheel_rear_left_joint",
    "wheel_front_right_joint",
    "wheel_rear_right_joint",
)

WHEEL_ACTUATORS = (
    "wheel_front_left_velocity",
    "wheel_rear_left_velocity",
    "wheel_front_right_velocity",
    "wheel_rear_right_velocity",
)

ARM_JOINTS = tuple(f"arm_joint_{index}" for index in range(1, 7))
ARM_ACTUATORS = tuple(f"arm_joint_{index}_position" for index in range(1, 7))

FINGER_JOINTS = ("left_finger_joint", "right_finger_joint")
FINGER_ACTUATORS = ("left_finger_position", "right_finger_position")

POLICY_CAMERAS = ("head_rgb", "head_depth", "wrist_rgb")
EVIDENCE_CAMERAS = ("third_person", *POLICY_CAMERAS)

ARM_HOME = (0.0, -0.65, 1.20, 0.0, -0.55, 0.0)
WHEEL_RADIUS = 0.09
TRACK_WIDTH = 0.51
FINGER_TRAVEL = 0.075
