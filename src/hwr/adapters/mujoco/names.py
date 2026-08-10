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

ARM_JOINTS = tuple(f"right_arm_joint_{index}" for index in range(1, 7))
ARM_ACTUATORS = tuple(f"right_arm_joint_{index}_position" for index in range(1, 7))
SECONDARY_ARM_JOINTS = tuple(f"left_arm_joint_{index}" for index in range(1, 7))
SECONDARY_ARM_ACTUATORS = tuple(
    f"left_arm_joint_{index}_position" for index in range(1, 7)
)
ALL_ARM_JOINTS = ARM_JOINTS + SECONDARY_ARM_JOINTS

FINGER_JOINTS = (
    "right_gripper_left_finger_joint",
    "right_gripper_right_finger_joint",
)
FINGER_ACTUATORS = (
    "right_gripper_left_finger_position",
    "right_gripper_right_finger_position",
)
SECONDARY_FINGER_JOINTS = (
    "left_gripper_left_finger_joint",
    "left_gripper_right_finger_joint",
)
SECONDARY_FINGER_ACTUATORS = (
    "left_gripper_left_finger_position",
    "left_gripper_right_finger_position",
)
ALL_FINGER_JOINTS = FINGER_JOINTS + SECONDARY_FINGER_JOINTS

POLICY_CAMERAS = ("head_rgb", "head_depth", "wrist_rgb")
EVIDENCE_CAMERAS = ("third_person", *POLICY_CAMERAS)

ARM_HOME = (0.0, 1.30, -2.50, 0.0, -1.50, 0.0)
SECONDARY_ARM_HOME = (0.0, 1.30, -2.50, 0.0, -1.50, 0.0)
WHEEL_RADIUS = 0.09
TRACK_WIDTH = 0.54
FINGER_TRAVEL = 0.085
