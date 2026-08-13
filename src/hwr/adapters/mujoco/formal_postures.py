"""Named right-arm postures used by the privileged formal-scene expert."""

ARM_STOW = (0.0, -0.90, -1.00, 0.0, 1.70, 0.0)
ARM_READY_TABLE = (0.0, -1.30, 0.70, 0.0, 0.60, 0.0)
ARM_READY_DRAWER = (0.0, -1.50, 0.50, 0.0, 1.00, 0.0)
ARM_READY_KITCHEN = (0.0, -1.60, 0.0, 0.0, 1.60, 0.0)

# This continuous branch moves above the handle, advances over it with vertical
# clearance, then lowers the U-shaped fingers around it without interpenetration.
ARM_DRAWER_ABOVE = (0.97134, -1.27975, 0.57568, -1.15433, 1.12627, 0.77129)
ARM_DRAWER_PREALIGN = (0.76740, -1.33392, 1.39325, -1.63218, 0.76922, 1.65614)
ARM_DRAWER_GRASP = (0.76740, -1.14570, 1.70959, -2.07676, 0.91688, 2.30958)

DRAWER_GRIPPER_ROTATION = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


def drawer_posture(stage_kind: str) -> tuple[float, ...]:
    return {
        "arm_drawer_above": ARM_DRAWER_ABOVE,
        "arm_drawer_prealign": ARM_DRAWER_PREALIGN,
        "arm_drawer_descend": ARM_DRAWER_GRASP,
    }[stage_kind]
