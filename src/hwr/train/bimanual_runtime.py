"""Runtime framing helpers shared by bimanual training components."""

from hwr.core.embodied import DualArmAction, DualArmActionFrame


def dual_arm_action_frame(
    timestamp_ns: int, action: DualArmAction, *, source: str
) -> DualArmActionFrame:
    period_ns = round(1_000_000_000 / 20.0)
    return DualArmActionFrame(
        timestamp_ns,
        timestamp_ns,
        timestamp_ns + 2 * period_ns,
        source,
        action,
    )
