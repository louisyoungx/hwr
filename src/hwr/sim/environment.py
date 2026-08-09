"""Deterministic continuous 2D mobile manipulation environment."""

from __future__ import annotations

import math
import random

from hwr.core.clock import DeterministicClock
from hwr.core.runtime import StepOutcome
from hwr.core.types import (
    ActionFrame,
    EpisodeEvent,
    EpisodeResult,
    ObservationFrame,
    SafetyState,
)
from hwr.safety import SafetyLimits, SafetySupervisor
from hwr.sim.geometry import (
    circle_hits_bounds,
    circle_hits_rect,
    clamp,
    range_scan,
    rotate_to_local,
    rotate_to_world,
    wrap_angle,
)
from hwr.sim.specs import HouseholdTaskSpec, RobotSpec, ZoneSpec
from hwr.sim.state import (
    ObjectSnapshot,
    RobotSnapshot,
    SimObjectState,
    SimRobotState,
    SimulationSnapshot,
)


class Household2DEnv:
    """Reference backend with a differential base and planar arm endpoint."""

    def __init__(self, robot_spec: RobotSpec, task_spec: HouseholdTaskSpec) -> None:
        self.robot_spec = robot_spec
        self.task_spec = task_spec
        self.clock = DeterministicClock()
        self.safety = SafetySupervisor(
            SafetyLimits(
                max_base_linear=robot_spec.max_linear_speed,
                max_base_angular=robot_spec.max_angular_speed,
                max_arm_command=robot_spec.arm_speed,
            ),
            arm_dof=2,
        )
        self.robot: SimRobotState | None = None
        self.objects: dict[str, SimObjectState] = {}
        self._rng = random.Random()
        self._sequence = 0
        self._steps = 0
        self._collisions = 0
        self._grasp_count = 0
        self._last_twist = (0.0, 0.0)
        self._result: EpisodeResult | None = None

    @property
    def dt(self) -> float:
        return 1.0 / self.robot_spec.control_hz

    def reset(self, *, seed: int, task_id: str) -> ObservationFrame:
        if task_id != self.task_spec.task_id:
            raise ValueError(f"backend provides {self.task_spec.task_id}, not {task_id}")
        self._rng.seed(seed)
        self.clock.current_ns = 0
        start_x, start_y, start_heading = self.task_spec.scene.robot_start
        jitter = self.task_spec.scene.start_jitter
        self.robot = SimRobotState(
            x=start_x + self._rng.uniform(-jitter, jitter),
            y=start_y + self._rng.uniform(-jitter, jitter),
            heading=wrap_angle(start_heading + self._rng.uniform(-jitter, jitter)),
        )
        self.objects = {}
        for item in self.task_spec.scene.objects:
            self.objects[item.object_id] = SimObjectState(
                object_id=item.object_id,
                label=item.label,
                x=item.center_x + self._rng.uniform(-item.spawn_jitter, item.spawn_jitter),
                y=item.center_y + self._rng.uniform(-item.spawn_jitter, item.spawn_jitter),
                radius=item.radius,
                mass=item.mass,
                target_zone_id=item.target_zone_id,
            )
        self._sequence = 0
        self._steps = 0
        self._collisions = 0
        self._grasp_count = 0
        self._last_twist = (0.0, 0.0)
        self._result = None
        return self._observation()

    def observe(self) -> ObservationFrame:
        self._require_state()
        return self._observation()

    def snapshot(self) -> SimulationSnapshot:
        """Return an immutable copy of state for diagnostics and rendering."""
        robot = self._require_state()
        end_effector_x, end_effector_y = self._world_end_effector()
        placed_count = sum(item.placed for item in self.objects.values())
        stage = (
            "complete"
            if placed_count == len(self.objects)
            else "deliver"
            if robot.carrying_object_id is not None
            else "acquire"
        )
        robot_snapshot = RobotSnapshot(
            x=robot.x,
            y=robot.y,
            heading=robot.heading,
            arm_x=robot.arm_x,
            arm_y=robot.arm_y,
            end_effector_x=end_effector_x,
            end_effector_y=end_effector_y,
            gripper=robot.gripper,
            carrying_object_id=robot.carrying_object_id,
        )
        object_snapshots = tuple(
            ObjectSnapshot(
                object_id=item.object_id,
                label=item.label,
                x=item.x,
                y=item.y,
                radius=item.radius,
                target_zone_id=item.target_zone_id,
                placed=item.placed,
            )
            for item in self.objects.values()
        )
        return SimulationSnapshot(
            sequence_id=self._sequence,
            timestamp_ns=self.clock.now_ns(),
            task_stage=stage,
            steps=self._steps,
            collisions=self._collisions,
            grasps=self._grasp_count,
            robot=robot_snapshot,
            objects=object_snapshots,
        )

    def apply(self, action: ActionFrame) -> StepOutcome:
        robot = self._require_state()
        if self._result is not None:
            raise RuntimeError("episode is complete; call reset")
        applied, safety_events = self.safety.filter(action, now_ns=self.clock.now_ns())
        events = list(safety_events)
        reward = 0.0

        previous_pose = (robot.x, robot.y, robot.heading)
        robot.heading = wrap_angle(robot.heading + applied.base_angular * self.dt)
        robot.x += applied.base_linear * math.cos(robot.heading) * self.dt
        robot.y += applied.base_linear * math.sin(robot.heading) * self.dt
        if self._base_collision():
            robot.x, robot.y, robot.heading = previous_pose
            self._collisions += 1
            reward -= self.task_spec.collision_penalty
            events.append(self._event("collision", {"kind": "base"}))

        robot.arm_x += applied.arm_command[0] * self.dt
        robot.arm_y += applied.arm_command[1] * self.dt
        arm_distance = math.hypot(robot.arm_x, robot.arm_y)
        if arm_distance > self.robot_spec.arm_reach:
            scale = self.robot_spec.arm_reach / arm_distance
            robot.arm_x *= scale
            robot.arm_y *= scale

        was_carrying = robot.carrying_object_id is not None
        robot.gripper = applied.gripper_target
        if robot.gripper >= 0.5 and not was_carrying:
            grasped = self._try_grasp()
            if grasped:
                reward += self.task_spec.grasp_reward
                events.append(self._event("object_grasped", {"object_id": grasped}))
        elif robot.gripper < 0.5 and was_carrying:
            released = self._release()
            events.append(self._event("object_released", {"object_id": released}))

        self._update_carried_object()
        self._last_twist = (applied.base_linear, applied.base_angular)
        self._steps += 1
        self.clock.advance_seconds(self.dt)
        self._sequence += 1
        terminated, truncated, terminal_reward, terminal_events = self._terminal_state()
        reward += terminal_reward
        events.extend(terminal_events)
        observation = self._observation()
        return StepOutcome(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            events=tuple(events),
            info={"applied_action": applied},
        )

    def result(self) -> EpisodeResult | None:
        return self._result

    def close(self) -> None:
        self.robot = None
        self.objects = {}

    def _observation(self) -> ObservationFrame:
        robot = self._require_state()
        target = self._current_target()
        object_relative = (0.0, 0.0)
        zone_relative = (0.0, 0.0)
        if target is not None:
            object_relative = self._relative(target.x, target.y)
            zone = self._zone(target.target_zone_id)
            zone_relative = self._relative(zone.center_x, zone.center_y)
        carrying = robot.carrying_object_id is not None
        placed_fraction = sum(item.placed for item in self.objects.values()) / len(self.objects)
        stage = "complete" if placed_fraction == 1.0 else "deliver" if carrying else "acquire"
        scan = range_scan(
            robot.x,
            robot.y,
            robot.heading,
            self.task_spec.scene.bounds,
            self.task_spec.scene.obstacles,
            self.robot_spec.lidar_rays,
            self.robot_spec.lidar_range,
        )
        return ObservationFrame(
            timestamp_ns=self.clock.now_ns(),
            sequence_id=self._sequence,
            task_id=self.task_spec.task_id,
            task_stage=stage,
            joint_position=(robot.arm_x, robot.arm_y),
            joint_velocity=(0.0, 0.0),
            gripper_position=robot.gripper,
            base_pose=(robot.x, robot.y, robot.heading),
            base_twist=self._last_twist,
            imu=(0.0, 0.0, applied_heading_rate(self._last_twist)),
            features={
                "target_object_relative": object_relative,
                "target_zone_relative": zone_relative,
                "lidar": scan,
                "carrying": (1.0 if carrying else 0.0,),
                "placed_fraction": (placed_fraction,),
            },
            safety_state=SafetyState.OK,
            quality={"simulation": 1.0},
        )

    def _base_collision(self) -> bool:
        robot = self._require_state()
        if circle_hits_bounds(
            robot.x,
            robot.y,
            self.robot_spec.base_radius,
            self.task_spec.scene.bounds,
        ):
            return True
        return any(
            circle_hits_rect(robot.x, robot.y, self.robot_spec.base_radius, obstacle)
            for obstacle in self.task_spec.scene.obstacles
        )

    def _world_end_effector(self) -> tuple[float, float]:
        robot = self._require_state()
        offset_x, offset_y = rotate_to_world(robot.arm_x, robot.arm_y, robot.heading)
        return robot.x + offset_x, robot.y + offset_y

    def _relative(self, target_x: float, target_y: float) -> tuple[float, float]:
        robot = self._require_state()
        return rotate_to_local(target_x - robot.x, target_y - robot.y, robot.heading)

    def _current_target(self) -> SimObjectState | None:
        robot = self._require_state()
        if robot.carrying_object_id:
            return self.objects[robot.carrying_object_id]
        return next((item for item in self.objects.values() if not item.placed), None)

    def _try_grasp(self) -> str | None:
        robot = self._require_state()
        ee_x, ee_y = self._world_end_effector()
        candidates = [item for item in self.objects.values() if not item.placed]
        candidates.sort(key=lambda item: math.hypot(item.x - ee_x, item.y - ee_y))
        for item in candidates:
            distance = math.hypot(item.x - ee_x, item.y - ee_y)
            if distance <= self.robot_spec.gripper_radius + item.radius and item.mass <= self.robot_spec.payload:
                robot.carrying_object_id = item.object_id
                self._grasp_count += 1
                self._update_carried_object()
                return item.object_id
        return None

    def _release(self) -> str:
        robot = self._require_state()
        object_id = robot.carrying_object_id
        if object_id is None:
            raise RuntimeError("release requested without a carried object")
        item = self.objects[object_id]
        zone = self._zone(item.target_zone_id)
        item.placed = math.hypot(item.x - zone.center_x, item.y - zone.center_y) <= zone.radius
        robot.carrying_object_id = None
        return object_id

    def _update_carried_object(self) -> None:
        robot = self._require_state()
        if robot.carrying_object_id is None:
            return
        item = self.objects[robot.carrying_object_id]
        item.x, item.y = self._world_end_effector()

    def _zone(self, zone_id: str) -> ZoneSpec:
        return next(zone for zone in self.task_spec.scene.zones if zone.zone_id == zone_id)

    def _terminal_state(self) -> tuple[bool, bool, float, tuple[EpisodeEvent, ...]]:
        all_placed = all(item.placed for item in self.objects.values())
        collision_failure = self.task_spec.collision_terminates and self._collisions > 0
        timeout = self._steps >= self.task_spec.max_steps
        if all_placed:
            self._result = self._make_result(True, "completed")
            return True, False, self.task_spec.success_reward, (self._event("task_succeeded"),)
        if collision_failure:
            self._result = self._make_result(False, "collision")
            return True, False, 0.0, (self._event("task_failed", {"reason": "collision"}),)
        if timeout:
            self._result = self._make_result(False, "timeout")
            return False, True, 0.0, (self._event("task_failed", {"reason": "timeout"}),)
        return False, False, 0.0, ()

    def _make_result(self, success: bool, reason: str) -> EpisodeResult:
        return EpisodeResult(
            success=success,
            reason=reason,
            ended_at_ns=self.clock.now_ns(),
            metrics={
                "steps": self._steps,
                "collisions": self._collisions,
                "grasps": self._grasp_count,
                "placed": sum(item.placed for item in self.objects.values()),
            },
        )

    def _event(self, event_type: str, details: dict[str, object] | None = None) -> EpisodeEvent:
        return EpisodeEvent(
            timestamp_ns=self.clock.now_ns(),
            event_type=event_type,
            source="simulation",
            details={} if details is None else details,
        )

    def _require_state(self) -> SimRobotState:
        if self.robot is None:
            raise RuntimeError("environment is not reset")
        return self.robot


def applied_heading_rate(twist: tuple[float, float]) -> float:
    return twist[1]
